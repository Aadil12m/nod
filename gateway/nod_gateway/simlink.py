"""Gateway side of the sim link (design 3.3, 3.5.4, 3.5.5, 4.2.2).

Sockets:
  ctl    DEALER -> sim ROUTER (5560)   requests; replies come back on the same socket:
                                       `sim.ack` (matched by req_id), `sim.frames` (req_id) and
                                       `sim.world` (answer to `ctl.world`, matched in order)
  estop  PUSH   -> sim PULL   (5561)   `ctl.estop` only, sent synchronously and non-blocking
  status SUB    <- sim PUB    (5562)   `sim.status`, `sim.estop_ack`, `sim.world`, `sim.episode_header`

`IMMEDIATE` is set on the outgoing sockets so nothing queues while the sim is down: a
command sent then would otherwise run whenever the sim came back.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import AsyncIterator

import zmq
import zmq.asyncio
from nod_protocol import (
    SIM_OUT,
    Abort,
    Capture,
    CtlEStop,
    CtlHeartbeat,
    Frame,
    LoadScene,
    Pause,
    Resume,
    RunScripted,
    RunSkill,
    SimAck,
    SimEpisodeHeader,
    SimEStopAck,
    SimFrames,
    SimStatus,
    SimWorld,
    WorldRequest,
    WorldState,
    dumps,
    loads,
    new_id,
    now_ms,
)

log = logging.getLogger(__name__)

StatusItem = SimStatus | SimEStopAck | WorldState | SimEpisodeHeader


class _StatusIter:
    """Async iterator over the status queue. Unlike an async generator, cancelling one
    `__anext__` (e.g. by `wait_for`) does not end the iteration."""

    def __init__(self, queue: asyncio.Queue[StatusItem]) -> None:
        self._queue = queue

    def __aiter__(self) -> _StatusIter:
        return self

    async def __anext__(self) -> StatusItem:
        return await self._queue.get()


class SimError(Exception):
    """The sim refused a command, did not answer in time, or is not connected."""


class SimLink:
    def __init__(self, ctl: str, estop: str, pub: str, *, ack_timeout_s: float = 3.0) -> None:
        self.addrs = {"ctl": ctl, "estop": estop, "pub": pub}
        self.ack_timeout_s = ack_timeout_s
        self._actx = zmq.asyncio.Context()
        self._sctx = zmq.Context()  # sync context for the e-stop socket: no event loop involved
        self._ctl: zmq.asyncio.Socket | None = None
        self._sub: zmq.asyncio.Socket | None = None
        self._estop: zmq.Socket | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._world_waiters: deque[asyncio.Future] = deque()
        self._queue: asyncio.Queue[StatusItem] = asyncio.Queue(maxsize=10_000)
        self._tasks: list[asyncio.Task] = []
        self.latest_world: WorldState | None = None
        self.last_seen_ms: int | None = None
        self.last_error: str | None = None
        self.estop_sent = 0

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        ctl = self._actx.socket(zmq.DEALER)
        ctl.setsockopt(zmq.LINGER, 0)
        ctl.setsockopt(zmq.IMMEDIATE, 1)
        ctl.connect(self.addrs["ctl"])
        sub = self._actx.socket(zmq.SUB)
        sub.setsockopt(zmq.LINGER, 0)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        sub.connect(self.addrs["pub"])
        est = self._sctx.socket(zmq.PUSH)
        est.setsockopt(zmq.LINGER, 0)
        est.setsockopt(zmq.IMMEDIATE, 1)
        est.setsockopt(zmq.SNDHWM, 100)
        est.connect(self.addrs["estop"])
        self._ctl, self._sub, self._estop = ctl, sub, est
        self._tasks = [
            asyncio.create_task(self._read_ctl(), name="simlink-ctl"),
            asyncio.create_task(self._read_sub(), name="simlink-sub"),
        ]

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        for s in (self._ctl, self._sub, self._estop):
            if s is not None:
                s.close(0)
        self._actx.term()
        self._sctx.term()

    @property
    def connected(self) -> bool:
        return self.last_seen_ms is not None and now_ms() - self.last_seen_ms < 3000

    # ------------------------------------------------------------ e-stop fast path

    def estop(self, seq: int) -> None:
        """Send `ctl.estop` on the dedicated socket. Synchronous, never blocks, never awaits."""
        if self._estop is None:
            self.last_error = "estop socket not started"
            raise SimError(self.last_error)
        try:
            self._estop.send(dumps(CtlEStop(estop_seq=seq, t_gw=now_ms())), zmq.NOBLOCK)
            self.estop_sent += 1
        except zmq.Again as exc:
            self.last_error = "sim not connected: ctl.estop not delivered"
            raise SimError(self.last_error) from exc

    # ------------------------------------------------------------ ctl socket

    def _send_now(self, msg) -> None:
        if self._ctl is None:
            raise SimError("sim link not started")
        fut = self._ctl.send(dumps(msg), zmq.DONTWAIT)
        exc = fut.exception() if fut.done() else None
        if isinstance(exc, zmq.Again):
            self.last_error = "sim not connected"
            raise SimError(self.last_error)
        if exc is not None:
            raise SimError(str(exc))

    def send_hb(self, t: int, phone_ok: bool = True) -> bool:
        """Relay one `ctl.hb` (no reply expected). Returns False if the sim is not connected."""
        try:
            self._send_now(CtlHeartbeat(t=t, phone_ok=phone_ok))
            return True
        except SimError:
            return False

    async def _request(self, msg, timeout: float | None = None):
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[msg.req_id] = fut
        try:
            self._send_now(msg)
            return await asyncio.wait_for(fut, timeout or self.ack_timeout_s)
        except TimeoutError as exc:
            self.last_error = f"no reply to {msg.type}"
            raise SimError(self.last_error) from exc
        finally:
            self._pending.pop(msg.req_id, None)

    async def _acked(self, msg) -> None:
        reply = await self._request(msg)
        if not isinstance(reply, SimAck):
            raise SimError(f"unexpected reply {reply.type} to {msg.type}")
        if not reply.ok:
            raise SimError(f"sim refused {msg.type}: {reply.error}")

    async def load_scene(self, scene_id: str, init_index: int, epoch: int = 0) -> None:
        await self._acked(LoadScene(req_id=new_id("q"), scene_id=scene_id, init_index=init_index, epoch=epoch))

    async def run(self, cmd: RunSkill | RunScripted) -> None:
        await self._acked(cmd)

    async def pause(self, epoch: int) -> None:
        await self._acked(Pause(req_id=new_id("q"), epoch=epoch))

    async def resume(self, epoch: int) -> None:
        await self._acked(Resume(req_id=new_id("q"), epoch=epoch))

    async def abort(self, epoch: int) -> None:
        await self._acked(Abort(req_id=new_id("q"), epoch=epoch))

    async def capture(self, cams: list[str], w: int, h: int) -> list[Frame]:
        reply = await self._request(Capture(req_id=new_id("q"), cameras=cams, w=w, h=h))
        if not isinstance(reply, SimFrames):
            if isinstance(reply, SimAck) and not reply.ok:
                raise SimError(f"capture failed: {reply.error}")
            raise SimError(f"unexpected reply {reply.type} to ctl.capture")
        return [f.model_copy(update={"step": f.step if f.step is not None else reply.step}) for f in reply.frames]

    async def world(self, timeout: float | None = None) -> WorldState:
        """Ask for a fresh world; the answer is the next `sim.world` on either socket."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._world_waiters.append(fut)
        try:
            self._send_now(WorldRequest(req_id=new_id("q")))
            return await asyncio.wait_for(fut, timeout or self.ack_timeout_s)
        except (TimeoutError, SimError) as exc:
            if self.latest_world is not None:
                return self.latest_world
            raise SimError("no world state from the sim") from exc
        finally:
            with contextlib.suppress(ValueError):
                self._world_waiters.remove(fut)

    def status(self) -> AsyncIterator[StatusItem]:
        """Status, e-stop acks, world states and episode headers, in arrival order. One consumer."""

        return _StatusIter(self._queue)

    # ------------------------------------------------------------ readers

    def _on_world(self, w: SimWorld) -> WorldState:
        world = w.to_world()
        self.latest_world = world
        while self._world_waiters:
            fut = self._world_waiters.popleft()
            if not fut.done():
                fut.set_result(world)
        return world

    def _enqueue(self, item: StatusItem) -> None:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:  # nobody consuming; drop the oldest
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            self._queue.put_nowait(item)

    async def _read_ctl(self) -> None:
        assert self._ctl is not None
        while True:
            data = await self._ctl.recv()
            try:
                msg = loads(data, SIM_OUT)
            except Exception as exc:  # malformed: log and keep reading
                log.warning("bad message on ctl socket: %s", exc)
                continue
            self.last_seen_ms = now_ms()
            if isinstance(msg, (SimAck, SimFrames)):
                fut = self._pending.get(msg.req_id)
                if fut is not None and not fut.done():
                    fut.set_result(msg)
            elif isinstance(msg, SimWorld):
                self._enqueue(self._on_world(msg))
            elif isinstance(msg, (SimStatus, SimEStopAck, SimEpisodeHeader)):
                self._enqueue(msg)

    async def _read_sub(self) -> None:
        assert self._sub is not None
        while True:
            data = await self._sub.recv()
            try:
                msg = loads(data, SIM_OUT)
            except Exception as exc:
                log.warning("bad message on status socket: %s", exc)
                continue
            self.last_seen_ms = now_ms()
            if isinstance(msg, SimWorld):
                self._enqueue(self._on_world(msg))
            elif isinstance(msg, (SimStatus, SimEStopAck, SimEpisodeHeader)):
                self._enqueue(msg)
