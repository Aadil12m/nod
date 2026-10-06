"""The safety fast path (design 5.5, 5.6, 4.2.2).

`SafetyChannel.on_estop` runs synchronously inside the WebSocket receive handler:

  1. push `ctl.estop` to the sim's dedicated socket FIRST (non-blocking send),
  2. then latch the session (ESTOPPED), cancel planner, Cosmos, ASR and run tasks,
     invalidate the open proposal and bump the epoch.

It never awaits, never touches the LLM, Cosmos or Token Factory. `estop.ack` (with
`lat_ms`) goes back once the sim confirms the halt (`sim.estop_ack`), or after
`estop_ack_wait_ms` without the sim fields if the sim does not answer. Duplicate
`estop_seq` values are ignored (the ack is re-sent if it already went out).

The watchdog (`watchdog_tick`, every 50 ms) relays `ctl.hb` to the sim every 100 ms only
while the phone heartbeat is fresh and no e-stop is waiting for its sim ack, and asks the
session for a protective pause on heartbeat loss (500 ms) or face loss (1000 ms) while
executing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from nod_protocol import EStop, EStopAck, EStopReset, Heartbeat, LatMs, SimEStopAck, now_ms

if TYPE_CHECKING:
    from .peers import Peer

log = logging.getLogger(__name__)

# Phone wall clocks are normally NTP-synced; if the ping/pong estimate says they are off by
# more than this, latencies are corrected with the estimate instead of trusted raw.
CLOCK_TRUST_MS = 1000


@dataclass
class HbState:
    """Phone heartbeat state shared by the safety channel and the session (gateway clock)."""

    last_hb_ms: int | None = None
    face_present: bool = False
    face_absent_since_ms: int | None = None
    track_conf: float = 0.0
    fps: float = 0.0
    latched: bool = False
    clock_offset_ms: float | None = None  # gateway minus phone, upper bound from ping samples

    def fresh(self, now: int, timeout_ms: int) -> bool:
        return self.last_hb_ms is not None and now - self.last_hb_ms <= timeout_ms


class SessionLike(Protocol):
    mode: str

    def latch_estop(self, reason: str) -> None: ...
    def reset_estop(self, msg: EStopReset) -> bool: ...
    def protective_pause(self, reason: str) -> None: ...
    def on_estop_seq(self, seq: int) -> None: ...
    def on_sim_epoch(self, epoch: int) -> None: ...


class SimLike(Protocol):
    def estop(self, seq: int) -> None: ...
    def send_hb(self, t: int, phone_ok: bool = True) -> bool: ...


@dataclass
class PendingEStop:
    gw_seq: int
    peer: Peer | None
    client_seq: int | None
    source: str
    t_rx: int  # gateway clock when the estop arrived
    t_sent: int | None  # gateway clock when ctl.estop left (None: the sim was not reachable)
    phone_gw_ms: float | None
    acked: bool = False
    ack: EStopAck | None = None
    sim_ack: SimEStopAck | None = None
    t_fwd_ms: float = 0.0  # gateway forward time (receive to send)
    extra: dict = field(default_factory=dict)


class SafetyChannel:
    def __init__(
        self, cfg, sim: SimLike, session: SessionLike, hb: HbState, *, metrics=None, presenter=None, clock=now_ms
    ) -> None:
        self.cfg = cfg
        self.sim = sim
        self.session = session
        self.hb = hb
        self.metrics = metrics
        self.presenter = presenter
        self.clock = clock
        self.gw_seq = 0  # the gateway's own e-stop counter, sent to the sim as ctl.estop.estop_seq
        self.pending: dict[int, PendingEStop] = {}
        self.handled: dict[tuple[str, int], int] = {}  # (origin, client estop_seq) -> gw_seq, while latched
        self.last_relay_ms = 0
        self.relayed = 0
        self.reset_at_ms = 0
        self.ping_samples: list[int] = []

    # ------------------------------------------------------------ e-stop

    def on_estop(self, msg: EStop, peer: Peer | None = None) -> None:
        """Synchronous fast path. Must not await anything."""
        t_rx = self.clock()
        origin = peer.id if peer is not None else f"anon:{msg.source}"
        key = (origin, msg.estop_seq)
        if key in self.handled:
            # Duplicate (the phone resends every 100 ms until acked): no side effects.
            p = self.pending.get(self.handled[key])
            if p is not None and p.ack is not None and peer is not None:
                peer.post(p.ack)
            return
        self._trigger(peer, msg.estop_seq, msg.source, t_rx, msg.ts)
        self.handled[key] = self.gw_seq

    def _trigger(
        self, peer: Peer | None, client_seq: int | None, source: str, t_rx: int, t_phone_send: int | None
    ) -> None:
        self.gw_seq += 1
        seq = self.gw_seq
        # 1. Forward to the sim FIRST.
        t_sent: int | None = None
        err = None
        try:
            self.sim.estop(seq)
            t_sent = self.clock()
        except Exception as exc:  # the stop still latches below; the sim watchdog covers the rest
            err = str(exc)
            log.error("ctl.estop not delivered: %s", exc)
        # 2. Everything else.
        p = PendingEStop(seq, peer, client_seq, source, t_rx, t_sent, self._phone_gw(t_rx, t_phone_send))
        p.t_fwd_ms = (t_sent - t_rx) if t_sent is not None else 0.0
        self.pending[seq] = p
        # Record the client's number before latching, so the ESTOPPED `state` the latch sends
        # already carries it (the phone and presenter compare `state.estop_seq` with their own).
        if client_seq is not None:
            self.session.on_estop_seq(client_seq)
        self.session.latch_estop(source)
        if self.metrics is not None:
            self.metrics.add("estop_gw_fwd_ms", p.t_fwd_ms)
        if self.presenter is not None:
            self.presenter.trace(
                "estop",
                None,
                {
                    "gw_seq": seq,
                    "client_seq": client_seq,
                    "source": source,
                    "t_rx": t_rx,
                    "forwarded": t_sent is not None,
                    "error": err,
                },
            )

    def _phone_gw(self, t_rx: int, t_phone: int | None) -> float | None:
        """One-way phone-to-gateway time, from the phone's send timestamp."""
        if t_phone is None:
            return None
        off = self.hb.clock_offset_ms
        corr = off if (off is not None and abs(off) > CLOCK_TRUST_MS) else 0.0
        v = t_rx - (t_phone + corr)
        return float(v) if -50 <= v <= 10_000 else None

    def on_sim_estop_ack(self, ack: SimEStopAck) -> None:
        p = self.pending.get(ack.estop_seq)
        if p is None or p.sim_ack is not None:
            return
        p.sim_ack = ack
        self.session.on_sim_epoch(ack.epoch)
        if p.acked:  # already acked without sim fields after the wait ran out
            return
        gw_sim = float(ack.t_rx - p.t_sent) if p.t_sent is not None else None
        sim_halt = float(ack.t_halt - ack.t_rx)
        self._ack(p, LatMs(phone_gw=p.phone_gw_ms, gw_sim=gw_sim, sim_halt=sim_halt), ack.stopped_at_step)

    def _ack(self, p: PendingEStop, lat: LatMs, stopped_at: int | None) -> None:
        p.acked = True
        if p.client_seq is not None:
            p.ack = EStopAck(estop_seq=p.client_seq, stopped_at_step=stopped_at, lat_ms=lat)
            if p.peer is not None:
                p.peer.post(p.ack)
        if self.metrics is not None:
            parts = [lat.phone_gw, lat.gw_sim, lat.sim_halt]
            self.metrics.add("estop_phone_gw_ms", lat.phone_gw)
            self.metrics.add("estop_gw_sim_ms", lat.gw_sim)
            self.metrics.add("estop_sim_halt_ms", lat.sim_halt)
            if all(v is not None for v in parts):
                self.metrics.add("estop_ms", sum(parts))  # type: ignore[arg-type]
        if self.presenter is not None:
            self.presenter.trace(
                "estop",
                None,
                {
                    "gw_seq": p.gw_seq,
                    "client_seq": p.client_seq,
                    "acked": True,
                    "lat_ms": lat.model_dump(),
                    "stopped_at_step": stopped_at,
                    "sim_confirmed": p.sim_ack is not None,
                },
            )

    def unacked(self) -> bool:
        """True while some e-stop has no `sim.estop_ack` yet (cleared by estop.reset)."""
        return any(p.sim_ack is None for p in self.pending.values())

    def on_estop_reset(self, msg: EStopReset, peer: Peer | None = None) -> bool:
        if self.session.reset_estop(msg):
            self.handled.clear()
            self.pending.clear()
            self.reset_at_ms = self.clock()
            return True
        return False

    # ------------------------------------------------------------ heartbeat and watchdog

    def on_hb(self, msg: Heartbeat, peer: Peer | None = None) -> None:
        now = self.clock()
        self.hb.last_hb_ms = now
        self.hb.track_conf, self.hb.fps = msg.track_conf, msg.fps
        if msg.face_present:
            self.hb.face_absent_since_ms = None
        elif self.hb.face_present or self.hb.face_absent_since_ms is None:
            self.hb.face_absent_since_ms = now
        self.hb.face_present = msg.face_present
        self.hb.latched = msg.latched
        # The phone says it is latched but the gateway is not (a lost estop, or a gateway restart):
        # latch too. Ignored briefly after a reset, while older heartbeats are still in flight.
        if msg.latched and self.session.mode != "ESTOPPED" and now - self.reset_at_ms > 500:
            self._trigger(peer, None, "latched_hb", now, msg.ts)

    def on_ping_sample(self, t0_phone: int, t_rx: int) -> None:
        """Keep min(t_rx - t0): gateway minus phone clock plus the smallest uplink delay."""
        self.ping_samples.append(t_rx - t0_phone)
        self.ping_samples = self.ping_samples[-20:]
        self.hb.clock_offset_ms = float(min(self.ping_samples))

    def phone_fresh(self, now: int | None = None) -> bool:
        return self.hb.fresh(self.clock() if now is None else now, self.cfg.hb_timeout_ms)

    def watchdog_tick(self) -> None:
        """Every 50 ms. Synchronous."""
        now = self.clock()
        fresh = self.phone_fresh(now)
        if fresh and not self.unacked() and now - self.last_relay_ms >= 100:
            if self.sim.send_hb(now, True):
                self.relayed += 1
            self.last_relay_ms = now
        mode = self.session.mode
        if mode in ("EXECUTING", "COUNTDOWN"):
            if not fresh:
                self.session.protective_pause("hb_lost")
            elif (
                self.hb.face_absent_since_ms is not None and now - self.hb.face_absent_since_ms >= self.cfg.face_lost_ms
            ):
                self.session.protective_pause("face_lost")
        # E-stops the sim has not acknowledged in time: ack the phone without sim fields
        # so it stops resending; the gateway stays latched and the sim gets no ctl.hb.
        for p in list(self.pending.values()):
            if not p.acked and now - p.t_rx >= self.cfg.estop_ack_wait_ms:
                self._ack(p, LatMs(phone_gw=p.phone_gw_ms, gw_sim=None, sim_halt=None), None)
