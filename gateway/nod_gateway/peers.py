"""WebSocket peers and the outbox the session and safety layer send through.

Sending never blocks the caller: `Peer.post` stamps the envelope (`sess`, per-socket `seq`)
and puts the JSON text on a queue that a writer task drains. The e-stop fast path can
therefore post `estop.ack` without awaiting anything.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Literal

from fastapi import WebSocket
from nod_protocol import Envelope, new_id, to_ws

log = logging.getLogger(__name__)

PeerKind = Literal["phone", "safety", "presenter"]


class Peer:
    def __init__(self, ws: WebSocket | None, kind: PeerKind, sess: str, maxsize: int = 2000) -> None:
        self.ws = ws
        self.kind = kind
        self.sess = sess
        self.id = new_id("c")
        self.seq = 0
        self.lease: Literal["user", "caregiver", "none"] = "none"
        self.resume_token: str | None = None
        self.last_in_seq = 0
        self.queue: asyncio.Queue[str | bytes | None] = asyncio.Queue(maxsize=maxsize)
        self.sent: list[Envelope] = []  # kept only when ws is None (tests)
        self.closed = False
        self._writer: asyncio.Task | None = None

    def start(self) -> None:
        if self.ws is not None:
            self._writer = asyncio.create_task(self._write(), name=f"writer-{self.kind}")

    def post(self, msg: Envelope) -> None:
        if self.closed:
            return
        self.seq += 1
        out = msg.model_copy(update={"sess": self.sess, "seq": self.seq})
        if self.ws is None:
            self.sent.append(out)
            return
        self._put(to_ws(out))

    def post_bytes(self, data: bytes) -> None:
        if not self.closed and self.ws is not None:
            self._put(data)

    def _put(self, item: str | bytes) -> None:
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            log.warning("%s peer %s is not reading; dropping a message", self.kind, self.id)

    async def _write(self) -> None:
        assert self.ws is not None
        try:
            while True:
                item = await self.queue.get()
                if item is None:
                    return
                if isinstance(item, bytes):
                    await self.ws.send_bytes(item)
                else:
                    await self.ws.send_text(item)
        except Exception as exc:  # socket closed under us
            log.debug("writer for %s stopped: %s", self.id, exc)
            self.closed = True

    async def close(self) -> None:
        self.closed = True
        if self._writer is not None:
            self._writer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._writer


class Outbox:
    """Fan-out: gateway-to-phone messages go to every phone socket and are mirrored to presenters."""

    def __init__(self, sess: str) -> None:
        self.sess = sess
        self.phones: set[Peer] = set()
        self.safety: set[Peer] = set()
        self.presenters: set[Peer] = set()
        self.log: list[Envelope] = []  # last messages to the phone (tests and debugging)

    def to_phone(self, msg: Envelope) -> None:
        self.log.append(msg)
        if len(self.log) > 500:
            del self.log[:100]
        for p in list(self.phones):
            p.post(msg)
        for p in list(self.presenters):
            p.post(msg)

    def to_presenter(self, msg: Envelope) -> None:
        for p in list(self.presenters):
            p.post(msg)

    def to_peer(self, peer: Peer | None, msg: Envelope) -> None:
        if peer is not None:
            peer.post(msg)

    def user_peer(self) -> Peer | None:
        return next((p for p in self.phones if p.lease == "user" and not p.closed), None)

    def all(self) -> list[Peer]:
        return [*self.phones, *self.safety, *self.presenters]


def safe_dict(model: Any) -> dict:
    return model.model_dump(mode="json") if hasattr(model, "model_dump") else dict(model)
