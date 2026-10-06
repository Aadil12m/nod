"""A real gateway (uvicorn on a random port) with an in-process MockRunner, plus phone,
safety and presenter WebSocket clients. Used by the end-to-end and API tests."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable
from typing import Any

import httpx
import uvicorn
import websockets
from conftest import free_port, sim_addrs
from nod_protocol import now_ms

from nod_gateway.app import create_app
from nod_gateway.config import Config


def fast_config(tmp_path, **kw: Any) -> Config:
    a = sim_addrs()
    base: dict[str, Any] = dict(
        mode="mock",
        sim_ctl=a["ctl"],
        sim_estop=a["estop"],
        sim_pub=a["pub"],
        sim_pose_pub=a["pose"],
        runs_dir=tmp_path / "runs",
        mock_hz=100,
        mock_steps=30,
        countdown_s=0.2,
        recordings_dir=tmp_path / "none",
    )
    base.update(kw)
    return Config(**base)


class WsClient:
    """Collects every message; `wait` finds the first new one matching a predicate."""

    def __init__(self, ws, name: str) -> None:
        self.ws = ws
        self.name = name
        self.msgs: list[dict] = []
        self.cursor = 0
        self.seq = 0
        self._new = asyncio.Event()
        self._task = asyncio.create_task(self._read())

    async def _read(self) -> None:
        with contextlib.suppress(websockets.ConnectionClosed, asyncio.CancelledError):
            async for raw in self.ws:
                if isinstance(raw, str):
                    self.msgs.append(json.loads(raw))
                    self._new.set()

    async def send(self, type_: str, **fields: Any) -> dict:
        self.seq += 1
        msg = {"v": 1, "type": type_, "ts": now_ms(), "seq": self.seq, "sess": None, **fields}
        await self.ws.send(json.dumps(msg))
        return msg

    async def wait(self, pred: Callable[[dict], bool], timeout: float = 5.0, *, from_start: bool = False) -> dict:
        deadline = time.monotonic() + timeout
        i = 0 if from_start else self.cursor
        while True:
            while i < len(self.msgs):
                m = self.msgs[i]
                i += 1
                if pred(m):
                    self.cursor = i
                    return m
            left = deadline - time.monotonic()
            if left <= 0:
                tail = [(m["type"], m.get("phase") or m.get("mode") or m.get("text")) for m in self.msgs[-15:]]
                raise AssertionError(f"{self.name}: no matching message in {timeout}s; last: {tail}")
            self._new.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._new.wait(), left)

    async def wait_type(self, type_: str, timeout: float = 5.0, **match: Any) -> dict:
        return await self.wait(lambda m: m["type"] == type_ and all(m.get(k) == v for k, v in match.items()), timeout)

    def of_type(self, type_: str) -> list[dict]:
        return [m for m in self.msgs if m["type"] == type_]

    async def close(self) -> None:
        self._task.cancel()
        with contextlib.suppress(Exception):
            await self.ws.close()


class Rig:
    def __init__(self, cfg: Config, **overrides: Any) -> None:
        self.cfg = cfg
        self.app = create_app(cfg, **overrides)
        self.gw = self.app.state.gw
        self.port = free_port()
        self.server = uvicorn.Server(
            uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning", ws_ping_interval=None)
        )
        self.http: httpx.AsyncClient | None = None
        self.clients: list[WsClient] = []
        self.hb_task: asyncio.Task | None = None
        self.hb_on = True
        self.face = True
        self.latched = False
        self.token: str | None = None  # resume_token of the user-lease phone (from its welcome)

    @property
    def mock(self):
        return self.gw.mock

    async def __aenter__(self) -> Rig:
        self._serve = asyncio.create_task(self.server.serve())
        while not self.server.started:  # noqa: ASYNC110 - uvicorn exposes a flag, not an event
            await asyncio.sleep(0.02)
        self.http = httpx.AsyncClient(base_url=f"http://127.0.0.1:{self.port}")
        for _ in range(200):  # scene loaded by the gateway
            if self.gw.scene_loaded and self.gw.session.world is not None:
                break
            await asyncio.sleep(0.02)
        return self

    async def __aexit__(self, *exc) -> None:
        if self.hb_task:
            self.hb_task.cancel()
        for c in self.clients:
            await c.close()
        if self.http:
            await self.http.aclose()
        self.server.should_exit = True
        await self._serve

    async def connect(self, path: str, name: str | None = None) -> WsClient:
        ws = await websockets.connect(f"ws://127.0.0.1:{self.port}{path}", max_size=None)
        c = WsClient(ws, name or path)
        self.clients.append(c)
        return c

    async def phone(self, role: str = "user") -> WsClient:
        c = await self.connect("/ws/phone", "phone")
        await c.send(
            "hello",
            device="test",
            app_ver="0",
            role=role,
            resume_token=None,
            calib={"profile_id": "cal_1", "long_blink_ms": 700, "fps": 24},
        )
        w = await c.wait_type("welcome")
        c.token = w["resume_token"]
        if role == "user":
            self.token = w["resume_token"]  # the user's safety socket sends it in hb and estop.reset
        return c

    async def safety(self, heartbeat: bool = True) -> WsClient:
        c = await self.connect("/ws/safety", "safety")
        if heartbeat:
            self.hb_task = asyncio.create_task(self._hb(c))
            await asyncio.sleep(0.25)
        return c

    async def _hb(self, c: WsClient) -> None:
        with contextlib.suppress(Exception):
            while True:
                if self.hb_on:
                    await c.send(
                        "hb",
                        face_present=self.face,
                        track_conf=0.95,
                        fps=24,
                        latched=self.latched,
                        resume_token=self.token,
                    )
                await asyncio.sleep(0.1)

    async def inject(self, kind: str, **body: Any) -> dict:
        assert self.http is not None
        r = await self.http.post("/api/dev/inject", json={"kind": kind, **body})
        assert r.status_code == 200, r.text
        return r.json()

    async def state(self) -> dict:
        assert self.http is not None
        return (await self.http.get("/api/dev/state")).json()

    async def until(self, pred: Callable[[dict], bool], timeout: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            s = await self.state()
            if pred(s):
                return s
            if time.monotonic() > deadline:
                raise AssertionError(f"state never matched: mode={s['mode']} run={s['run']}")
            await asyncio.sleep(0.03)

    async def proposal(self, phone: WsClient, timeout: float = 5.0, **match: Any) -> dict:
        return await phone.wait(
            lambda m: m["type"] == "proposal" and all(m.get(k) == v for k, v in match.items()), timeout
        )

    async def confirm(
        self, phone: WsClient, p: dict, option_id: str, *, focus: bool = True, held_ms: int = 750
    ) -> None:
        """Phone-side confirm: focus the option, let the arm delay and dwell pass, long-blink."""
        shown = now_ms()
        if focus and option_id != p.get("focus"):
            await phone.send("ui.focus", proposal_id=p["proposal_id"], option_id=option_id)
        focused = now_ms()
        await asyncio.sleep(0.4)
        await phone.send(
            "ui.confirm",
            proposal_id=p["proposal_id"],
            option_id=option_id,
            nonce=p["nonce"],
            gesture="long_blink",
            held_ms=held_ms,
            shown_at=shown - 100,
            focused_at=focused,
        )
