"""FastAPI app factory and runtime wiring (design 4.2.2, 4.3).

Endpoints:
  GET  /, /presenter, static        PWA and presenter page (from cfg.pwa_dir)
  WS   /ws/phone                    events and audio (3.5.1, 3.5.3)
  WS   /ws/safety                   hb, estop, estop.reset (3.5.2); separate socket so a stop
                                    never waits behind audio or proposals
  WS   /ws/presenter                trace + metrics out; estop in (3.5.8)
  GET  /healthz                     sim, policy, Cosmos, Token Factory, HA, ASR with last errors
  GET  /api/registry                skills with live stats
  GET  /api/world                   latest WorldState
  GET  /api/runs?limit=20, /api/runs/{id}
  GET  /api/metrics                 latency histograms
  POST /api/dev/inject              mock mode or NOD_DEV=1: gestures, utterances, e-stops
  GET  /api/dev/state               mock mode or NOD_DEV=1: session internals for tests
  POST /api/scene/reset             caregiver lease only: scripted table reset
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import struct
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from nod_protocol import (
    PHONE_IN,
    PRESENTER_IN,
    SAFETY_IN,
    EStop,
    EStopReset,
    Gesture,
    Heartbeat,
    Hello,
    Notice,
    Ping,
    Pong,
    SimEpisodeHeader,
    SimEStopAck,
    SimStatus,
    UiCancel,
    UiConfirm,
    UiFocus,
    UiMenu,
    UtterEnd,
    UtterStart,
    UtterText,
    Welcome,
    WorldState,
    new_id,
    now_ms,
    parse_ws,
)
from pydantic import ValidationError

from .config import Config
from .metrics import Health, MetricsHub
from .mock_runner import MockRunner
from .peers import Outbox, Peer
from .presenter import Presenter
from .recorder import Recorder
from .registry import Registry, load_registry
from .safety import HbState, SafetyChannel
from .session import Session, Utterance
from .simlink import SimError, SimLink

log = logging.getLogger(__name__)


class Gateway:
    """Everything one gateway process runs. Built by `create_app`, started by its lifespan."""

    def __init__(
        self,
        cfg: Config,
        *,
        registry: Registry | None = None,
        planner=None,
        asr=None,
        verifier=None,
        home=None,
        sim=None,
        mock_runner: MockRunner | None = None,
    ) -> None:
        self.cfg = cfg
        self.reg = registry or load_registry(cfg.registry_path, cfg.stats_path, cfg.scene_id)
        self.http = httpx.AsyncClient()
        self.out = Outbox(new_id("s"))
        self.metrics = MetricsHub()
        self.health = Health()
        self.recorder = Recorder(cfg.runs_dir)
        self.presenter = Presenter(self.out, self.recorder, self.metrics)
        self.hb = HbState()
        self.mock = mock_runner
        if self.mock is None and cfg.run_mock_runner:
            self.mock = MockRunner(
                cfg.recordings_dir,
                cfg.sim_ctl,
                cfg.sim_estop,
                cfg.sim_pub,
                cfg.sim_pose_pub,
                cfg.mock_hz,
                registry=cfg.registry_path,
                scene_id=cfg.scene_id,
                steps=cfg.mock_steps,
                failure_rate=cfg.mock_failure_rate,
                seed=cfg.mock_seed,
                hb_timeout_ms=cfg.mock_hb_timeout_ms,
            )
        self.sim = sim or SimLink(cfg.sim_ctl, cfg.sim_estop, cfg.sim_pub, ack_timeout_s=cfg.sim_ack_timeout_s)
        self.session = Session(
            cfg,
            self.reg,
            self.sim,
            planner or self._make_planner(),
            asr or self._make_asr(),
            None,
            home or self._make_home(),
            self.out,
            self.hb,
            recorder=self.recorder,
            presenter=self.presenter,
            metrics=self.metrics,
            health=self.health,
        )
        self.session.verifier = verifier if verifier is not None else self._make_verifier()
        self.safety = SafetyChannel(
            cfg,
            self.sim,
            self.session,
            self.hb,
            metrics=self.metrics,
            presenter=self.presenter,
            lease_of=self.lease_of,
        )
        self.dev_peer = Peer(None, "safety", self.out.sess)  # collects acks for /api/dev/inject e-stops
        self.dev_peer.lease = "user"  # dev injection (mock mode or NOD_DEV=1) acts as the user's phone
        self.tasks: list[asyncio.Task] = []
        self.scene_loaded = False

    # ------------------------------------------------------------ dependencies by mode

    def _make_planner(self):
        if self.cfg.planner_kind == "fake":
            from .planner.fake import FakePlanner

            return FakePlanner(self.cfg.canned_dir)
        from .planner.nemotron import NemotronPlanner

        if not self.cfg.tf_api_key:
            self.health.error("token_factory", "no API key: the deterministic menu will be used")
        return NemotronPlanner(self.cfg, self.http)

    def _make_asr(self):
        if self.cfg.asr_kind == "fake":
            from .asr.fake import FakeASR

            return FakeASR(self.cfg.fake_transcripts)
        from .asr.parakeet import ParakeetASR

        return ParakeetASR()

    def _make_verifier(self):
        kind = self.cfg.cosmos_kind
        if kind == "none":
            return None
        if kind == "fake":
            from .cosmos.fake import FakeVerifier

            return FakeVerifier(self.reg, lambda: self.session.world)
        from .cosmos.vllm import VllmVerifier

        return VllmVerifier(self.cfg, self.http)

    def _make_home(self):
        if self.cfg.home_kind == "fake":
            from .home.fake import FakeHome

            return FakeHome()
        from .home.hass import HassHome

        return HassHome(self.cfg, self.http)

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        if self.mock is not None:
            await asyncio.to_thread(self.mock.start)
        if isinstance(self.sim, SimLink):
            await self.sim.start()
        self.tasks = [
            asyncio.create_task(self._status_pump(), name="status-pump"),
            asyncio.create_task(self._watchdog(), name="watchdog"),
            asyncio.create_task(self.presenter.metrics_loop(), name="metrics"),
            asyncio.create_task(self._load_scene(), name="load-scene"),
        ]

    async def stop(self) -> None:
        for t in self.tasks + list(self.session.tasks):
            t.cancel()
        for t in self.tasks + list(self.session.tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        for p in self.out.all():
            await p.close()
        if isinstance(self.sim, SimLink):
            await self.sim.close()
        if self.mock is not None:
            await asyncio.to_thread(self.mock.stop)
        await self.http.aclose()

    async def _load_scene(self) -> None:
        delay = 0.05
        while True:
            try:
                await self.sim.load_scene(self.cfg.scene_id, self.cfg.init_index, self.session.epoch)
                self.session.world = await self.sim.world()
                self.scene_loaded = True
                self.health.ok("sim", scene_id=self.cfg.scene_id)
                return
            except SimError as exc:
                self.health.error("sim", str(exc))
                await asyncio.sleep(delay)
                delay = min(2.0, delay * 2)

    async def _status_pump(self) -> None:
        async for item in self.sim.status():
            try:
                if isinstance(item, SimEStopAck):
                    self.safety.on_sim_estop_ack(item)
                elif isinstance(item, WorldState):
                    self.session.on_world(item)
                elif isinstance(item, SimStatus):
                    await self.session.on_sim_status(item)
                elif isinstance(item, SimEpisodeHeader):
                    self.health.ok("sim", model_hash=item.model_hash)
            except Exception:
                log.exception("status pump")

    async def _watchdog(self) -> None:
        while True:
            try:
                self.safety.watchdog_tick()
            except Exception:
                log.exception("watchdog tick")
            await asyncio.sleep(0.05)

    # ------------------------------------------------------------ message dispatch

    def lease_of(self, peer: Peer | None, token: str | None) -> str:
        """The lease of the phone session a /ws/safety message belongs to (by resume_token).

        A safety socket has no hello of its own: the phone puts the token from its `welcome`
        in every `hb` and `estop.reset`. Unknown or missing tokens get "none"."""
        if peer is self.dev_peer:
            return self.dev_peer.lease
        if not token:
            return "none"
        for p in self.out.phones:
            if p.resume_token == token and not p.closed:
                return p.lease
        return "none"

    def welcome(self, peer: Peer, hello: Hello) -> None:
        if hello.role == "caregiver":
            peer.lease = "caregiver"
        else:
            # One user lease: the newest user phone takes it over (single user, 2.2).
            for other in self.out.phones:
                if other is not peer and other.lease == "user":
                    other.lease = "none"
                    other.post(Notice(level="warn", text="Another device took over", speak=False))
            peer.lease = "user"
        peer.resume_token = new_id("rt")
        peer.post(Welcome(resume_token=peer.resume_token, lease=peer.lease, server_time=now_ms()))
        peer.post(self.session.state_msg())
        if self.session.live is not None:
            peer.post(self.session.live.proposal)

    async def on_phone_msg(self, peer: Peer, msg: Any) -> None:
        if msg.seq and msg.seq <= peer.last_in_seq and not isinstance(msg, Ping):
            return  # duplicate (seq increases per sender per socket)
        peer.last_in_seq = max(peer.last_in_seq, msg.seq)
        s = self.session
        if isinstance(msg, Hello):
            self.welcome(peer, msg)
        elif isinstance(msg, Ping):
            self.safety.on_ping_sample(msg.t0, now_ms())
            peer.post(Pong(t0=msg.t0, t_srv=now_ms()))
        elif peer.lease == "none":
            peer.post(Notice(level="warn", text="This device does not hold the session", speak=False))
        elif isinstance(msg, Gesture):
            self.presenter.trace("run", None, {"gesture": msg.name, "phase": msg.phase, "held_ms": msg.held_ms})
        elif isinstance(msg, UiMenu):
            await s.on_menu(msg.source)
        elif isinstance(msg, UiFocus):
            await s.on_focus(msg)
        elif isinstance(msg, UiConfirm):
            await s.on_confirm(msg, peer.lease)
        elif isinstance(msg, UiCancel):
            await s.on_cancel(msg)
        elif isinstance(msg, UtterStart):
            s.on_utter_start(msg.utt_id, msg.sr)
        elif isinstance(msg, UtterEnd):
            await s.on_utter_end(msg.utt_id)
        elif isinstance(msg, UtterText):
            await s.on_utterance(Utterance(utt_id=msg.utt_id, text=msg.text))

    def on_safety_msg(self, peer: Peer | None, msg: Any) -> None:
        """Synchronous on purpose: the e-stop path must not await."""
        if isinstance(msg, EStop):
            self.safety.on_estop(msg, peer)
        elif isinstance(msg, Heartbeat):
            self.safety.on_hb(msg, peer)
        elif isinstance(msg, EStopReset):
            self.safety.on_estop_reset(msg, peer)
        elif isinstance(msg, Ping) and peer is not None:
            self.safety.on_ping_sample(msg.t0, now_ms())
            peer.post(Pong(t0=msg.t0, t_srv=now_ms()))


def _bad(peer: Peer, exc: Exception) -> None:
    peer.post(Notice(level="warn", text=f"Bad message: {str(exc)[:120]}", speak=False))


def create_app(cfg: Config | None = None, **overrides: Any) -> FastAPI:
    cfg = cfg or Config()
    gw = Gateway(cfg, **overrides)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await gw.start()
        try:
            yield
        finally:
            await gw.stop()

    app = FastAPI(title="Nod gateway", version="0.1.0", lifespan=lifespan)
    app.state.gw = gw

    # ------------------------------------------------------------ WebSockets

    @app.websocket("/ws/phone")
    async def ws_phone(ws: WebSocket) -> None:
        await ws.accept()
        peer = Peer(ws, "phone", gw.out.sess)
        peer.start()
        gw.out.phones.add(peer)
        try:
            while True:
                m = await ws.receive()
                if m["type"] == "websocket.disconnect":
                    break
                if m.get("bytes") is not None:
                    data = m["bytes"]
                    if len(data) >= 4 and peer.lease != "none":
                        (utt_id,) = struct.unpack_from("<I", data, 0)
                        gw.session.on_audio(utt_id, data[4:])
                    continue
                try:
                    msg = parse_ws(m.get("text") or "", PHONE_IN)
                except ValidationError as exc:
                    _bad(peer, exc)
                    continue
                await gw.on_phone_msg(peer, msg)
        except WebSocketDisconnect:
            pass
        finally:
            gw.out.phones.discard(peer)
            await peer.close()

    @app.websocket("/ws/safety")
    async def ws_safety(ws: WebSocket) -> None:
        await ws.accept()
        peer = Peer(ws, "safety", gw.out.sess)
        peer.start()
        gw.out.safety.add(peer)
        peer.post(gw.session.state_msg())
        try:
            while True:
                text = await ws.receive_text()
                try:
                    msg = parse_ws(text, SAFETY_IN)
                except ValidationError as exc:
                    _bad(peer, exc)
                    continue
                gw.on_safety_msg(peer, msg)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            gw.out.safety.discard(peer)
            await peer.close()

    @app.websocket("/ws/presenter")
    async def ws_presenter(ws: WebSocket) -> None:
        await ws.accept()
        peer = Peer(ws, "presenter", gw.out.sess)
        peer.start()
        gw.out.presenters.add(peer)
        peer.post(gw.session.state_msg())
        peer.post(gw.metrics.message())
        try:
            while True:
                text = await ws.receive_text()
                try:
                    msg = parse_ws(text, PRESENTER_IN)
                except ValidationError as exc:
                    _bad(peer, exc)
                    continue
                gw.on_safety_msg(peer, msg)  # presenter may send STOP only (and ping)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            gw.out.presenters.discard(peer)
            await peer.close()

    # ------------------------------------------------------------ HTTP API

    @app.get("/healthz")
    async def healthz() -> dict:
        h = gw.health.view()
        sim_ok = isinstance(gw.sim, SimLink) and gw.sim.connected
        h["sim"].update({"connected": sim_ok, "scene_loaded": gw.scene_loaded})
        h["policy"]["note"] = "reported through sim.status" if cfg.mode == "live" else "mock: no policy"
        return {
            "ok": bool(sim_ok and gw.scene_loaded),
            "mode": cfg.mode,
            "session": gw.session.mode,
            "placeholder_stats": gw.reg.placeholder_stats,
            "components": h,
            "deps": {
                "planner": cfg.planner_kind,
                "asr": cfg.asr_kind,
                "cosmos": cfg.cosmos_kind,
                "home": cfg.home_kind,
                "mock_runner": gw.mock is not None,
            },
        }

    @app.get("/api/registry")
    async def api_registry() -> dict:
        stats = gw.reg.stats
        return {
            "scene_id": gw.reg.scene_id,
            "placeholder_stats": gw.reg.placeholder_stats,
            "stats_job_id": stats.job_id if stats else None,
            "stats_at": stats.at if stats else None,
            "chains": {k: v.model_dump(mode="json") for k, v in (stats.chains if stats else {}).items()},
            "skills": gw.reg.public_view(),
        }

    @app.get("/api/world")
    async def api_world() -> JSONResponse:
        w = gw.session.world
        if w is None:
            return JSONResponse({"error": "no world state yet"}, status_code=503)
        return JSONResponse(w.model_dump(mode="json"))

    @app.get("/api/runs")
    async def api_runs(limit: int = 20) -> dict:
        return {"runs": gw.recorder.list(max(1, min(limit, 200)))}

    @app.get("/api/runs/{run_id}")
    async def api_run(run_id: str) -> dict:
        rec = gw.recorder.get(run_id)
        if rec is None:
            raise HTTPException(404, "unknown run")
        return rec

    @app.get("/api/metrics")
    async def api_metrics() -> dict:
        return {
            "histograms": gw.metrics.summary(),
            "summary": gw.metrics.message().model_dump(exclude={"v", "type", "id", "ts", "sess", "seq"}),
            "hb_relayed": gw.safety.relayed,
        }

    @app.post("/api/scene/reset")
    async def api_scene_reset(request: Request) -> dict:
        token = request.headers.get("x-nod-token")
        caregiver = any(p.lease == "caregiver" and p.resume_token == token for p in gw.out.phones) if token else False
        if not (caregiver or cfg.dev_enabled):
            raise HTTPException(403, "caregiver lease only")
        if not gw.session.start_reset_scene():
            raise HTTPException(409, f"cannot reset in mode {gw.session.mode}")
        return {"ok": True}

    # ------------------------------------------------------------ dev

    def dev_state() -> dict:
        s = gw.session
        return {
            "mode": s.mode,
            "epoch": s.epoch,
            "caregiver_enabled": s.caregiver_enabled,
            "proposal": s.live.proposal.model_dump(mode="json") if s.live else None,
            "purpose": s.live.purpose if s.live else None,
            "run": {
                "run_id": s.run.run_id,
                "skill_id": s.run.spec.skill_id,
                "phase": s.run.phase,
                "sim_step": s.run.sim_step,
            }
            if s.run
            else None,
            "plan": {"skill_ids": s.plan.skill_ids, "step": s.plan.step} if s.plan else None,
            "world": s.world.model_dump(mode="json") if s.world else None,
            "hb_fresh": gw.safety.phone_fresh(),
            "face_present": gw.hb.face_present,
            "dev_acks": [m.model_dump(mode="json") for m in gw.dev_peer.sent[-10:]],
            "last_messages": [m.model_dump(mode="json") for m in gw.out.log[-30:]],
        }

    if cfg.dev_enabled:

        @app.get("/api/dev/state")
        async def api_dev_state() -> dict:
            return dev_state()

        @app.post("/api/dev/inject")
        async def api_dev_inject(body: dict) -> dict:
            await _inject(gw, body)
            return {"ok": True, **dev_state()}

    else:

        @app.api_route("/api/dev/{rest:path}", methods=["GET", "POST"])
        async def api_dev_disabled(rest: str) -> JSONResponse:
            return JSONResponse({"error": "dev endpoints are off (live mode without NOD_DEV=1)"}, status_code=404)

    # ------------------------------------------------------------ static PWA

    pwa = Path(cfg.pwa_dir)
    if (pwa / "presenter.html").exists():

        @app.get("/presenter")
        async def presenter_page() -> FileResponse:
            return FileResponse(pwa / "presenter.html")

    if pwa.is_dir():
        app.mount("/", StaticFiles(directory=pwa, html=True), name="pwa")
    else:

        @app.get("/")
        async def no_pwa() -> HTMLResponse:
            return HTMLResponse(
                f"<h1>Nod gateway ({cfg.mode})</h1><p>PWA directory {pwa} not found. "
                "See <a href='/healthz'>/healthz</a> and <a href='/docs'>/docs</a>.</p>"
            )

    return app


async def _inject(gw: Gateway, body: dict) -> None:
    """Dev/test injection (4.3). Messages go through the same handlers as the sockets."""
    kind = body.get("kind")
    s = gw.session
    lp = s.live
    now = now_ms()
    if kind == "menu":
        await s.on_menu(body.get("source", "dev"))
    elif kind == "utterance":
        await s.on_utterance(Utterance(utt_id=int(body.get("utt_id", 0)), text=str(body["text"])))
    elif kind == "audio":
        await s.on_utterance(
            Utterance(
                utt_id=int(body.get("utt_id", 0)), pcm=base64.b64decode(body["pcm_b64"]), sr=int(body.get("sr", 16000))
            )
        )
    elif kind == "focus":
        if lp is None:
            raise HTTPException(409, "no live proposal")
        await s.on_focus(UiFocus(proposal_id=lp.proposal.proposal_id, option_id=body["option_id"]))
    elif kind == "confirm":
        if lp is None:
            raise HTTPException(409, "no live proposal")
        held = int(body.get("held_ms", 750))
        c = UiConfirm(
            proposal_id=body.get("proposal_id", lp.proposal.proposal_id),
            option_id=body.get("option_id", lp.focus_option or ""),
            nonce=body.get("nonce", lp.proposal.nonce),
            gesture=body.get("gesture", "long_blink"),
            held_ms=held,
            shown_at=int(body.get("shown_at", lp.sent_ms)),
            focused_at=int(body.get("focused_at", lp.focus_since_ms or lp.sent_ms)),
            ts=now,
        )
        await s.on_confirm(c, body.get("lease", "user"))
    elif kind == "cancel":
        await s.on_cancel(UiCancel(proposal_id=lp.proposal.proposal_id if lp else None, source="dev"))
    elif kind == "estop":
        gw.on_safety_msg(
            gw.dev_peer,
            EStop(estop_seq=int(body.get("estop_seq", 1)), source=body.get("source", "button"), t_detect=now, ts=now),
        )
    elif kind == "estop_reset":
        gw.on_safety_msg(
            gw.dev_peer, EStopReset(estop_seq=int(body.get("estop_seq", 1)), gesture=body.get("gesture", "touch"))
        )
    elif kind == "hb":
        gw.on_safety_msg(
            gw.dev_peer,
            Heartbeat(
                face_present=bool(body.get("face_present", True)),
                track_conf=float(body.get("track_conf", 0.95)),
                fps=float(body.get("fps", 24)),
                latched=bool(body.get("latched", False)),
            ),
        )
    elif kind == "caregiver":
        s.set_caregiver(bool(body.get("enabled", True)))
    elif kind == "msg":
        sock = body.get("socket", "phone")
        data = body["data"]
        import json as _json

        text = _json.dumps(data)
        try:
            if sock == "safety":
                gw.on_safety_msg(gw.dev_peer, parse_ws(text, SAFETY_IN))
            else:
                await gw.on_phone_msg(gw.dev_peer, parse_ws(text, PHONE_IN))
        except ValidationError as exc:
            raise HTTPException(422, str(exc)) from exc
    else:
        raise HTTPException(400, f"unknown kind {kind!r}")
