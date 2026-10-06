"""HTTP endpoints (design 4.3), phone socket handling, static PWA serving, config and CLI."""

from __future__ import annotations

import asyncio

import pytest
from conftest import sim_addrs
from harness import Rig, fast_config

from nod_gateway import cli
from nod_gateway.config import Config


async def test_healthz_registry_world_metrics(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        h = (await rig.http.get("/healthz")).json()
        assert h["ok"] is True and h["mode"] == "mock" and h["session"] == "IDLE"
        assert set(h["components"]) == {"sim", "policy", "cosmos", "token_factory", "hass", "asr"}
        assert h["components"]["sim"]["connected"] and h["placeholder_stats"] is True
        assert h["deps"] == {"planner": "fake", "asr": "fake", "cosmos": "fake", "home": "fake", "mock_runner": True}
        reg = (await rig.http.get("/api/registry")).json()
        assert len(reg["skills"]) == 18 and reg["placeholder_stats"] is True
        bop = next(s for s in reg["skills"] if s["skill_id"] == "bowl_on_plate")
        assert bop["stats"]["alone"]["n"] >= 20  # live stats merged
        assert set(reg["chains"]) == {"A", "B"}
        w = (await rig.http.get("/api/world")).json()
        assert w["scene_id"] == "libero_goal_v1" and w["atoms"]
        m = (await rig.http.get("/api/metrics")).json()
        assert "estop_ms" in m["histograms"] and "estop_ms_p50" in m["summary"]
        assert (await rig.http.get("/api/runs")).json() == {"runs": []}
        assert (await rig.http.get("/api/runs/r_missing")).status_code == 404


async def test_dev_inject_errors(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        r = await rig.http.post("/api/dev/inject", json={"kind": "nope"})
        assert r.status_code == 400
        r = await rig.http.post("/api/dev/inject", json={"kind": "confirm", "option_id": "o1"})
        assert r.status_code == 409  # no live proposal
        r = await rig.http.post("/api/dev/inject", json={"kind": "msg", "socket": "phone", "data": {"type": "bogus"}})
        assert r.status_code == 422


async def test_dev_inject_raw_messages_and_estop(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        await rig.inject("hb", face_present=True)
        s = await rig.inject("msg", socket="phone", data={"type": "ui.menu", "source": "keyboard"})
        assert s["mode"] == "AWAITING_CONFIRM"
        s = await rig.inject("estop", estop_seq=5, source="button")
        assert s["mode"] == "ESTOPPED"
        await asyncio.sleep(0.3)
        s = await rig.state()
        assert s["dev_acks"] and s["dev_acks"][-1]["type"] == "estop.ack" and s["dev_acks"][-1]["estop_seq"] == 5
        s = await rig.inject("estop_reset", estop_seq=5)
        assert s["mode"] == "IDLE"
        s = await rig.inject("caregiver", enabled=True)
        assert s["caregiver_enabled"] is True


async def test_caregiver_enables_high_risk(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="turn on the stove")
        p = await rig.proposal(phone)
        o = p["options"][0]
        assert o["skill_ids"] == ["stove_on"] and o["locked"] and "needs_caregiver" in o["badges"]
        await rig.confirm(phone, p, "o1")
        await phone.wait(lambda m: m["type"] == "notice" and m["text"] == "Try again")
        await rig.inject("caregiver", enabled=True)
        await rig.inject("utterance", text="turn on the stove")
        p = await rig.proposal(phone)
        o = p["options"][0]
        assert not o["locked"] and o["confirm"] == "double"


async def test_phone_lease_ping_and_bad_messages(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        a = await rig.phone()
        welcome = next(m for m in a.msgs if m["type"] == "welcome")
        assert welcome["lease"] == "user" and welcome["sess"].startswith("s_") and welcome["seq"] == 1
        assert welcome["resume_token"].startswith("rt_")
        state = await a.wait_type("state")
        assert state["seq"] == 2 and state["scene_id"] == "libero_goal_v1"  # seq increases per socket
        await a.send("ping", t0=123)
        pong = await a.wait_type("pong")
        assert pong["t0"] == 123 and pong["t_srv"] > 0
        await a.ws.send('{"type": "hb", "face_present": true}')  # safety message on the phone socket
        bad = await a.wait_type("notice")
        assert "Bad message" in bad["text"]
        b = await rig.phone()  # a second user phone takes the lease
        await a.wait(lambda m: m["type"] == "notice" and "took over" in m["text"])
        await a.send("ui.menu", source="brow_raise")
        await a.wait(lambda m: m["type"] == "notice" and "does not hold" in m["text"])
        cg = await rig.phone(role="caregiver")
        assert next(m for m in cg.msgs if m["type"] == "welcome")["lease"] == "caregiver"
        assert b is not None


async def test_duplicate_seq_dropped(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await phone.send("ui.menu", source="brow_raise")
        await rig.proposal(phone)
        phone.seq -= 1  # resend with the same seq: ignored
        await phone.send("ui.cancel", proposal_id=None, source="button")
        await asyncio.sleep(0.2)
        assert (await rig.state())["mode"] == "AWAITING_CONFIRM"


async def test_safety_socket_rejects_phone_messages(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        s = await rig.safety(heartbeat=False)
        await s.send("ui.menu", source="x")
        bad = await s.wait_type("notice")
        assert "Bad message" in bad["text"]
        await s.send("ping", t0=5)
        assert (await s.wait_type("pong"))["t0"] == 5


async def test_static_pwa(tmp_path):
    pwa = tmp_path / "pwa"
    pwa.mkdir()
    (pwa / "index.html").write_text("<html>nod phone</html>")
    (pwa / "presenter.html").write_text("<html>nod presenter</html>")
    (pwa / "js").mkdir()
    (pwa / "js" / "ui.js").write_text("export {}")
    async with Rig(fast_config(tmp_path, pwa_dir=pwa)) as rig:
        assert "nod phone" in (await rig.http.get("/")).text
        assert "nod presenter" in (await rig.http.get("/presenter")).text
        assert (await rig.http.get("/js/ui.js")).status_code == 200
        assert (await rig.http.get("/healthz")).status_code == 200  # API routes win over the mount


async def test_no_pwa_dir(tmp_path):
    async with Rig(fast_config(tmp_path, pwa_dir=tmp_path / "missing")) as rig:
        r = await rig.http.get("/")
        assert r.status_code == 200 and "PWA directory" in r.text


async def test_live_mode_without_sim(tmp_path):
    a = sim_addrs()
    cfg = Config(
        mode="live",
        sim_ctl=a["ctl"],
        sim_estop=a["estop"],
        sim_pub=a["pub"],
        sim_pose_pub=a["pose"],
        runs_dir=tmp_path / "runs",
        planner="fake",
        cosmos="none",
        home="fake",
        asr="fake",
    )
    async with Rig(cfg) as rig:
        assert rig.mock is None
        h = (await rig.http.get("/healthz")).json()
        assert h["ok"] is False and h["components"]["sim"]["ok"] is False
        assert (await rig.http.post("/api/dev/inject", json={"kind": "menu"})).status_code == 404
        assert (await rig.http.post("/api/scene/reset")).status_code == 403
        assert (await rig.http.get("/api/world")).status_code == 503
        # an e-stop still latches with no sim; the phone gets its ack after the wait
        s = await rig.safety(heartbeat=False)
        await s.send("estop", estop_seq=1, source="button", t_detect=0)
        ack = await s.wait_type("estop.ack", timeout=3)
        assert ack["lat_ms"]["sim_halt"] is None
        assert rig.gw.session.mode == "ESTOPPED"


async def test_live_mode_dev_flag(tmp_path):
    a = sim_addrs()
    cfg = Config(
        mode="live",
        dev=True,
        sim_ctl=a["ctl"],
        sim_estop=a["estop"],
        sim_pub=a["pub"],
        sim_pose_pub=a["pose"],
        runs_dir=tmp_path / "r",
        planner="fake",
        cosmos="none",
        asr="fake",
    )
    async with Rig(cfg) as rig:
        assert (await rig.http.get("/api/dev/state")).status_code == 200


def test_config_defaults_by_mode(monkeypatch):
    m = Config(mode="mock")
    assert (m.planner_kind, m.asr_kind, m.cosmos_kind, m.home_kind, m.run_mock_runner) == (
        "fake",
        "fake",
        "fake",
        "fake",
        True,
    )
    assert m.sim_ctl == "tcp://127.0.0.1:5560" and m.dev_enabled
    live = Config(mode="live")
    assert (live.planner_kind, live.asr_kind, live.cosmos_kind, live.home_kind, live.run_mock_runner) == (
        "nemotron",
        "parakeet",
        "none",
        "fake",
        False,
    )
    assert live.sim_ctl == "tcp://sim:5560" and not live.dev_enabled
    live2 = Config(mode="live", cosmos_url="http://127.0.0.1:8001/v1", hass_url="http://ha", hass_token="t")
    assert live2.cosmos_kind == "vllm" and live2.home_kind == "hass"
    monkeypatch.setenv("NOD_MODE", "live")
    monkeypatch.setenv("NOD_PLANNER", "fake")
    monkeypatch.setenv("NOD_HB_TIMEOUT_MS", "300")
    env = Config()
    assert env.mode == "live" and env.planner_kind == "fake" and env.hb_timeout_ms == 300
    assert Config(mode="mock").fast_model == "nvidia/Nemotron-3_5-Lightning"


def test_cli(monkeypatch):
    seen = {}

    def fake_run(app, **kw):
        seen["app"], seen["kw"] = app, kw

    monkeypatch.setattr("uvicorn.run", fake_run)
    monkeypatch.delenv("NOD_MODE", raising=False)
    cli.main(["--mock", "--port", "8123"])
    assert seen["kw"]["port"] == 8123 and seen["app"].state.gw.cfg.mode == "mock"
    with pytest.raises(SystemExit):
        cli.main(["--mock", "--live"])
