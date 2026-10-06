"""Regression tests for review findings on the gateway: heartbeat binding, protective pauses that
race the end of a motion, motion commands whose ack is lost, stop resets, R7/R14 on plans, the
confirm gesture allow-list and the planner's per-model mechanism memory.

The end-to-end tests run the real gateway (uvicorn) against the in-process MockRunner over ZMQ.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time

import httpx
from conftest import make_world, view
from harness import Rig, fast_config
from nod_protocol import (
    FixedOption,
    PlannerOption,
    PlannerOutput,
    Proposal,
    SimStatus,
    UiConfirm,
)

from nod_gateway.config import Config
from nod_gateway.planner.nemotron import NemotronPlanner
from nod_gateway.simlink import SimError
from nod_gateway.validator import SafetyPolicy, validate_confirm, validate_proposal


async def hb_loop(c, face: bool, token: str | None = None, latched: bool = False) -> None:
    with contextlib.suppress(Exception):
        while True:
            await c.send("hb", face_present=face, track_conf=0.9, fps=24, latched=latched, resume_token=token)
            await asyncio.sleep(0.1)


async def start_run(rig: Rig, phone, text: str = "tidy up the bottle") -> dict:
    await rig.inject("utterance", text=text)
    p = await rig.proposal(phone)
    await rig.confirm(phone, p, "o1")
    return await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "executing")


# ------------------------------------------------------------------ heartbeat binding (5.6, R9)


async def test_second_device_cannot_mask_face_loss(tmp_path):
    """A second PWA (no lease, or a caregiver phone) reporting a face must not hide the user's
    face loss: the gateway still pauses within the face-lost window."""
    async with Rig(fast_config(tmp_path, mock_steps=400, mock_hz=50)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await start_run(rig, phone)
        other = await rig.connect("/ws/safety", "other-device")  # never said hello, holds no lease
        cg = await rig.phone(role="caregiver")
        cg_safety = await rig.connect("/ws/safety", "caregiver-safety")
        tasks = [
            asyncio.create_task(hb_loop(other, True)),
            asyncio.create_task(hb_loop(other, True, token="rt_forged")),
            asyncio.create_task(hb_loop(cg_safety, True, token=cg.token)),
        ]
        rig.face = False
        try:
            s = await rig.until(lambda s: s["mode"] == "PAUSED", timeout=2.5)
        finally:
            for t in tasks:
                t.cancel()
        assert s["face_present"] is False


async def test_second_device_cannot_keep_heartbeat_fresh(tmp_path):
    async with Rig(fast_config(tmp_path, mock_steps=400, mock_hz=50)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await start_run(rig, phone)
        other = await rig.connect("/ws/safety", "other-device")
        t = asyncio.create_task(hb_loop(other, True))
        rig.hb_on = False  # the user's phone stops sending heartbeats
        try:
            s = await rig.until(lambda s: s["mode"] == "PAUSED", timeout=1.5)
        finally:
            t.cancel()
        assert s["hb_fresh"] is False
        await asyncio.sleep(0.8)  # and no ctl.hb is relayed: the mock runner's own watchdog holds too
        assert rig.mock.run is not None and rig.mock.run.phase == "paused"


async def test_previous_device_loses_heartbeat_role_when_lease_moves(tmp_path):
    """The user's previous phone keeps its old token; once another phone takes the user lease,
    the old one's heartbeats no longer count."""
    async with Rig(fast_config(tmp_path)) as rig:
        old = await rig.phone()
        old_token = old.token
        old_safety = await rig.connect("/ws/safety", "old-safety")
        t = asyncio.create_task(hb_loop(old_safety, True, token=old_token))
        await rig.until(lambda s: s["hb_fresh"] is True, timeout=2)
        await rig.phone()  # the new phone takes the lease (it sends no heartbeat in this test)
        try:
            await rig.until(lambda s: s["hb_fresh"] is False, timeout=2)
        finally:
            t.cancel()


async def test_latched_heartbeat_from_any_device_still_stops(tmp_path):
    async with Rig(fast_config(tmp_path, mock_steps=400, mock_hz=50)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await start_run(rig, phone)
        other = await rig.connect("/ws/safety", "other-device")
        t = asyncio.create_task(hb_loop(other, True, latched=True))
        try:
            await rig.until(lambda s: s["mode"] == "ESTOPPED", timeout=2)
        finally:
            t.cancel()
        await asyncio.sleep(0.1)
        assert rig.mock.run is None


# ------------------------------------------------------------------ protective pause races


async def test_pause_racing_skill_end_holds_return_home_until_resume(tmp_path):
    """A face-lost pause that lands as the skill finishes: no return_home while PAUSED; the
    "Resume?" card stays, and resuming runs return_home and the post-check."""
    async with Rig(fast_config(tmp_path, mock_steps=30, mock_hz=50, countdown_s=0.05)) as rig:
        phone = await rig.phone()
        await rig.safety()
        s = rig.gw.session
        sent = []
        real_run = rig.gw.sim.run

        async def spy_run(cmd):
            sent.append((cmd.type, getattr(cmd, "script", None), s.mode))
            await real_run(cmd)

        rig.gw.sim.run = spy_run
        real_on_status = s.on_sim_status
        fired = []

        async def on_status(st):
            ours = s.run is not None and st.run_id == s.run.run_id
            if st.phase == "done" and ours and s.mode == "EXECUTING" and not fired:
                fired.append(1)
                s.protective_pause("face_lost")
            await real_on_status(st)

        s.on_sim_status = on_status
        await rig.inject("utterance", text="tidy up the bottle")
        p = await rig.proposal(phone)
        await rig.confirm(phone, p, "o1")
        await phone.wait(lambda m: m["type"] == "notice" and m["text"].startswith("Paused"), timeout=10)
        await asyncio.sleep(0.6)
        assert ("ctl.run_scripted", "return_home", "PAUSED") not in sent
        assert all(c[0] != "ctl.run_scripted" for c in sent)
        st = await rig.state()
        assert st["mode"] == "PAUSED" and st["purpose"] == "resume"
        resume = st["proposal"]
        await rig.confirm(phone, resume, "resume", focus=False)
        end = await phone.wait(
            lambda m: m["type"] == "run" and m["phase"] in ("done", "failed", "uncertain", "aborted"), timeout=10
        )
        assert end["phase"] == "done"
        assert ("ctl.run_scripted", "return_home", "EXECUTING") in sent


async def test_cancel_while_paused_between_motions_ends_the_run(tmp_path):
    async with Rig(fast_config(tmp_path, mock_steps=30, mock_hz=50, countdown_s=0.05)) as rig:
        phone = await rig.phone()
        await rig.safety()
        s = rig.gw.session
        real_on_status = s.on_sim_status

        async def on_status(st):
            if st.phase == "done" and s.run is not None and st.run_id == s.run.run_id and s.mode == "EXECUTING":
                s.protective_pause("face_lost")
            await real_on_status(st)

        s.on_sim_status = on_status
        run = await start_run(rig, phone)
        await rig.until(lambda st: st["mode"] == "PAUSED" and st["purpose"] == "resume", timeout=10)
        await asyncio.sleep(0.2)
        lp = (await rig.state())["proposal"]
        await phone.send("ui.cancel", proposal_id=lp["proposal_id"], source="jaw_hold")
        await phone.wait(lambda m: m["type"] == "run" and m["run_id"] == run["run_id"] and m["phase"] == "aborted")
        await rig.until(lambda st: st["mode"] == "IDLE" and st["run"] is None, timeout=3)
        assert rig.mock.run is None


async def test_refused_pause_escalates_to_abort(tmp_path):
    """If the sim refuses (or never answers) ctl.pause while our motion may be moving, the
    gateway aborts the motion instead of only logging it."""
    async with Rig(fast_config(tmp_path, mock_steps=400, mock_hz=50)) as rig:
        phone = await rig.phone()
        await rig.safety()

        async def refuse(epoch):
            raise SimError("sim refused ctl.pause: no run")

        rig.gw.sim.pause = refuse
        run = await start_run(rig, phone)
        rig.face = False
        await phone.wait(
            lambda m: m["type"] == "run" and m["run_id"] == run["run_id"] and m["phase"] == "aborted", timeout=4
        )
        await asyncio.sleep(0.1)
        assert rig.mock.run is None


# ------------------------------------------------------------------ lost acks and stray motions


async def test_run_ack_timeout_aborts_the_late_run(tmp_path):
    """The sim accepts ctl.run_skill after the gateway's ack timeout: the gateway sends ctl.abort,
    so the robot does not move while the gateway and the phone say IDLE."""
    async with Rig(fast_config(tmp_path, mock_steps=150, mock_hz=50, countdown_s=0.05, sim_ack_timeout_s=0.5)) as rig:
        phone = await rig.phone()
        await rig.safety()
        m = rig.mock
        real = m._start_run

        def slow_start(ident, msg):
            time.sleep(0.8)  # the sim loop was busy when the request came in
            real(ident, msg)

        m._start_run = slow_start
        await rig.inject("utterance", text="tidy up the bottle")
        p = await rig.proposal(phone)
        await rig.confirm(phone, p, "o1")
        await phone.wait(lambda x: x["type"] == "notice" and "did not start" in x["text"], timeout=5)
        deadline = time.monotonic() + 3
        while m.run is not None and time.monotonic() < deadline:  # noqa: ASYNC110 - polls another thread
            await asyncio.sleep(0.02)
        assert m.run is None, f"the late run is still {m.run.phase} at step {m.run.step}"
        assert (await rig.state())["mode"] == "IDLE"


async def test_stray_sim_motion_is_aborted(tmp_path):
    """A sim.status for a motion the gateway does not track (current epoch) gets a ctl.abort."""
    async with Rig(fast_config(tmp_path)) as rig:
        await rig.phone()
        await rig.safety()
        aborts = []
        real_abort = rig.gw.sim.abort

        async def spy(epoch):
            aborts.append(epoch)
            await real_abort(epoch)

        rig.gw.sim.abort = spy
        s = rig.gw.session
        await s.on_sim_status(SimStatus(run_id="r_ghost", epoch=s.epoch, phase="executing", step=5))
        await s.on_sim_status(SimStatus(run_id="r_ghost", epoch=s.epoch, phase="executing", step=6))  # rate limited
        await s.on_sim_status(SimStatus(run_id="r_old", epoch=s.epoch - 1, phase="executing", step=5))  # stale
        await asyncio.sleep(0.2)
        assert len(aborts) == 1


async def test_cancel_in_idle_also_aborts_in_the_sim(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        phone = await rig.phone()
        await rig.safety()
        aborts = []
        real_abort = rig.gw.sim.abort

        async def spy(epoch):
            aborts.append(epoch)
            await real_abort(epoch)

        rig.gw.sim.abort = spy
        await phone.send("ui.cancel", proposal_id=None, source="jaw_hold")
        await asyncio.sleep(0.3)
        assert aborts and (await rig.state())["mode"] == "IDLE"


async def test_gateway_adopts_sim_epoch_after_restart(tmp_path):
    """The sim's epoch survives a gateway restart: its idle status after ctl.load_scene carries
    it and the gateway adopts it, so the first run is not refused as stale."""
    async with Rig(fast_config(tmp_path, countdown_s=0.05)) as rig:
        rig.mock.epoch = 5  # as if an earlier gateway process had stopped it five times
        await rig.gw.sim.load_scene("libero_goal_v1", 0, rig.gw.session.epoch)
        await rig.until(lambda s: s["epoch"] >= 5, timeout=2)
        phone = await rig.phone()
        await rig.safety()
        await start_run(rig, phone)


# ------------------------------------------------------------------ estop.reset


async def test_reset_requires_lease_seq_and_gesture(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        await rig.phone()
        user = await rig.safety()
        aborts = []
        real_abort = rig.gw.sim.abort

        async def spy(epoch):
            aborts.append(epoch)
            await real_abort(epoch)

        rig.gw.sim.abort = spy
        await user.send("estop", estop_seq=7, source="jaw_hold", t_detect=1, held_ms=320)
        await user.wait(lambda m: m["type"] == "estop.ack" and m["estop_seq"] == 7)
        stranger = await rig.connect("/ws/safety", "stranger")
        bad = [
            (stranger, dict(estop_seq=7, gesture="touch", held_ms=None)),  # no token
            (stranger, dict(estop_seq=7, gesture="touch", held_ms=None, resume_token="rt_forged")),
            (user, dict(estop_seq=1, gesture="long_blink", held_ms=820, resume_token=rig.token)),  # old seq
            (user, dict(estop_seq=7, gesture="whatever", held_ms=820, resume_token=rig.token)),
            (user, dict(estop_seq=7, gesture="long_blink", held_ms=0, resume_token=rig.token)),
            (user, dict(estop_seq=7, gesture="long_blink", held_ms=None, resume_token=rig.token)),
        ]
        for sock, body in bad:
            await sock.send("estop.reset", **body)
            await asyncio.sleep(0.1)
            assert (await rig.state())["mode"] == "ESTOPPED", body
        await user.send("estop.reset", estop_seq=7, gesture="long_blink", held_ms=820, resume_token=rig.token)
        await rig.until(lambda s: s["mode"] == "IDLE", timeout=2)
        await asyncio.sleep(0.1)
        assert aborts, "a reset clears anything the sim still holds with ctl.abort"


async def test_newer_stop_is_not_cleared_by_an_older_reset(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        await rig.phone()
        user = await rig.safety()
        await user.send("estop", estop_seq=3, source="jaw_hold", t_detect=1, held_ms=320)
        await user.wait(lambda m: m["type"] == "estop.ack" and m["estop_seq"] == 3)
        pres = await rig.connect("/ws/presenter", "presenter")
        await pres.send("estop", estop_seq=4, source="presenter", t_detect=1)
        await rig.until(lambda s: any(m.get("estop_seq") == 4 for m in s["last_messages"] if m["type"] == "state"))
        await user.send("estop.reset", estop_seq=3, gesture="long_blink", held_ms=820, resume_token=rig.token)
        await asyncio.sleep(0.2)
        assert (await rig.state())["mode"] == "ESTOPPED"
        await user.send("estop.reset", estop_seq=4, gesture="long_blink", held_ms=820, resume_token=rig.token)
        await rig.until(lambda s: s["mode"] == "IDLE", timeout=2)


async def test_caregiver_phone_may_reset_by_touch(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        await rig.phone()
        await rig.safety()
        cg = await rig.phone(role="caregiver")
        cg_safety = await rig.connect("/ws/safety", "caregiver-safety")
        await rig.inject("estop", estop_seq=2)
        await rig.until(lambda s: s["mode"] == "ESTOPPED")
        await rig.until(lambda s: any(a["estop_seq"] == 2 for a in s["dev_acks"]), timeout=2)
        await cg_safety.send("estop.reset", estop_seq=2, gesture="touch", held_ms=None, resume_token=cg.token)
        await rig.until(lambda s: s["mode"] == "IDLE", timeout=2)


# ------------------------------------------------------------------ R7 / R14 on plans, gestures


def _proposal(opts) -> Proposal:
    return Proposal(
        proposal_id="p_1",
        expires_at=10**13,
        arm_delay_ms=300,
        source="nemotron",
        model=None,
        transcript=None,
        options=opts,
        fixed=[FixedOption(option_id="none", label="None of these")],
        focus=opts[0].option_id,
        nonce="n_1",
    )


def _confirm(gesture="long_blink", held=750, now=1_000_000) -> UiConfirm:
    return UiConfirm(
        proposal_id="p_1",
        option_id="o1",
        nonce="n_1",
        gesture=gesture,
        held_ms=held,
        shown_at=now - 2000,
        focused_at=now - 2000,
        ts=now,
    )


def _confirm_view(now=1_000_000, **kw):
    return view(
        mode="AWAITING_CONFIRM",
        proposal_id="p_1",
        proposal_sent_ms=now - 2000,
        focus_option="o1",
        focus_since_ms=now - 2000,
        **kw,
    )


def _plan_option(reg, ids, **sess):
    out = PlannerOutput(
        intent="manipulate",
        clarify_question=None,
        options=[PlannerOption(kind="plan", skill_id=None, plan=ids, home_action=None, why="x")],
    )
    res = validate_proposal(out, make_world(reg), reg, view(mode="IDLE", **sess), SafetyPolicy(), 1_000_000)
    assert res.options, res.dropped
    return res.options[0].model_copy(update={"option_id": "o1"})


def test_r7_plan_starting_with_high_risk_step_needs_double_confirm(reg):
    o = _plan_option(reg, ["stove_on", "bowl_on_plate"], caregiver_enabled=True)
    assert o.skill_ids[0] == "stove_on" and o.risk == "high" and not o.locked
    assert o.confirm == "double"
    d = validate_confirm(_confirm(), _proposal([o]), _confirm_view(caregiver_enabled=True), 1_000_000, SafetyPolicy())
    assert d.outcome == "second"  # one long blink no longer starts stove_on


def test_r7_plan_with_high_risk_step_locked_without_caregiver(reg):
    o = _plan_option(reg, ["stove_on", "bowl_on_plate"])
    assert o.locked and "needs_caregiver" in o.badges


def test_plan_with_low_risk_first_step_stays_per_step(reg):
    o = _plan_option(reg, ["bowl_on_plate", "cheese_in_bowl", "bowl_to_user"])
    assert o.confirm == "per_step"


def test_confirm_gesture_allow_list(reg):
    out = PlannerOutput(
        intent="manipulate",
        clarify_question=None,
        options=[PlannerOption(kind="skill", skill_id="bottle_on_rack", plan=None, home_action=None, why="x")],
    )
    res = validate_proposal(out, make_world(reg), reg, view(mode="IDLE"), SafetyPolicy(), 1_000_000)
    p = _proposal([res.options[0].model_copy(update={"option_id": "o1"})])
    pol = SafetyPolicy()
    outcomes = {
        (g, h): validate_confirm(_confirm(g, h), p, _confirm_view(), 1_000_000, pol).outcome
        for g, h in (
            ("long_blink", 0),
            ("touch", 0),
            ("Long_Blink", 750),
            ("", 0),
            ("touch", 900),
            ("double_blink", 750),
            ("head_hold", 900),
            ("long_blink", 750),
        )
    }
    assert outcomes.pop(("long_blink", 750)) == "accept"
    assert set(outcomes.values()) == {"reject"}, outcomes


async def test_whole_plan_mode_asks_before_a_step_that_needs_double_confirm(tmp_path):
    """plan_mode=whole: bowl_to_user (near the user, first use: R14 double confirm) gets its own
    card instead of starting on the 3 s countdown."""
    cfg = dict(plan_mode="whole", plan_countdown_s=0.1, spacing_ms=0, countdown_s=0.05)
    async with Rig(fast_config(tmp_path, **cfg)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="I'm a bit hungry")
        p = await rig.proposal(phone)
        assert p["options"][0]["skill_ids"] == ["bowl_on_plate", "cheese_in_bowl", "bowl_to_user"]
        await rig.confirm(phone, p, "o1")
        done1 = await phone.wait(
            lambda m: m["type"] == "run" and m["phase"] == "done" and m["skill_id"] == "cheese_in_bowl", timeout=15
        )
        card = await phone.wait(lambda m: m["type"] == "proposal", timeout=5)
        o = card["options"][0]
        assert o["skill_ids"] == ["bowl_to_user"] and o["confirm"] == "double"
        await asyncio.sleep(0.5)
        later = [m for m in phone.msgs if m["type"] == "run" and m.get("skill_id") == "bowl_to_user"]
        assert later == [] and rig.mock.run is None
        assert done1["plan_step"] == 1


# ------------------------------------------------------------------ Nemotron mechanism per model


def _completion(content=None, tool_args=None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_args is not None:
        msg["tool_calls"] = [
            {"id": "c1", "type": "function", "function": {"name": "propose_options", "arguments": tool_args}}
        ]
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
    }


async def test_nemotron_remembers_mechanism_per_model(reg):
    """The fast URL accepts only json_schema; the plan URL rejects json_schema but accepts tools.
    Alternating calls must all succeed (the shared sticky mechanism made every plan call fail)."""
    from test_planner import GOOD, request

    seen: list[tuple[str, str]] = []

    def handler(r: httpx.Request) -> httpx.Response:
        body = json.loads(r.content)
        mech = "tools" if "tools" in body else body.get("response_format", {}).get("type", "?")
        plan = "us-central1" in str(r.url)
        seen.append(("plan" if plan else "fast", mech))
        if not plan and mech == "json_schema":
            return httpx.Response(200, json=_completion(json.dumps(GOOD)))
        if plan and mech == "tools":
            return httpx.Response(200, json=_completion(None, tool_args=json.dumps(GOOD)))
        return httpx.Response(400, json={"error": {"message": f"{mech} not supported"}})

    p = NemotronPlanner(Config(mode="live", tf_api_key="k"), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    results = []
    for mode in ("fast", "plan", "fast", "plan", "fast", "plan"):
        res = await p.propose_ex(request(reg, "hi", mode=mode))
        results.append(res.mechanism)
    assert results == ["json_schema", "tools"] * 3
    # After learning, each route goes straight to its own mechanism.
    assert seen[-2:] == [("fast", "json_schema"), ("plan", "tools")]


async def test_nemotron_rejected_sticky_falls_through_in_same_call(reg):
    from test_planner import GOOD, request

    state = {"schema_ok": True}

    def handler(r: httpx.Request) -> httpx.Response:
        body = json.loads(r.content)
        if "tools" in body:
            return httpx.Response(200, json=_completion(None, tool_args=json.dumps(GOOD)))
        if body.get("response_format", {}).get("type") == "json_schema" and state["schema_ok"]:
            return httpx.Response(200, json=_completion(json.dumps(GOOD)))
        return httpx.Response(400, json={"error": {"message": "no"}})

    p = NemotronPlanner(Config(mode="live", tf_api_key="k"), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert (await p.propose_ex(request(reg, "hi"))).mechanism == "json_schema"
    state["schema_ok"] = False  # the model stops accepting it
    assert (await p.propose_ex(request(reg, "hi"))).mechanism == "tools"
