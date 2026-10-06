"""Mock end-to-end tests (design 4.6): the gateway with MockRunner over real sockets.

menu -> utterance -> focus -> confirm -> run -> done; e-stop mid-run (estop.ack, aborted, never
resumed); watchdog pause on heartbeat loss and face loss; plans; failure and recovery.
"""

from __future__ import annotations

import asyncio

from harness import Rig, fast_config

from nod_gateway.planner.fake import FakePlanner

CHEESE = "on(cream_cheese_1,akita_black_bowl_1)"


def run_phases(phone, run_id: str) -> list[str]:
    return [m["phase"] for m in phone.of_type("run") if m["run_id"] == run_id]


async def test_menu_utterance_focus_confirm_run_done(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        phone = await rig.phone()
        await rig.safety()
        # menu (brow raise): deterministic menu, no LLM
        await phone.send("ui.menu", source="brow_raise")
        menu = await rig.proposal(phone, source="menu")
        assert len(menu["options"]) == 3 and menu["options"][-1]["home_action"] == "call_caregiver"
        assert menu["fixed"] == [{"option_id": "none", "label": "None of these"}]
        # utterance replaces the menu
        await phone.send("utter.text", utt_id=8, text="I'm a bit hungry")
        tr = await phone.wait_type("transcript")
        assert tr["text"] == "I'm a bit hungry" and tr["utt_id"] == 8
        p = await rig.proposal(phone, source="fake")
        assert [o["kind"] for o in p["options"]] == ["plan", "skill", "home"]
        assert p["options"][1]["label"] == "Put the cream cheese in the bowl"  # R13
        assert p["options"][1]["success"]["n"] >= 20
        assert p["transcript"] == "I'm a bit hungry" and p["focus"] == "o1" and p["nonce"].startswith("n_")
        # focus o2, dwell, long blink
        await rig.confirm(phone, p, "o2")
        await phone.wait_type("state", mode="COUNTDOWN")
        run = await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "countdown")
        rid = run["run_id"]
        assert run["skill_id"] == "cheese_in_bowl" and run["proposal_id"] == p["proposal_id"]
        await phone.wait_type("state", mode="EXECUTING")
        done = await phone.wait(
            lambda m: (
                m["type"] == "run" and m["run_id"] == rid and m["phase"] in ("done", "failed", "uncertain", "aborted")
            ),
            timeout=10,
        )
        assert done["phase"] == "done" and done["oracle"] is True
        assert done["cosmos"]["answer"] == "yes" and done["cosmos"]["decision"] == "done"
        phases = run_phases(phone, rid)
        for a, b in [
            ("countdown", "executing"),
            ("executing", "returning_home"),
            ("returning_home", "postcheck"),
            ("postcheck", "done"),
        ]:
            assert phases.index(a) < phases.index(b), phases
        await phone.wait_type("state", mode="IDLE")
        world = (await rig.http.get("/api/world")).json()
        assert world["atoms"][CHEESE] is True
        # the trace is complete
        runs = (await rig.http.get("/api/runs")).json()["runs"]
        rec = (await rig.http.get(f"/api/runs/{rid}")).json()
        assert rec["outcome"] == "done" and rec["skill_id"] == "cheese_in_bowl"
        prec = (await rig.http.get(f"/api/runs/{p['proposal_id']}")).json()
        assert prec["files"]["planner_trace.json"]["response"]["intent"] == "manipulate"
        assert any(e["kind"] == "validator" for e in prec["events"])
        assert {r["id"] for r in runs} >= {rid, p["proposal_id"]}
        assert (tmp_path / "runs" / rid / "events.jsonl").exists()


async def test_estop_mid_run_aborted_never_resumed(tmp_path):
    async with Rig(fast_config(tmp_path, mock_steps=400)) as rig:
        phone = await rig.phone()
        safety = await rig.safety()
        await rig.inject("utterance", text="tidy up the bottle")
        p = await rig.proposal(phone)
        await rig.confirm(phone, p, "o1")
        run = await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "executing")
        rid = run["run_id"]
        await asyncio.sleep(0.3)
        await safety.send("estop", estop_seq=3, source="jaw_hold", t_detect=run["ts"], held_ms=310)
        ack = await safety.wait_type("estop.ack", timeout=2)
        assert ack["estop_seq"] == 3 and ack["stopped_at_step"] is not None and ack["stopped_at_step"] > 0
        assert ack["lat_ms"]["gw_sim"] is not None and ack["lat_ms"]["sim_halt"] is not None
        assert ack["lat_ms"]["sim_halt"] <= 50  # one control step
        await phone.wait(lambda m: m["type"] == "run" and m["run_id"] == rid and m["phase"] == "aborted")
        st = await phone.wait_type("state", mode="ESTOPPED")
        assert st["estop_seq"] == 3  # the phone and presenter compare this with their own number
        # the phone resends until acked: duplicates do not stop the sim again but are re-acked
        await safety.send("estop", estop_seq=3, source="jaw_hold", t_detect=run["ts"], held_ms=310)
        await safety.wait_type("estop.ack", estop_seq=3)
        assert rig.mock.estops == [1]
        assert rig.mock.run is None  # the sim run was aborted, not paused
        # nothing moves while latched
        await rig.inject("menu")
        await rig.inject("utterance", text="tidy up the bottle")
        assert (await rig.state())["mode"] == "ESTOPPED"
        # reset -> IDLE; the interrupted run is never resumed
        await safety.send("estop.reset", estop_seq=3, gesture="long_blink", held_ms=820, resume_token=rig.token)
        await phone.wait_type("state", mode="IDLE")
        await asyncio.sleep(0.6)
        later = [m for m in phone.msgs[phone.cursor :] if m["type"] == "run" and m["run_id"] == rid]
        assert later == []
        assert run_phases(phone, rid)[-1] == "aborted"
        s = await rig.state()
        assert s["mode"] == "IDLE" and s["run"] is None and s["plan"] is None
        assert rig.mock.run is None
        assert s["world"]["atoms"]["on(wine_bottle_1,wine_rack_1_top_region)"] is False
        rec = (await rig.http.get(f"/api/runs/{rid}")).json()
        assert rec["outcome"] == "aborted"
        m = (await rig.http.get("/api/metrics")).json()
        assert m["histograms"]["estop_sim_halt_ms"]["n"] == 1


async def test_estop_from_presenter_and_trace(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        phone = await rig.phone()
        await rig.safety()
        pres = await rig.connect("/ws/presenter", "presenter")
        await pres.wait_type("state")
        await rig.inject("utterance", text="open the drawer please")
        await pres.wait(lambda m: m["type"] == "trace" and m["kind"] == "planner")
        await pres.wait(lambda m: m["type"] == "trace" and m["kind"] == "validator")
        await pres.wait_type("proposal")  # mirrored
        await pres.send("estop", estop_seq=1, source="presenter", t_detect=0)
        ack = await pres.wait_type("estop.ack", timeout=2)
        assert ack["estop_seq"] == 1
        await phone.wait_type("state", mode="ESTOPPED")
        await pres.wait(lambda m: m["type"] == "trace" and m["kind"] == "estop")
        await pres.wait_type("metrics", timeout=3)


async def test_presenter_cannot_confirm(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        await rig.phone()
        await rig.safety()
        pres = await rig.connect("/ws/presenter", "presenter")
        await rig.inject("utterance", text="open the drawer please")
        p = await pres.wait_type("proposal")
        await pres.send(
            "ui.confirm",
            proposal_id=p["proposal_id"],
            option_id="o1",
            nonce=p["nonce"],
            gesture="long_blink",
            held_ms=800,
            shown_at=0,
            focused_at=0,
        )
        bad = await pres.wait_type("notice")
        assert "Bad message" in bad["text"]
        assert (await rig.state())["mode"] == "AWAITING_CONFIRM"


async def test_watchdog_pause_on_heartbeat_loss_then_resume(tmp_path):
    async with Rig(fast_config(tmp_path, mock_steps=150)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="tidy up the bottle")
        p = await rig.proposal(phone)
        await rig.confirm(phone, p, "o1")
        run = await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "executing")
        rid = run["run_id"]
        await asyncio.sleep(0.2)
        rig.hb_on = False  # the phone goes quiet
        await phone.wait_type("state", mode="PAUSED", timeout=2)
        await phone.wait(lambda m: m["type"] == "run" and m["run_id"] == rid and m["phase"] == "paused")
        resume = await rig.proposal(phone)
        assert resume["options"][0]["option_id"] == "resume" and "Lost contact" in resume["options"][0]["why"]
        await asyncio.sleep(0.2)
        assert rig.mock.run is not None and rig.mock.run.phase == "paused"
        step = rig.mock.run.step
        await asyncio.sleep(0.2)
        assert rig.mock.run.step == step  # frozen
        assert (await rig.state())["hb_fresh"] is False
        # confirm is refused while the heartbeat is stale
        await rig.confirm(phone, resume, "resume", focus=False)
        await phone.wait(lambda m: m["type"] == "notice" and m["text"] == "Try again")
        rig.hb_on = True
        await asyncio.sleep(0.3)
        resume2 = (await rig.state())["proposal"]
        await rig.confirm(phone, resume2, "resume", focus=False)
        await phone.wait_type("state", mode="EXECUTING")
        done = await phone.wait(
            lambda m: m["type"] == "run" and m["run_id"] == rid and m["phase"] == "done", timeout=10
        )
        assert done["oracle"] is True


async def test_face_lost_pauses_and_cancel_aborts(tmp_path):
    async with Rig(fast_config(tmp_path, mock_steps=300)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="tidy up the bottle")
        p = await rig.proposal(phone)
        await rig.confirm(phone, p, "o1")
        run = await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "executing")
        rig.face = False
        await phone.wait_type("state", mode="PAUSED", timeout=3)
        resume = await rig.proposal(phone)
        assert "I can't see you" in resume["options"][0]["why"]
        await phone.send("ui.cancel", proposal_id=resume["proposal_id"], source="jaw_hold")
        await phone.wait(lambda m: m["type"] == "run" and m["run_id"] == run["run_id"] and m["phase"] == "aborted")
        await phone.wait_type("state", mode="IDLE")
        assert rig.mock.run is None


async def test_plan_runs_step_by_step_with_double_confirm_for_near_user(tmp_path):
    async with Rig(fast_config(tmp_path, countdown_s=0.05)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="I'm a bit hungry")
        p = await rig.proposal(phone)
        plan = p["options"][0]
        assert plan["kind"] == "plan" and plan["confirm"] == "per_step"
        assert plan["skill_ids"] == ["bowl_on_plate", "cheese_in_bowl", "bowl_to_user"]
        await rig.confirm(phone, p, "o1")
        done0 = await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "done", timeout=10)
        assert done0["skill_id"] == "bowl_on_plate" and done0["plan_step"] == 0 and done0["plan_len"] == 3
        nxt = await rig.proposal(phone)
        assert nxt["options"][0]["skill_ids"] == ["cheese_in_bowl"] and nxt["focus"] == "o1"
        await rig.confirm(phone, nxt, "o1")
        done1 = await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "done", timeout=10)
        assert done1["skill_id"] == "cheese_in_bowl" and done1["plan_step"] == 1
        last = await rig.proposal(phone)
        o = last["options"][0]
        assert o["skill_ids"] == ["bowl_to_user"] and o["confirm"] == "double"
        assert {"near_user", "first_use"} <= set(o["badges"])
        await rig.confirm(phone, last, "o1")
        # second half of the double confirm: same card, fresh nonce (pre-check updates keep the old one)
        again = await phone.wait(
            lambda m: (
                m["type"] == "proposal" and m["proposal_id"] == last["proposal_id"] and m["nonce"] != last["nonce"]
            )
        )
        await phone.wait(lambda m: m["type"] == "notice" and "once more" in m["text"])
        await rig.confirm(phone, again, "o1", focus=False)
        done2 = await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "done", timeout=10)
        assert done2["skill_id"] == "bowl_to_user" and done2["plan_step"] == 2
        phases = run_phases(phone, done2["run_id"])
        assert "returning_home" in phases
        w = (await rig.http.get("/api/world")).json()["atoms"]
        assert w["on(akita_black_bowl_1,main_table_handover_region)"] and w[CHEESE]
        assert not w["on(akita_black_bowl_1,plate_1)"]
        await phone.wait_type("state", mode="IDLE")
        assert (await rig.state())["plan"] is None


async def test_reusing_a_nonce_is_rejected(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="tidy up the bottle")
        p = await rig.proposal(phone)
        await rig.confirm(phone, p, "o1")
        await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "done", timeout=10)
        await phone.wait_type("state", mode="IDLE")
        await rig.inject("utterance", text="open the drawer please")
        p2 = await rig.proposal(phone)
        await phone.send(
            "ui.confirm",
            proposal_id=p2["proposal_id"],
            option_id="o1",
            nonce=p["nonce"],
            gesture="long_blink",
            held_ms=800,
            shown_at=0,
            focused_at=0,
        )
        await phone.wait(lambda m: m["type"] == "notice" and m["text"] == "Try again")
        assert (await rig.state())["mode"] == "AWAITING_CONFIRM"


async def test_confirm_too_early_rejected_then_accepted(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="open the drawer please")
        p = await rig.proposal(phone)
        await phone.send(
            "ui.confirm",
            proposal_id=p["proposal_id"],
            option_id="o1",
            nonce=p["nonce"],
            gesture="long_blink",
            held_ms=800,
            shown_at=0,
            focused_at=0,
        )
        await phone.wait(lambda m: m["type"] == "notice" and m["text"] == "Try again")
        await rig.confirm(phone, p, "o1")  # the nonce was not consumed by the rejection
        await phone.wait_type("state", mode="COUNTDOWN")


async def test_failure_offers_recovery(tmp_path):
    async with Rig(fast_config(tmp_path, mock_hz=400)) as rig:
        phone = await rig.phone()
        await rig.safety()
        rig.mock.forced.append(False)
        await rig.inject("utterance", text="tidy up the bottle")
        p = await rig.proposal(phone)
        await rig.confirm(phone, p, "o1")
        failed = await phone.wait(lambda m: m["type"] == "run" and m["phase"] in ("failed", "done"), timeout=10)
        assert failed["phase"] == "failed" and failed["fault"] == "budget" and failed["oracle"] is False
        assert failed["cosmos"]["answer"] == "no"
        await phone.wait(lambda m: m["type"] == "notice" and m["text"] == "That didn't work")
        rec = await rig.proposal(phone)
        assert rec["options"][0]["skill_ids"] == ["bottle_on_rack"]  # retry
        assert rec["options"][-1]["home_action"] == "call_caregiver"


async def test_uncertain_when_cosmos_disagrees(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="tidy up the bottle")
        p = await rig.proposal(phone)
        await phone.wait(lambda m: m["type"] == "proposal" and m["options"][0]["precheck"]["status"] != "pending")
        rig.gw.session.verifier.script = ["unsure"]  # the post-check camera answer
        await rig.confirm(phone, p, "o1")
        r = await phone.wait(lambda m: m["type"] == "run" and m["phase"] in ("done", "uncertain"), timeout=10)
        assert r["phase"] == "uncertain" and r["oracle"] is True and r["cosmos"]["answer"] == "unsure"


async def test_unsure_precheck_requires_double_confirm(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        phone = await rig.phone()
        await rig.safety()
        rig.gw.session.verifier.script = ["unsure", "unsure"]
        await rig.inject("utterance", text="tidy up the bottle")
        upd = await phone.wait(lambda m: m["type"] == "proposal" and m["options"][0]["precheck"]["status"] == "unsure")
        o = upd["options"][0]
        assert o["confirm"] == "double" and "unsure_from_camera" in o["badges"]


async def test_home_action(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="it's too dark in here")
        p = await rig.proposal(phone)
        assert p["options"][0]["home_action"] == "lights_on" and p["options"][0]["precheck"] is None
        await rig.confirm(phone, p, "o1")
        r = await phone.wait(lambda m: m["type"] == "run" and m["skill_id"] == "lights_on")
        assert r["phase"] == "done"
        await phone.wait(lambda m: m["type"] == "notice" and "Lights on" in m["text"])
        assert rig.gw.session.home.lights_on


async def test_planner_offline_falls_back_to_menu(tmp_path):
    async with Rig(fast_config(tmp_path), planner=FakePlanner(tmp_path, fail=True)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="I'm a bit hungry")
        await phone.wait(lambda m: m["type"] == "notice" and "Planner offline" in m["text"])
        p = await rig.proposal(phone, source="menu")
        assert p["transcript"] == "I'm a bit hungry" and len(p["options"]) == 3
        h = (await rig.http.get("/healthz")).json()
        assert h["components"]["token_factory"]["ok"] is False


async def test_proposal_expires(tmp_path):
    async with Rig(fast_config(tmp_path, proposal_ttl_s=1)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("menu")
        await rig.proposal(phone)
        await phone.wait(lambda m: m["type"] == "notice" and "timed out" in m["text"], timeout=3)
        await phone.wait_type("state", mode="IDLE")


async def test_none_of_these_cancels(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("menu")
        p = await rig.proposal(phone)
        await rig.confirm(phone, p, "none")
        await phone.wait_type("state", mode="IDLE")


async def test_cancel_during_countdown(tmp_path):
    async with Rig(fast_config(tmp_path, countdown_s=1.0)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="tidy up the bottle")
        p = await rig.proposal(phone)
        await rig.confirm(phone, p, "o1")
        await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "countdown")
        await phone.send("ui.cancel", proposal_id=None, source="jaw_hold")
        r = await phone.wait(lambda m: m["type"] == "run" and m["phase"] in ("aborted", "executing"))
        assert r["phase"] == "aborted"
        await asyncio.sleep(1.2)
        assert rig.mock.run is None and not any(m["phase"] == "executing" for m in phone.of_type("run"))


async def test_audio_utterance_through_fake_asr(tmp_path):
    async with Rig(fast_config(tmp_path, fake_transcripts=["open the drawer please"])) as rig:
        phone = await rig.phone()
        await rig.safety()
        await phone.send("utter.start", utt_id=7, sr=16000, fmt="pcm16le")
        await phone.wait_type("state", mode="LISTENING")
        await phone.ws.send((7).to_bytes(4, "little") + b"\x00\x01" * 1600)
        await phone.send("utter.end", utt_id=7, dur_ms=100)
        tr = await phone.wait_type("transcript")
        assert tr["utt_id"] == 7 and tr["text"] == "open the drawer please" and tr["asr_ms"] is not None
        p = await rig.proposal(phone)
        assert p["options"][0]["skill_ids"] == ["open_middle_drawer"]


async def test_scene_reset(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="tidy up the bottle")
        p = await rig.proposal(phone)
        await rig.confirm(phone, p, "o1")
        await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "done", timeout=10)
        await phone.wait_type("state", mode="IDLE")
        r = await rig.http.post("/api/scene/reset")
        assert r.status_code == 200
        done = await phone.wait(
            lambda m: m["type"] == "run" and m["skill_id"] == "reset_scene" and m["phase"] == "done", timeout=10
        )
        assert done["oracle"] is True
        assert not any((await rig.http.get("/api/world")).json()["atoms"].values())
