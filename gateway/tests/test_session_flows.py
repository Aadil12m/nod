"""Session corner cases: late pre-checks (5.4), whole-plan mode with R11 spacing, the run-start
re-check, R14 speed scale, and the state machine's refusal paths."""

from __future__ import annotations

import asyncio
import time

from harness import Rig, fast_config

from nod_gateway.cosmos.fake import FakeVerifier


class SlowVerifier(FakeVerifier):
    def __init__(self, *a, delay: float, **kw) -> None:
        super().__init__(*a, **kw)
        self.delay = delay

    async def check(self, req, images=()):
        await asyncio.sleep(self.delay)
        return await super().check(req, images)


def slow_rig(tmp_path, delay: float, **cfg) -> Rig:
    rig = Rig(fast_config(tmp_path, **cfg))
    rig.gw.session.verifier = SlowVerifier(rig.gw.reg, lambda: rig.gw.session.world, delay=delay)
    return rig


async def test_run_waits_for_pending_precheck(tmp_path):
    async with slow_rig(tmp_path, 1.0, countdown_s=0.05) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="tidy up the bottle")
        p = await rig.proposal(phone)
        assert p["options"][0]["precheck"]["status"] == "pending"
        await rig.confirm(phone, p, "o1")
        phases = []
        while "executing" not in phases:
            m = await phone.wait(lambda m: m["type"] == "run", timeout=5)
            phases.append(m["phase"])
        assert phases[:3] == ["countdown", "precheck", "executing"]


async def test_late_unsure_precheck_asks_again_with_double_confirm(tmp_path):
    async with slow_rig(tmp_path, 0.8, countdown_s=0.05) as rig:
        phone = await rig.phone()
        await rig.safety()
        rig.gw.session.verifier.script = ["unsure"]
        await rig.inject("utterance", text="open the drawer please")
        p = await rig.proposal(phone)
        assert p["options"][0]["confirm"] == "single"
        await rig.confirm(phone, p, "o1")  # confirmed before the camera answered
        r = await phone.wait(lambda m: m["type"] == "run" and m["phase"] in ("aborted", "executing"))
        assert r["phase"] == "aborted"  # nothing moved
        again = await phone.wait(lambda m: m["type"] == "proposal" and m["options"][0]["confirm"] == "double")
        o = again["options"][0]
        assert o["skill_ids"] == ["open_middle_drawer"] and "unsure_from_camera" in o["badges"]
        assert rig.mock.run is None


async def test_whole_plan_mode_respects_r11_spacing(tmp_path):
    cfg = dict(plan_mode="whole", plan_countdown_s=0.1, spacing_ms=800, countdown_s=0.05)
    async with Rig(fast_config(tmp_path, **cfg)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="I'm a bit hungry")
        p = await rig.proposal(phone)
        await rig.confirm(phone, p, "o1")  # the plan; later steps start without a confirm
        done0 = await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "done", timeout=10)
        t_done = time.monotonic()
        await phone.wait(lambda m: m["type"] == "notice" and m["text"].startswith("Next:"))
        nxt = await phone.wait(
            lambda m: m["type"] == "run" and m["phase"] == "executing" and m["run_id"] != done0["run_id"], timeout=10
        )
        assert time.monotonic() - t_done >= 0.75  # R11: spacing, not the 0.1 s countdown
        assert nxt["skill_id"] == "cheese_in_bowl" and nxt["plan_step"] == 1
        trace = (await rig.http.get(f"/api/runs/{nxt['run_id']}")).json()
        assert any(e["data"].get("rule") == "R11_spacing" for e in trace["events"])


async def test_run_start_recheck_catches_world_change(tmp_path):
    async with Rig(fast_config(tmp_path, countdown_s=0.1)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="tidy up the bottle")
        p = await rig.proposal(phone)
        # the table changes after the card was shown (someone moved the bottle)
        rig.mock.atoms["on(wine_bottle_1,wine_rack_1_top_region)"] = True
        await rig.confirm(phone, p, "o1")
        r = await phone.wait(lambda m: m["type"] == "run" and m["phase"] in ("failed", "executing"))
        assert r["phase"] == "failed"
        await phone.wait(lambda m: m["type"] == "notice" and "no longer possible" in m["text"])
        assert rig.mock.run is None


async def test_near_user_skill_runs_at_reduced_speed(tmp_path):
    async with Rig(fast_config(tmp_path, countdown_s=0.05)) as rig:
        phone = await rig.phone()
        await rig.safety()
        sent = []
        real_run = rig.gw.sim.run

        async def spy(cmd):
            sent.append(cmd)
            await real_run(cmd)

        rig.gw.sim.run = spy
        await rig.inject("utterance", text="bring it to me")
        p = await rig.proposal(phone)
        assert p["options"][0]["skill_ids"] == ["bowl_to_user"]
        await rig.confirm(phone, p, "o1")
        again = await phone.wait(lambda m: m["type"] == "proposal" and m["nonce"] != p["nonce"])
        await rig.confirm(phone, again, "o1", focus=False)
        await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "done", timeout=10)
        skill_cmd = next(c for c in sent if c.type == "ctl.run_skill")
        assert skill_cmd.action_scale == 0.6 and skill_cmd.skill_id == "bowl_to_user"
        assert any(c.type == "ctl.run_scripted" and c.script == "handover_retreat" for c in sent)
        # second use in the session: no first-use double confirm (it is still double, but only
        # because the in-chain numbers for bowl_to_user are in the "less reliable" band)
        await phone.wait_type("state", mode="IDLE")
        rig.mock.atoms["on(akita_black_bowl_1,main_table_handover_region)"] = False
        await asyncio.sleep(0.15)
        await rig.inject("utterance", text="bring it to me")
        p2 = await rig.proposal(phone)
        o2 = p2["options"][0]
        assert "first_use" not in o2["badges"] and "less_reliable" in o2["badges"]
        assert o2["success"]["basis"] == "in_chain"


async def test_menu_ignored_while_moving(tmp_path):
    async with Rig(fast_config(tmp_path, mock_steps=200)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="tidy up the bottle")
        p = await rig.proposal(phone)
        await rig.confirm(phone, p, "o1")
        await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "executing")
        await phone.send("ui.menu", source="brow_raise")
        await phone.send("utter.text", utt_id=3, text="open the drawer")
        await asyncio.sleep(0.2)
        s = await rig.state()
        assert s["mode"] == "EXECUTING" and s["proposal"] is None


async def test_cancel_while_executing_aborts(tmp_path):
    async with Rig(fast_config(tmp_path, mock_steps=300)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="tidy up the bottle")
        p = await rig.proposal(phone)
        await rig.confirm(phone, p, "o1")
        r = await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "executing")
        await phone.send("ui.cancel", proposal_id=None, source="button")
        await phone.wait(lambda m: m["type"] == "run" and m["run_id"] == r["run_id"] and m["phase"] == "aborted")
        await phone.wait_type("state", mode="IDLE")
        assert rig.mock.run is None


async def test_prefetched_suggestions_used_by_menu(tmp_path):
    async with Rig(fast_config(tmp_path)) as rig:
        phone = await rig.phone()
        await rig.safety()
        await rig.inject("utterance", text="tidy up the bottle")
        p = await rig.proposal(phone)
        await rig.confirm(phone, p, "o1")
        await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "done", timeout=10)
        await phone.wait_type("state", mode="IDLE")
        for _ in range(50):
            if rig.gw.session.prefetched is not None:
                break
            await asyncio.sleep(0.05)
        assert rig.gw.session.prefetched is not None
        await phone.send("ui.menu", source="brow_raise")
        menu = await rig.proposal(phone)
        assert menu["source"] == "fake"  # the prefetched set, not the deterministic menu
