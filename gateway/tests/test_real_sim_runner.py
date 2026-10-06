"""Gateway against the REAL sim runner (cross-module integration, CPU).

`python -m nod_sim.runner --fake-env` runs the actual `SimRunner` (sockets, 20 Hz loop, e-stop,
heartbeat watchdog, scripted actor, oracle) over `nod_sim.fakes.FakeEnv` in its own process from
`sim/.venv`, and the gateway connects to it with its real `SimLink` instead of the MockRunner.
This checks that the two sides of design 3.5.4 to 3.5.6 agree on the wire, which the MockRunner
cannot (it is the gateway's own stand-in).

Skipped when `sim/.venv` does not exist (`make setup-sim`).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

import pytest
from conftest import REPO, SKILLS, sim_addrs
from harness import Rig, fast_config

SIM_PY = REPO / "sim" / ".venv" / "bin" / "python"
BOWL_ON_PLATE = "on(akita_black_bowl_1,plate_1)"

pytestmark = pytest.mark.skipif(not SIM_PY.exists(), reason="sim/.venv missing (make setup-sim)")


@pytest.fixture()
def real_sim(tmp_path):
    a = sim_addrs()
    log = open(tmp_path / "sim.log", "w")  # noqa: SIM115 - closed below
    # The sim must not import the gateway's tests dir or cwd modules.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.Popen(
        [
            str(SIM_PY),
            "-m",
            "nod_sim.runner",
            "--fake-env",
            "--registry",
            str(SKILLS),
            "--ctl",
            a["ctl"],
            "--estop",
            a["estop"],
            "--pub",
            a["pub"],
            "--pose",
            a["pose"],
            "--hb-timeout-ms",
            "500",
        ],
        cwd=REPO / "sim",
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    try:
        yield a, proc
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        if proc.returncode not in (0, -15, None):
            sys.stderr.write((tmp_path / "sim.log").read_text()[-3000:])


def sim_config(tmp_path, a):
    return fast_config(
        tmp_path,
        mock_runner=False,
        sim_ctl=a["ctl"],
        sim_estop=a["estop"],
        sim_pub=a["pub"],
        sim_pose_pub=a["pose"],
    )


async def wait_loaded(rig: Rig, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not (rig.gw.scene_loaded and rig.gw.session.world is not None):
        assert time.monotonic() < deadline, "the gateway never loaded the scene on the real sim runner"
        await asyncio.sleep(0.05)


def run_end(m: dict, rid: str) -> bool:
    return m["type"] == "run" and m["run_id"] == rid and m["phase"] in ("done", "failed", "uncertain", "aborted")


async def test_plan_step_runs_on_real_sim_then_estop_aborts(tmp_path, real_sim):
    a, proc = real_sim
    async with Rig(sim_config(tmp_path, a)) as rig:
        await wait_loaded(rig)
        h = (await rig.http.get("/healthz")).json()
        assert h["components"]["sim"]["scene_loaded"] and h["deps"]["mock_runner"] is False
        w = (await rig.http.get("/api/world")).json()
        assert w["scene_id"] == "libero_goal_v1" and w["atoms"][BOWL_ON_PLATE] is False
        assert "akita_black_bowl_1" in w["objects"]

        phone = await rig.phone()
        safety = await rig.safety()
        await phone.send("utter.text", utt_id=1, text="I'm a bit hungry")
        p = await rig.proposal(phone, source="fake")
        assert p["options"][0]["skill_ids"][0] == "bowl_on_plate"
        await rig.confirm(phone, p, "o1")

        # Step 1 runs on the real runner (scripted actor), then the scripted return home.
        run = await phone.wait(lambda m: m["type"] == "run" and m["phase"] == "executing", timeout=10)
        rid = run["run_id"]
        moving = await phone.wait(lambda m: m["type"] == "run" and m["run_id"] == rid and m["sim_step"] > 0, timeout=10)
        assert moving["max_steps"] == 400
        done = await phone.wait(lambda m: run_end(m, rid), timeout=60)
        assert done["phase"] == "done", done
        assert done["oracle"] is True  # the real runner's oracle says the bowl is on the plate
        phases = [m["phase"] for m in phone.of_type("run") if m["run_id"] == rid]
        assert "returning_home" in phases and "postcheck" in phases
        world = (await rig.http.get("/api/world")).json()
        assert world["atoms"][BOWL_ON_PLATE] is True

        # Step 2 is offered; confirm it and stop it mid-motion.
        p2 = await rig.proposal(phone, timeout=10)
        assert p2["options"][0]["skill_ids"] == ["cheese_in_bowl"]
        await rig.confirm(phone, p2, p2["options"][0]["option_id"])
        run2 = await phone.wait(
            lambda m: m["type"] == "run" and m["skill_id"] == "cheese_in_bowl" and m["sim_step"] >= 5, timeout=15
        )
        rid2 = run2["run_id"]
        await safety.send("estop", estop_seq=1, source="jaw_hold", t_detect=run2["ts"], held_ms=300)
        ack = await safety.wait_type("estop.ack", timeout=3)
        assert ack["estop_seq"] == 1
        # Both latency parts come from the real sim's sim.estop_ack (not the 1 s fallback ack).
        assert ack["lat_ms"]["gw_sim"] is not None and ack["lat_ms"]["sim_halt"] is not None
        assert ack["stopped_at_step"] >= 5
        aborted = await phone.wait(lambda m: run_end(m, rid2), timeout=3)
        assert aborted["phase"] == "aborted"
        st = await phone.wait_type("state", mode="ESTOPPED")
        assert st["estop_seq"] == 1

        # Reset: back to IDLE; the sim stays idle and the run is never resumed.
        await safety.send("estop.reset", estop_seq=1, gesture="long_blink", held_ms=820)
        await phone.wait_type("state", mode="IDLE")
        n = len(phone.msgs)
        await asyncio.sleep(1.0)
        assert not [m for m in phone.msgs[n:] if m["type"] == "run" and m["run_id"] == rid2]
        s = await rig.state()
        assert s["mode"] == "IDLE" and s["run"] is None
        assert rig.gw.session.epoch >= 1
        assert proc.poll() is None, "sim runner died"


async def test_heartbeat_loss_pauses_on_real_sim(tmp_path, real_sim):
    a, proc = real_sim
    async with Rig(sim_config(tmp_path, a)) as rig:
        await wait_loaded(rig)
        phone = await rig.phone()
        await rig.safety()
        await phone.send("utter.text", utt_id=1, text="I'm a bit hungry")
        p = await rig.proposal(phone, source="fake")
        await rig.confirm(phone, p, "o1")
        run = await phone.wait(lambda m: m["type"] == "run" and m["sim_step"] >= 5, timeout=15)
        rid = run["run_id"]
        rig.hb_on = False  # the phone goes silent: the gateway stops relaying ctl.hb
        await phone.wait_type("state", mode="PAUSED", timeout=3)
        paused = await phone.wait(lambda m: m["type"] == "run" and m["run_id"] == rid and m["phase"] == "paused")
        n = len(phone.msgs)
        await asyncio.sleep(1.0)
        later = [m for m in phone.msgs[n:] if m["type"] == "run" and m["run_id"] == rid]
        assert all(m["phase"] == "paused" for m in later), [m["phase"] for m in later]
        assert all(m["sim_step"] <= paused["sim_step"] + 4 for m in later)  # the real runner holds still
        assert (await rig.state())["mode"] == "PAUSED"
        assert proc.poll() is None
