"""SimLink against MockRunner over real ZMQ sockets: the same protocol the sim runner speaks."""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import pytest
import zmq
from conftest import SKILLS, sim_addrs
from nod_protocol import (
    Pose,
    RunScripted,
    RunSkill,
    SimEpisodeHeader,
    SimEStopAck,
    SimStatus,
    WorldState,
    loads,
    new_id,
)

from nod_gateway.mock_runner import PLACEHOLDER_JPEG, MockRunner, scan_library
from nod_gateway.simlink import SimError, SimLink


def run_cmd(
    skill_id="bowl_on_plate",
    epoch=0,
    max_steps=400,
    instruction="put the bowl on the plate",
    goal_ref="libero_goal/put_the_bowl_on_the_plate",
) -> RunSkill:
    return RunSkill(
        req_id=new_id("q"),
        run_id=new_id("r"),
        epoch=epoch,
        skill_id=skill_id,
        instruction=instruction,
        goal_ref=goal_ref,
        max_steps=max_steps,
    )


@pytest.fixture()
async def rig(tmp_path):
    a = sim_addrs()
    mock = MockRunner(
        None, a["ctl"], a["estop"], a["pub"], a["pose"], hz=100, registry=SKILLS, steps=30, hb_timeout_ms=None
    ).start()
    link = SimLink(a["ctl"], a["estop"], a["pub"], ack_timeout_s=2.0)
    await link.start()
    for _ in range(100):  # wait for the connection (IMMEDIATE: sends fail until connected)
        try:
            await link.load_scene("libero_goal_v1", 0, 0)
            break
        except SimError:
            await asyncio.sleep(0.02)
    yield link, mock, a
    await link.close()
    mock.stop()


async def statuses(link: SimLink, run_id: str, until: set[str], timeout: float = 5.0) -> list[SimStatus]:
    out: list[SimStatus] = []
    it = link.status()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        item = await asyncio.wait_for(it.__anext__(), deadline - time.monotonic())
        if isinstance(item, SimStatus) and item.run_id == run_id:
            out.append(item)
            if item.phase in until:
                return out
    raise AssertionError(f"no {until} for {run_id}: {[s.phase for s in out]}")


async def test_load_scene_and_world(rig):
    link, mock, _ = rig
    w = await link.world()
    assert isinstance(w, WorldState) and w.scene_id == "libero_goal_v1"
    assert w.atoms and not any(w.atoms.values())
    assert "akita_black_bowl_1" in w.objects


async def test_unknown_scene_refused(rig):
    link, _, _ = rig
    with pytest.raises(SimError, match="unknown scene"):
        await link.load_scene("other", 0, 0)


async def test_run_skill_succeeds_and_updates_oracle(rig):
    link, mock, _ = rig
    cmd = run_cmd()
    await link.run(cmd)
    st = await statuses(link, cmd.run_id, {"done", "failed", "aborted"})
    assert st[-1].phase == "done" and st[-1].success and st[-1].step == 30
    w = await link.world()
    assert w.atoms["on(akita_black_bowl_1,plate_1)"] is True


async def test_failure_ends_with_budget_fault(rig):
    link, mock, _ = rig
    mock.forced.append(False)
    cmd = run_cmd(max_steps=40)
    await link.run(cmd)
    st = await statuses(link, cmd.run_id, {"done", "failed", "aborted"})
    assert st[-1].phase == "failed" and st[-1].fault == "budget" and st[-1].step == 40
    assert not (await link.world()).atoms["on(akita_black_bowl_1,plate_1)"]


async def test_busy_and_stale_epoch_refused(rig):
    link, mock, _ = rig
    await link.run(run_cmd(max_steps=400))
    with pytest.raises(SimError, match="busy"):
        await link.run(run_cmd())
    await link.abort(0)
    mock.epoch = 5
    with pytest.raises(SimError, match="stale epoch"):
        await link.run(run_cmd(epoch=4))


async def test_unknown_skill_refused(rig):
    link, _, _ = rig
    with pytest.raises(SimError, match="unknown skill"):
        await link.run(run_cmd(skill_id="nope"))


async def test_estop_freezes_within_one_tick(rig):
    link, mock, _ = rig
    cmd = run_cmd()
    await link.run(cmd)
    await statuses(link, cmd.run_id, {"executing"})
    await asyncio.sleep(0.05)
    t0 = time.monotonic()
    link.estop(1)  # synchronous, non-blocking
    assert time.monotonic() - t0 < 0.005
    it = link.status()
    ack = None
    aborted = None
    while ack is None or aborted is None:
        item = await asyncio.wait_for(it.__anext__(), 2.0)
        if isinstance(item, SimEStopAck):
            ack = item
        elif isinstance(item, SimStatus) and item.run_id == cmd.run_id and item.phase == "aborted":
            aborted = item
    assert ack.estop_seq == 1 and ack.epoch == 1
    assert ack.t_halt - ack.t_rx <= 10  # well within one 10 ms tick at 100 Hz
    assert ack.stopped_at_step == aborted.step
    # never resumes: no further executing status for that run
    t_end = time.monotonic() + 0.3
    while time.monotonic() < t_end:
        try:
            item = await asyncio.wait_for(it.__anext__(), 0.1)
        except TimeoutError:
            continue
        assert not (isinstance(item, SimStatus) and item.run_id == cmd.run_id and item.phase == "executing")
    with pytest.raises(SimError, match="stale epoch"):  # the old epoch is dead
        await link.run(run_cmd(epoch=0))
    await link.run(run_cmd(epoch=1))


async def test_estop_while_idle_still_acked(rig):
    link, mock, _ = rig
    link.estop(7)
    it = link.status()
    while True:
        item = await asyncio.wait_for(it.__anext__(), 2.0)
        if isinstance(item, SimEStopAck):
            assert item.estop_seq == 7 and item.stopped_at_step is None
            break


async def test_pause_resume_abort(rig):
    link, mock, _ = rig
    cmd = run_cmd()
    await link.run(cmd)
    await statuses(link, cmd.run_id, {"executing"})
    await link.pause(0)
    st = await statuses(link, cmd.run_id, {"paused"})
    step = st[-1].step
    await asyncio.sleep(0.1)
    assert mock.run is not None and mock.run.step == step  # frozen while paused
    await link.resume(0)
    await statuses(link, cmd.run_id, {"executing"})
    await link.abort(0)
    st = await statuses(link, cmd.run_id, {"aborted"})
    assert st[-1].phase == "aborted"


async def test_scripted_run_and_reset(rig):
    link, mock, _ = rig
    cmd = run_cmd()
    await link.run(cmd)
    await statuses(link, cmd.run_id, {"done"})
    r = RunScripted(req_id=new_id("q"), run_id=new_id("r"), epoch=0, script="reset_scene", max_steps=160)
    await link.run(r)
    st = await statuses(link, r.run_id, {"done"})
    assert st[-1].step == 15
    assert not any((await link.world()).atoms.values())


async def test_capture_returns_jpeg(rig):
    link, _, _ = rig
    frames = await link.capture(["frontview", "agentview"], 640, 480)
    assert [f.camera for f in frames] == ["frontview", "agentview"]
    assert (
        frames[0].jpeg == PLACEHOLDER_JPEG and frames[0].jpeg[:2] == b"\xff\xd8" and frames[0].jpeg[-2:] == b"\xff\xd9"
    )


async def test_sim_hb_watchdog_pauses(tmp_path):
    a = sim_addrs()
    mock = MockRunner(
        None, a["ctl"], a["estop"], a["pub"], a["pose"], hz=100, registry=SKILLS, steps=500, hb_timeout_ms=200
    ).start()
    link = SimLink(a["ctl"], a["estop"], a["pub"])
    await link.start()
    try:
        for _ in range(100):
            try:
                await link.load_scene("libero_goal_v1", 0, 0)
                break
            except SimError:
                await asyncio.sleep(0.02)
        cmd = run_cmd()
        await link.run(cmd)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.4:  # heartbeats keep it running
            link.send_hb(0)
            await asyncio.sleep(0.05)
        assert mock.run is not None and mock.run.phase == "executing"
        st = await statuses(link, cmd.run_id, {"paused"}, timeout=2.0)  # stop relaying: controlled stop
        assert st[-1].fault == "hb_lost"
    finally:
        await link.close()
        mock.stop()


async def test_poses_published(rig):
    link, mock, a = rig
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(a["pose"])
    try:
        assert sub.poll(2000)
        msg = loads(sub.recv())
        assert isinstance(msg, Pose) and msg.xpos.shape == (0, 3)
    finally:
        sub.close(0)


async def test_library_replay(tmp_path):
    rec = tmp_path / "lib" / "bowl_on_plate_ok_0"
    rec.mkdir(parents=True)
    T, nb, nq = 12, 3, 9
    np.savez(
        rec / "poses.npz",
        xpos=np.random.rand(T, nb, 3).astype(np.float32),
        xquat=np.tile(np.array([1, 0, 0, 0], np.float32), (T, nb, 1)),
        qpos=np.zeros((T, nq), np.float32),
    )
    names = ["world", "table", "bowl"]
    (rec / "manifest.json").write_text(
        json.dumps(
            {
                "skill_id": "bowl_on_plate",
                "outcome": "success",
                "steps": T,
                "model_hash": "sha256:x",
                "mj_ver": "3.3.1",
                "body_names": names,
                "dyn_body_ids": [2],
            }
        )
    )
    (rec / "model_patch.json").write_text(json.dumps({"body_pos": {"table": [0.0, 0.0, 0.875]}, "body_quat": {}}))
    lib = scan_library(tmp_path / "lib")
    assert len(lib) == 1 and lib[0].success and lib[0].steps == T
    a = sim_addrs()
    mock = MockRunner(
        tmp_path / "lib",
        a["ctl"],
        a["estop"],
        a["pub"],
        a["pose"],
        hz=100,
        registry=SKILLS,
        steps=50,
        hb_timeout_ms=None,
    ).start()
    link = SimLink(a["ctl"], a["estop"], a["pub"])
    await link.start()
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(a["pose"])
    try:
        for _ in range(100):
            try:
                await link.load_scene("libero_goal_v1", 0, 0)
                break
            except SimError:
                await asyncio.sleep(0.02)
        cmd = run_cmd()
        await link.run(cmd)
        st = await statuses(link, cmd.run_id, {"done"})
        assert st[-1].step == T  # the recording's length, not the scripted 50
        shapes, headers = set(), []
        for _ in range(400):
            if not sub.poll(100) or ((nb, 3) in shapes and headers):
                break
            msg = loads(sub.recv())
            if isinstance(msg, SimEpisodeHeader):
                headers.append(msg)
            else:
                shapes.add(msg.xpos.shape)
        assert (nb, 3) in shapes
        # The viewer subscribes only to the pose socket: the header with the library's
        # fixture placement must arrive there too (as nod_sim.runner does).
        assert headers, "no sim.episode_header on the pose socket"
        assert headers[0].body_names == names and headers[0].model_hash == "sha256:x"
        assert tuple(headers[0].model_patch.body_pos["table"]) == (0.0, 0.0, 0.875)
        # After the run the last recorded pose is held, not replaced by empty arrays.
        await asyncio.sleep(0.1)
        for _ in range(20):
            msg = loads(sub.recv())
            if isinstance(msg, Pose):
                assert msg.xpos.shape == (nb, 3)
    finally:
        sub.close(0)
        await link.close()
        mock.stop()


async def test_estop_without_sim_raises():
    a = sim_addrs()
    link = SimLink(a["ctl"], a["estop"], a["pub"])
    await link.start()
    try:
        with pytest.raises(SimError):
            link.estop(1)  # nothing listening: reported, not queued for later
        assert link.send_hb(0) is False
        with pytest.raises(SimError):
            await link.run(run_cmd())
    finally:
        await link.close()
