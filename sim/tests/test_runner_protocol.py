"""SimRunner over real ZMQ sockets (ipc) with the gateway's socket types (design 3.3, 3.5.4-3.5.6)."""

from __future__ import annotations

import time

import numpy as np
from nod_protocol import (
    Abort,
    Capture,
    CtlHeartbeat,
    LoadScene,
    Pause,
    Pose,
    Resume,
    SimAck,
    SimEpisodeHeader,
    SimEStopAck,
    SimFrames,
    SimStatus,
    SimWorld,
    WorldRequest,
    loads,
    now_ms,
)

from conftest import GOALS, moving_policy, run_scripted_cmd, run_skill_cmd
from nod_sim.fakes import FakePolicy


def ack_for(gw, req_id, timeout=3.0) -> SimAck:
    return gw.wait_for(lambda m: isinstance(m, SimAck) and m.req_id == req_id, timeout)


def final_status(gw, run_id, timeout=10.0) -> SimStatus:
    return gw.wait_for(lambda m: isinstance(m, SimStatus) and m.run_id == run_id
                       and m.phase in ("done", "failed", "aborted"), timeout)


def test_startup_publishes_header_and_world(served):
    runner, gw = served()
    gw.send(WorldRequest(req_id="q_w"))
    w = gw.wait_for(lambda m: isinstance(m, SimWorld))
    assert w.scene_id == "libero_goal_v1"
    assert w.atoms["on(akita_black_bowl_1,plate_1)"] is False
    assert "on(akita_black_bowl_1,main_table_handover_region)" in w.atoms
    assert w.eef.gripper == "open"
    assert "akita_black_bowl_1" in w.objects


def test_ack_protocol(served):
    runner, gw = served(policy=moving_policy(dx=0.0))
    # a valid run is acked ok
    cmd = run_skill_cmd(max_steps=2000)
    gw.send(cmd)
    assert ack_for(gw, cmd.req_id).ok
    gw.wait_for(lambda m: isinstance(m, SimStatus) and m.run_id == cmd.run_id and m.phase == "executing")
    # a second run while one is in flight is refused
    busy = run_skill_cmd()
    gw.send(busy)
    a = ack_for(gw, busy.req_id)
    assert not a.ok and "busy" in a.error
    # load_scene while running is refused
    gw.send(LoadScene(req_id="q_ls", scene_id="libero_goal_v1", epoch=0))
    assert not ack_for(gw, "q_ls").ok
    # abort ends the run
    gw.send(Abort(req_id="q_ab", epoch=0))
    assert ack_for(gw, "q_ab").ok
    assert final_status(gw, cmd.run_id).phase == "aborted"
    # unknown goal_ref is refused
    bad = run_skill_cmd()
    bad = bad.model_copy(update={"goal_ref": "libero_goal/make_coffee"})
    gw.send(bad)
    a = ack_for(gw, bad.req_id)
    assert not a.ok and "goal_ref" in a.error
    # unknown scene is refused
    gw.send(LoadScene(req_id="q_ls2", scene_id="kitchen_v9", epoch=0))
    assert not ack_for(gw, "q_ls2").ok
    # pause with no run is refused, abort with no run is fine
    gw.send(Pause(req_id="q_p", epoch=0))
    assert not ack_for(gw, "q_p").ok


def test_estop_over_socket_ack_and_stale_epoch(served, env):
    runner, gw = served(policy=moving_policy(dx=0.1))
    cmd = run_skill_cmd(max_steps=2000)
    gw.send(cmd)
    assert ack_for(gw, cmd.req_id).ok
    gw.wait_for(lambda m: isinstance(m, SimStatus) and m.run_id == cmd.run_id and m.step >= 6)
    t_gw = gw.estop(11)
    ack = gw.wait_for(lambda m: isinstance(m, SimEStopAck))
    t_seen = now_ms()
    assert ack.estop_seq == 11 and ack.epoch == 1
    assert ack.t_rx - t_gw <= 50, "e-stop not received within one control step"
    assert ack.t_halt - ack.t_rx <= 50
    assert ack.stopped_at_step is not None
    assert t_seen - t_gw < 500
    assert final_status(gw, cmd.run_id).phase == "aborted"
    # the run is never resumed; commands from the old epoch are refused
    stale = run_skill_cmd(epoch=0)
    gw.send(stale)
    a = ack_for(gw, stale.req_id)
    assert not a.ok and "stale epoch" in a.error
    gw.send(Resume(req_id="q_r", epoch=0))
    assert not ack_for(gw, "q_r").ok
    # the gateway, having bumped its own epoch, can start a new run
    fresh = run_skill_cmd(epoch=1, max_steps=5)
    gw.send(fresh)
    assert ack_for(gw, fresh.req_id).ok
    assert final_status(gw, fresh.run_id).fault == "budget"
    # a duplicate of the same estop_seq is acknowledged again, epoch unchanged
    gw.estop(11)
    acks = []
    end = time.monotonic() + 2
    while len(acks) < 2 and time.monotonic() < end:
        gw.drain(0.05)
        acks = [m for m in gw.seen if isinstance(m, SimEStopAck)]
    assert [a.epoch for a in acks] == [1, 1]


def test_pause_resume_over_socket(served, env):
    runner, gw = served(policy=moving_policy(dx=0.02))
    cmd = run_skill_cmd(max_steps=5000)
    gw.send(cmd)
    ack_for(gw, cmd.req_id)
    gw.wait_for(lambda m: isinstance(m, SimStatus) and m.run_id == cmd.run_id and m.step >= 4)
    gw.send(Pause(req_id="q_p", epoch=0))
    assert ack_for(gw, "q_p").ok
    p = gw.wait_for(lambda m: isinstance(m, SimStatus) and m.phase == "paused")
    assert p.fault is None  # requested pause, not a watchdog fault
    n = len(env.actions)
    time.sleep(0.2)
    assert len(env.actions) == n
    stop = env.actions[-4:]
    assert all(np.all(a[:6] == 0) for a in stop)
    gw.send(Resume(req_id="q_r", epoch=0))
    assert ack_for(gw, "q_r").ok
    gw.wait_for(lambda m: isinstance(m, SimStatus) and m.phase == "executing" and m.step > p.step)
    gw.send(Abort(req_id="q_a", epoch=0))
    assert final_status(gw, cmd.run_id).phase == "aborted"


def test_heartbeat_loss_pauses_over_socket(served, env):
    runner, gw = served(policy=moving_policy(dx=0.02), hb_timeout_ms=200)
    gw.send(CtlHeartbeat(t=now_ms(), phone_ok=True))
    cmd = run_skill_cmd(max_steps=5000)
    gw.send(cmd)
    ack_for(gw, cmd.req_id)
    for _ in range(5):  # keep it alive for a while
        gw.send(CtlHeartbeat(t=now_ms(), phone_ok=True))
        time.sleep(0.05)
    t_last = time.monotonic()
    p = gw.wait_for(lambda m: isinstance(m, SimStatus) and m.phase == "paused", timeout=3)
    assert p.fault == "hb_lost"
    assert time.monotonic() - t_last < 0.6
    # heartbeat back and resume confirmed by the user -> executing again
    gw.send(CtlHeartbeat(t=now_ms(), phone_ok=True))
    gw.send(Resume(req_id="q_r", epoch=0))
    assert ack_for(gw, "q_r").ok
    for _ in range(4):
        gw.send(CtlHeartbeat(t=now_ms(), phone_ok=True))
        time.sleep(0.03)
    gw.wait_for(lambda m: isinstance(m, SimStatus) and m.phase == "executing" and m.step > p.step)
    gw.send(Abort(req_id="q_a", epoch=0))
    final_status(gw, cmd.run_id)


def test_capture_and_world_replies(served):
    runner, gw = served()
    gw.send(Capture(req_id="q_c", cameras=["frontview", "agentview"], w=64, h=48))
    f = gw.wait_for(lambda m: isinstance(m, SimFrames) and m.req_id == "q_c")
    assert [x.camera for x in f.frames] == ["frontview", "agentview"]
    assert f.frames[0].w == 64 and f.frames[0].h == 48
    assert f.frames[0].jpeg[:2] == b"\xff\xd8"


def test_pose_stream_for_viewer(served):
    runner, gw = served(policy=moving_policy(dx=0.05))
    cmd = run_skill_cmd(max_steps=40)
    gw.send(cmd)
    gw.pose.setsockopt(__import__("zmq").RCVTIMEO, 3000)
    on_pose = [loads(gw.pose.recv()) for _ in range(12)]
    msg = next(m for m in on_pose if isinstance(m, Pose))
    assert msg.xpos.shape[1] == 3 and msg.xquat.shape[1] == 4 and msg.xpos.dtype == np.float32
    # the viewer only subscribes to the pose socket: the episode header is repeated there
    end = time.monotonic() + 5
    while not any(isinstance(m, SimEpisodeHeader) for m in on_pose) and time.monotonic() < end:
        on_pose.append(loads(gw.pose.recv()))
    assert any(isinstance(m, SimEpisodeHeader) for m in on_pose)
    final_status(gw, cmd.run_id)
    gw.send(LoadScene(req_id="q_l", scene_id="libero_goal_v1", init_index=0, epoch=0))
    header = gw.wait_for(lambda m: isinstance(m, SimEpisodeHeader), timeout=10)
    assert header.body_names[0] == "world" and header.model_patch.body_pos


def test_goal_swap_in_one_session_no_reset(served, env):
    """Two skills back to back: the instruction and goal change, the scene is never reset (1.3)."""
    seen = []

    def fn(obs, i):
        seen.append(obs["annotation.human.action.task_description"])
        c = np.zeros((8, 7), np.float32)
        c[:, 6] = 1.0
        return c

    runner, gw = served(policy=FakePolicy(fn, horizon=8), hz=None)
    resets = env.resets
    for skill in ("bowl_on_plate", "cheese_in_bowl"):
        cmd = run_skill_cmd(skill, max_steps=16)
        gw.send(cmd)
        assert ack_for(gw, cmd.req_id).ok
        final_status(gw, cmd.run_id)
    assert seen[0] == GOALS["bowl_on_plate"][0] and seen[-1] == GOALS["cheese_in_bowl"][0]
    assert env.goal == [["on", "cream_cheese_1", "akita_black_bowl_1"]]
    assert env.resets == resets


def test_scripted_actor_runs_chain_b_and_return_home(served, env):
    """Tier T4: with no policy, learned skills run on scripted stand-ins; return_home between."""
    runner, gw = served(policy=None, hz=None)
    for skill in ("bowl_on_plate", "return_home", "cheese_in_bowl", "return_home", "bowl_to_user"):
        cmd = run_scripted_cmd(skill) if skill == "return_home" else run_skill_cmd(skill)
        gw.send(cmd)
        assert ack_for(gw, cmd.req_id).ok
        st = final_status(gw, cmd.run_id)
        assert st.phase == "done" and st.success, skill
    w = runner._world_msg()
    assert w.atoms["on(akita_black_bowl_1,main_table_handover_region)"]
    assert w.atoms["on(cream_cheese_1,akita_black_bowl_1)"]


def test_reset_scene_and_close_drawer_scripts(served, env):
    runner, gw = served(policy=None, hz=None)
    cmd = run_skill_cmd("open_middle_drawer")
    gw.send(cmd)
    assert final_status(gw, cmd.run_id).success
    close = run_scripted_cmd("close_middle_drawer")
    gw.send(close)
    st = final_status(gw, close.run_id)
    assert st.phase == "done" and st.success
    assert env.eval_atom("open(wooden_cabinet_1_middle_region)") is False
    env.obj["akita_black_bowl_1"][:2] = (0.05, 0.05)
    reset = run_scripted_cmd("reset_scene")
    gw.send(reset)
    assert final_status(gw, reset.run_id).success
    assert env.obj["akita_black_bowl_1"][0] < -0.05
