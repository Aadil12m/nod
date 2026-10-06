"""Regression tests for review findings on the sim runner (queued runs, pauses, e-stop seqs).

Each test reproduces a reviewer's failure scenario against the real SimRunner (FakeEnv), over
real ZMQ sockets where the race needs them, and asserts the fixed behaviour.
"""

from __future__ import annotations

import time

import numpy as np
from nod_protocol import LoadScene, Pause, Resume, SimAck, SimEStopAck, SimStatus

from conftest import in_thread, make_runner, moving_policy, run_skill_cmd

TERMINAL = ("done", "failed", "aborted")


def _final(gw, run_id, timeout=10.0):
    return gw.wait_for(lambda m: isinstance(m, SimStatus) and m.run_id == run_id and m.phase in TERMINAL, timeout)


def test_estop_after_run_accepted_but_before_start_drops_the_run(served, env):
    """ctl.estop arriving after ctl.run_skill was accepted, before the loop started it: the
    queued run must never start (it used to run to the end after the e-stop was acked)."""
    runner, gw = served(policy=moving_policy(dx=0.2))
    orig = runner.handle_ctl

    def handle_ctl(msg, envelope=None):
        orig(msg, envelope)
        if msg.type == "ctl.run_skill":
            gw.estop(1)  # on the dedicated socket, queued before the loop goes round
            time.sleep(0.01)

    runner.handle_ctl = handle_ctl
    cmd = run_skill_cmd(max_steps=60)
    gw.send(cmd)
    gw.wait_for(lambda m: isinstance(m, SimEStopAck))
    n_at_ack = len(env.actions)
    st = _final(gw, cmd.run_id)
    time.sleep(0.3)
    moving = [a for a in env.actions[n_at_ack:] if np.any(a[:3] != 0)]
    assert st.phase == "aborted" and st.step == 0
    assert moving == []
    assert runner.run is None and runner._next_cmd is None


def test_estop_right_after_run_never_leaks(served, env):
    """No patching: run_skill and ctl.estop back to back. Every accepted run ends aborted."""
    runner, gw = served(policy=moving_policy(dx=0.0))
    leaks = 0
    for i in range(120):
        cmd = run_skill_cmd(epoch=runner.epoch, max_steps=4)
        gw.send(cmd)
        time.sleep(0.0002 * (i % 25))
        gw.estop(i + 1)
        gw.wait_for(lambda m, s=i + 1: isinstance(m, SimEStopAck) and m.estop_seq == s)
        a = gw.wait_for(lambda m, r=cmd.req_id: isinstance(m, SimAck) and m.req_id == r)
        if a.ok and _final(gw, cmd.run_id, 5).phase != "aborted":
            leaks += 1
        while runner.run is not None or runner._next_cmd is not None:
            time.sleep(0.001)
    assert leaks == 0


def test_stale_queued_command_is_not_started(env):
    """serve_forever re-checks the epoch of a queued command right before starting it."""
    runner = make_runner(env, moving_policy(dx=0.2))
    cmd = run_skill_cmd(max_steps=20)
    runner.handle_ctl(cmd)
    assert runner._next_cmd is cmd
    runner.epoch += 1  # the epoch moved on after the command was accepted
    t, _ = in_thread(runner.serve_forever)
    end = time.monotonic() + 2
    while runner._next_cmd is not None and time.monotonic() < end:
        time.sleep(0.005)
    time.sleep(0.05)
    runner._shutdown.set()
    t.join(2)
    st = [m for m in runner.published if isinstance(m, SimStatus) and m.run_id == cmd.run_id]
    assert st and st[-1].phase == "aborted"
    assert env.actions == []


def test_pause_right_after_run_starts_it_paused(served, env):
    """A pause sent right after run_skill (queued, not started) is no longer refused with
    'no run': the run starts paused, never moves, and resumes on ctl.resume."""
    runner, gw = served(policy=moving_policy(dx=0.05))
    cmd = run_skill_cmd(max_steps=40)
    gw.send(cmd)
    gw.send(Pause(req_id="q_pause", epoch=0))
    pa = gw.wait_for(lambda m: isinstance(m, SimAck) and m.req_id == "q_pause")
    assert pa.ok, pa.error
    p = gw.wait_for(lambda m: isinstance(m, SimStatus) and m.run_id == cmd.run_id and m.phase == "paused")
    assert p.step == 0
    n = len(env.actions)
    time.sleep(0.3)
    assert all(np.all(a[:6] == 0) for a in env.actions)  # only the controlled-stop hold steps
    assert len(env.actions) == n
    gw.send(Resume(req_id="q_res", epoch=0))
    assert gw.wait_for(lambda m: isinstance(m, SimAck) and m.req_id == "q_res").ok
    assert _final(gw, cmd.run_id).phase in ("done", "failed")
    assert any(np.any(a[:3] != 0) for a in env.actions[n:])


def test_abort_of_queued_run_is_reported(env):
    runner = make_runner(env, moving_policy())
    cmd = run_skill_cmd()
    runner.handle_ctl(cmd)
    from nod_protocol import Abort

    runner.handle_ctl(Abort(req_id="q_a", epoch=0))
    assert runner._next_cmd is None
    st = [m for m in runner.published if isinstance(m, SimStatus) and m.run_id == cmd.run_id]
    assert st[-1].phase == "aborted"


def test_estop_seq_reuse_after_gateway_restart_still_stops(env):
    """The gateway's e-stop counter restarts with the process. A reused seq must still stop a run."""
    runner = make_runner(env, moving_policy(dx=0.1), hz=50)
    for seq in (1, 3):  # the old gateway's stops
        runner.request_estop(seq)
        runner._poll_estop()
        runner._handle_estops()
    assert runner.epoch == 2
    cmd = run_skill_cmd(epoch=2, max_steps=200)
    t, _ = in_thread(runner.run_policy_skill, cmd)
    while len(env.actions) < 10:
        time.sleep(0.002)
    runner.request_estop(3)  # same seq as an old stop, but a run is in flight: it must stop
    t.join(5)
    assert not t.is_alive()
    assert runner.last_result.phase == "aborted"
    assert runner.epoch == 3
    n = len(env.actions)
    time.sleep(0.2)
    assert len(env.actions) == n


def test_load_scene_forgets_old_seqs_and_publishes_epoch(env):
    runner = make_runner(env, moving_policy(dx=0.1))
    runner.request_estop(1)
    runner._poll_estop()
    runner._handle_estops()
    assert runner.epoch == 1
    runner.handle_ctl(LoadScene(req_id="q_l", scene_id="libero_goal_v1", init_index=0, epoch=0))
    idle = [m for m in runner.published if isinstance(m, SimStatus) and m.phase == "idle"]
    assert idle and idle[-1].epoch == 1  # a restarted gateway adopts this
    runner.request_estop(1)  # the new gateway's first stop: a new stop, not a duplicate
    runner._poll_estop()
    runner._handle_estops()
    assert runner.epoch == 2


def test_duplicate_while_idle_reacks_with_fresh_times(env):
    runner = make_runner(env, None)
    runner.request_estop(5)
    runner._poll_estop()
    runner._handle_estops()
    time.sleep(0.02)
    runner.request_estop(5)
    runner._poll_estop()
    runner._handle_estops()
    a = [m for m in runner.published if isinstance(m, SimEStopAck)]
    assert [x.estop_seq for x in a] == [5, 5] and runner.epoch == 1
    assert a[1].t_rx > a[0].t_rx
