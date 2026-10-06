"""SimRunner safety behaviour on CPU with FakeEnv/FakePolicy (design 5.3, 5.5, 5.6)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from nod_protocol import Limits, SimEStopAck, SimStatus

from conftest import in_thread, make_runner, moving_policy, run_skill_cmd
from nod_sim.env_api import PolicyError
from nod_sim.fakes import FakePolicy
from nod_sim.runner import CONTROLLED_STOP_STEPS


def acks(runner):
    return [m for m in runner.published if isinstance(m, SimEStopAck)]


def statuses(runner):
    return [m for m in runner.published if isinstance(m, SimStatus)]


def wait_until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.002)
    raise AssertionError("condition not met in time")


# ----------------------------------------------------------------------------- e-stop


def test_estop_halts_within_one_step_then_controlled_stop(env):
    """A stop requested mid-motion allows at most the step already in progress, then 4 hold steps."""
    runner = make_runner(env, moving_policy(dx=0.3, grip=0.0), hz=20)
    t, out = in_thread(runner.run_policy_skill, run_skill_cmd(max_steps=400))
    wait_until(lambda: env.step_count >= 15)
    t_req = time.monotonic()
    runner.request_estop(7)
    t.join(5)
    res = out[0]
    assert res.phase == "aborted" and res.fault is None and not res.success
    after = [a for a, ts in zip(env.actions, env.step_times) if ts >= t_req]
    moving_after = [a for a in after if np.any(a[:6] != 0)]
    assert len(moving_after) <= 1, "more than the in-progress step moved after the stop"
    stop_steps = after[-CONTROLLED_STOP_STEPS:]
    assert len(after) - len(moving_after) == CONTROLLED_STOP_STEPS
    for a in stop_steps:
        assert np.all(a[:6] == 0.0)
        assert a[6] == 0.0  # gripper command held closed: nothing is dropped
    ack = acks(runner)[-1]
    assert ack.estop_seq == 7 and ack.epoch == 1
    assert ack.stopped_at_step == res.stopped_at_step
    assert ack.t_halt - ack.t_rx <= 50, "sim halt took longer than one 50 ms control step"
    # physics stepping halts: nothing else is stepped afterwards
    n = len(env.actions)
    time.sleep(0.1)
    assert len(env.actions) == n


def test_estop_latency_against_pacing_and_inference(env):
    """The stop is seen within one control period even while waiting on a slow policy chunk."""
    gate = threading.Event()
    policy = moving_policy(dx=0.1, gate=gate)
    runner = make_runner(env, policy, hz=20)
    t, out = in_thread(runner.run_policy_skill, run_skill_cmd())
    wait_until(lambda: policy.calls >= 1)  # blocked inside the policy call
    time.sleep(0.05)
    t0 = time.monotonic()
    runner.request_estop(1)
    t.join(2)
    assert time.monotonic() - t0 < 0.05
    assert out[0].phase == "aborted"
    gate.set()


def test_estop_bumps_epoch_and_discards_late_chunk(env):
    """A chunk that was in flight at the stop arrives late and is never executed."""
    gate = threading.Event()
    marker = np.zeros((16, 7), np.float32)
    marker[:, 1] = 0.77  # a value no other action uses
    marker[:, 6] = 1.0
    policy = FakePolicy(lambda _o, _i: marker, gate=gate)
    runner = make_runner(env, policy, hz=None)
    t, out = in_thread(runner.run_policy_skill, run_skill_cmd())
    wait_until(lambda: policy.calls >= 1)
    runner.request_estop(3)
    t.join(2)
    gate.set()  # the late chunk is returned now
    time.sleep(0.05)
    res = out[0]
    assert res.phase == "aborted"
    assert runner.epoch == 1
    assert res.discarded_chunks >= 1
    assert not any(np.isclose(a[1], 0.77) for a in env.actions)


def test_queued_actions_of_old_epoch_are_dropped(env):
    """Actions already queued from a chunk are not executed after a stop (no resume after estop)."""
    runner = make_runner(env, moving_policy(dx=0.2, horizon=16), hz=20)
    t, out = in_thread(runner.run_policy_skill, run_skill_cmd())
    wait_until(lambda: env.step_count >= 3)  # mid-chunk: 5 of 8 actions still queued
    runner.request_estop(4)
    t.join(2)
    n = len(env.actions)
    time.sleep(0.2)
    assert len(env.actions) == n
    assert out[0].phase == "aborted"


def test_duplicate_estop_acked_without_second_epoch_bump(env):
    runner = make_runner(env, None)
    runner.request_estop(5)
    runner._poll_estop()
    runner._handle_estops()
    runner.request_estop(5)
    runner._poll_estop()
    runner._handle_estops()
    a = acks(runner)
    assert [x.estop_seq for x in a] == [5, 5]
    assert runner.epoch == 1 and a[0].epoch == a[1].epoch == 1
    assert a[0].stopped_at_step is None  # nothing was running


# ----------------------------------------------------------------------------- limits (5.3)


def test_workspace_box_zeroes_translation_then_faults_after_three(env):
    box = Limits(eef_box=((-0.40, -0.40, 0.85), (0.20, 0.40, 1.19)))  # home z 1.179, close to the top
    up = np.zeros(7, np.float32)
    up[2] = 1.0  # +5 cm per step: predicted 1.229 > 1.19
    up[6] = 1.0
    runner = make_runner(env, FakePolicy.constant(up))
    z0 = env.eef_pos()[2]
    res = runner.run_policy_skill(run_skill_cmd(limits=box))
    assert res.phase == "failed" and res.fault == "eef_out_of_box"
    # the two clipped steps ran with zero translation, the third violation raised before stepping
    assert res.steps == 2
    assert all(np.all(a[:3] == 0.0) for a in env.actions)
    assert env.eef_pos()[2] == pytest.approx(z0)
    stop = env.actions[-CONTROLLED_STOP_STEPS:]
    assert len(env.actions) == 2 + CONTROLLED_STOP_STEPS and all(np.all(a[:6] == 0) for a in stop)


def test_workspace_violation_counter_resets_when_back_inside(env):
    box = Limits(eef_box=((-0.40, -0.40, 0.85), (0.20, 0.40, 1.19)))

    def fn(_obs, i):
        c = np.zeros((4, 7), np.float32)
        c[:, 6] = 1.0
        c[:2, 2] = 1.0  # two violating steps, then two steps that stay inside
        return c

    runner = make_runner(env, FakePolicy(fn, horizon=4))
    res = runner.run_policy_skill(run_skill_cmd(limits=box, max_steps=40, n_action_steps=4))
    assert res.fault == "budget"  # never three in a row, so the run is only stopped by its budget
    assert res.steps == 40


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1.5, -1.01])
def test_nan_or_out_of_range_action_faults(env, bad):
    a = np.zeros(7, np.float32)
    a[6] = 1.0
    chunk = np.tile(a, (8, 1))
    chunk[3, 4] = bad
    runner = make_runner(env, FakePolicy(lambda _o, _i: chunk))
    res = runner.run_policy_skill(run_skill_cmd())
    assert res.phase == "failed" and res.fault == "nan_action"
    assert res.steps == 3  # the bad action itself was never stepped
    assert all(np.all(np.isfinite(x)) and np.all(np.abs(x) <= 1.0) for x in env.actions)


def test_bad_chunk_shape_is_nan_action(env):
    runner = make_runner(env, FakePolicy(lambda _o, _i: np.zeros((8, 6), np.float32)))
    res = runner.run_policy_skill(run_skill_cmd())
    assert res.fault == "nan_action"


def test_step_budget_fault(env):
    runner = make_runner(env, moving_policy(dx=0.0))
    res = runner.run_policy_skill(run_skill_cmd(max_steps=24))
    assert res.phase == "failed" and res.fault == "budget" and res.steps == 24
    assert len(env.actions) == 24 + CONTROLLED_STOP_STEPS


def test_policy_timeout_fault(env):
    policy = moving_policy(delay_s=1.0)
    runner = make_runner(env, policy, policy_timeout_ms=100)
    t0 = time.monotonic()
    res = runner.run_policy_skill(run_skill_cmd())
    assert time.monotonic() - t0 < 0.6
    assert res.phase == "failed" and res.fault == "policy_timeout"
    assert len(env.actions) == CONTROLLED_STOP_STEPS


def test_policy_error_is_reported_as_policy_timeout(env):
    runner = make_runner(env, FakePolicy(error=PolicyError("server error: boom")))
    res = runner.run_policy_skill(run_skill_cmd())
    assert res.fault == "policy_timeout" and "boom" in res.error


def test_action_scale_for_near_user_skills(env):
    runner = make_runner(env, moving_policy(dx=0.5))
    runner.run_policy_skill(run_skill_cmd("bowl_to_user", max_steps=8, action_scale=0.6))
    moving = [a for a in env.actions if a[0] != 0]
    assert moving and all(a[0] == pytest.approx(0.3) for a in moving)


# ----------------------------------------------------------------------------- watchdog (5.6)


def test_heartbeat_loss_pauses_then_resume_and_abort(env):
    runner = make_runner(env, moving_policy(dx=0.05), hz=20, hb_timeout_ms=100)
    runner.last_hb = time.monotonic()
    t, out = in_thread(runner.run_policy_skill, run_skill_cmd(max_steps=10_000))
    wait_until(lambda: any(s.phase == "paused" for s in statuses(runner)), timeout=3)
    paused = next(s for s in statuses(runner) if s.phase == "paused")
    assert paused.fault == "hb_lost"
    n = len(env.actions)
    time.sleep(0.15)
    assert len(env.actions) == n, "physics kept stepping while paused"
    # resume: heartbeat back plus ctl.resume (delivered through the run's request flags)
    runner.last_hb = time.monotonic()
    runner.run.resume_requested = True
    keep = threading.Event()

    def beat():
        while not keep.is_set():
            runner.last_hb = time.monotonic()
            time.sleep(0.02)

    hb = threading.Thread(target=beat, daemon=True)
    hb.start()
    wait_until(lambda: len(env.actions) > n + 3)
    assert statuses(runner)[-1].phase == "executing" or any(
        s.phase == "executing" for s in statuses(runner)[-5:])
    runner.run.abort_requested = True
    runner.stop_event.set()
    t.join(2)
    keep.set()
    assert out[0].phase == "aborted"


def test_no_heartbeat_watchdog_when_disabled(env):
    runner = make_runner(env, moving_policy(dx=0.0), hb_timeout_ms=None)
    res = runner.run_policy_skill(run_skill_cmd(max_steps=30))
    assert res.fault == "budget"
