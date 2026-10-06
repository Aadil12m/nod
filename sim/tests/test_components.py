"""Recorder, oracle, scripted skills, GR00T client, capture and scene helpers on CPU."""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest
import zmq
from nod_protocol import load_skills, registry_atoms

from conftest import REGISTRY, make_runner, moving_policy, run_skill_cmd
from nod_sim.capture import capture_frames, encode_jpeg
from nod_sim.env_api import PolicyError, PolicyTimeout, to_robosuite
from nod_sim.fakes import FakeEnv
from nod_sim.oracle import atom_to_state, state_to_atom, world_state
from nod_sim.policy import Gr00tPolicy, batch_observation, chunk_from_response, pack, unpack
from nod_sim.recorder import load_recording
from nod_sim.scene import GOAL_REFS, instruction_from_name, scene_fingerprint, strip_blocks
from nod_sim.scripted import ScriptError, make_script, scripted_skill
from nod_sim.skills import swap_skill


# ----------------------------------------------------------------------------- recorder


def test_recorder_writes_mock_library_format(env, tmp_path):
    runner = make_runner(env, None, record_dir=tmp_path)
    res = runner.run_policy_skill(run_skill_cmd("bowl_on_plate"))
    assert res.success and res.record_dir
    man, arr = load_recording(res.record_dir)
    # the gateway MockRunner reads skill_id, success/outcome, steps and xpos/xquat/qpos
    assert man["skill_id"] == "bowl_on_plate" and man["success"] and man["outcome"] == "success"
    assert man["actor"] == "scripted" and man["steps"] == res.steps
    T = man["frames"]
    assert T == res.steps + 1
    assert arr["xpos"].shape[0] == T and arr["xpos"].dtype == np.float32
    assert arr["xquat"].shape[1:] == (arr["xpos"].shape[1], 4)
    assert arr["qpos"].shape[0] == T and arr["action"].shape == (T, 7)
    assert arr["sim_state"].shape[0] == T
    events = [json.loads(line) for line in open(f"{res.record_dir}/events.jsonl")]
    assert events[0]["event"] == "start" and events[-1]["event"] == "end"
    patch = json.loads(open(f"{res.record_dir}/model_patch.json").read())
    assert "body_pos" in patch
    # the saved states replay exactly
    env2 = FakeEnv()
    env2.set_state(arr["sim_state"][-1])
    assert env2.eval_atom("on(akita_black_bowl_1,plate_1)")


def test_recorder_marks_estop_and_stop_frames(env, tmp_path):
    runner = make_runner(env, moving_policy(dx=0.1), record_dir=tmp_path)
    env.on_step = lambda e: runner.request_estop(2) if e.step_count == 10 else None
    res = runner.run_policy_skill(run_skill_cmd())
    man, arr = load_recording(res.record_dir)
    assert man["outcome"] == "aborted" and man["phase"] == "aborted"
    assert int(arr["stop"].sum()) == 4
    events = [json.loads(line)["event"] for line in open(f"{res.record_dir}/events.jsonl")]
    assert "estop" in events and "chunk" in events


# ----------------------------------------------------------------------------- oracle


def test_atom_conversions():
    assert atom_to_state("on(akita_black_bowl_1,plate_1)") == ["on", "akita_black_bowl_1", "plate_1"]
    assert atom_to_state("open(wooden_cabinet_1_middle_region)") == ["open", "wooden_cabinet_1_middle_region"]
    assert state_to_atom(["On", "wine_bottle_1", "wine_rack_1_top_region"]) == "on(wine_bottle_1,wine_rack_1_top_region)"


def test_world_state_reports_every_registry_atom():
    specs = load_skills(REGISTRY)
    atoms = registry_atoms(specs)
    w = world_state(FakeEnv(), atoms, ["akita_black_bowl_1", "plate_1", "nonexistent_obj"])
    assert set(w.atoms) == set(atoms)
    assert not any(w.atoms.values())  # every atom is false in the initial layout
    assert "nonexistent_obj" not in w.objects
    assert w.eef.gripper == "open" and len(w.eef.pos) == 3


def test_world_state_leaves_unknown_atoms_out():
    w = world_state(FakeEnv(), ["on(mug_1,plate_1)", "on(akita_black_bowl_1,plate_1)"])
    assert "on(mug_1,plate_1)" not in w.atoms and "on(akita_black_bowl_1,plate_1)" in w.atoms


def test_libero_eval_atom_with_a_stub_problem():
    from nod_sim.oracle import libero_eval_atom

    class Problem:
        object_states_dict = {"akita_black_bowl_1": 1, "plate_1": 2}
        calls = []

        def _eval_predicate(self, state):
            self.calls.append(state)
            return True

    p = Problem()
    assert libero_eval_atom(p, "on(akita_black_bowl_1,plate_1)") is True
    assert p.calls == [["on", "akita_black_bowl_1", "plate_1"]]
    assert libero_eval_atom(p, "on(mug,plate_1)") is None


# ----------------------------------------------------------------------------- scripted


def drive(env, gen, limit=2000):
    n = 0
    try:
        while n < limit:
            env.step(next(gen))
            n += 1
    except StopIteration as stop:
        return stop.value, n
    raise AssertionError("script did not finish")


@pytest.mark.parametrize("skill", ["bowl_on_plate", "cheese_in_bowl", "bowl_to_user", "bowl_on_cabinet",
                                   "bowl_on_stove", "bottle_on_cabinet", "plate_to_stove_front", "open_middle_drawer"])
def test_scripted_stand_ins_reach_their_goal_on_fake(skill):
    env = FakeEnv()
    spec = {s.skill_id: s for s in load_skills(REGISTRY)}[skill]
    env.set_skill(spec.policy.instruction, spec.goal_ref)
    ok, steps = drive(env, scripted_skill(skill)(env))
    assert ok and env.check_success(), skill
    assert steps <= spec.policy.max_steps


def test_return_home_and_handover_retreat():
    env = FakeEnv()
    env.eef = np.array([0.1, 0.1, 0.95])
    ok, _ = drive(env, make_script("return_home", env))
    assert ok and np.allclose(env.eef_pos(), env.home_eef, atol=0.011)
    env.eef = np.array([0.12, 0.1, 0.93])
    ok, _ = drive(env, make_script("handover_retreat", env))
    assert ok and np.allclose(env.eef_pos(), env.home_eef, atol=0.011)


def test_scripted_unknown_and_reset():
    with pytest.raises(ScriptError):
        scripted_skill("stove_on")
    with pytest.raises(ScriptError):
        make_script("reset_scene", FakeEnv())
    with pytest.raises(ScriptError):
        make_script("dance", FakeEnv())


def test_scripted_actor_refuses_skill_without_script(env):
    runner = make_runner(env, None)
    cmd = run_skill_cmd("bowl_on_plate").model_copy(update={"skill_id": "stove_on",
                                                           "goal_ref": "libero_goal/turn_on_the_stove"})
    runner.handle_ctl(cmd)
    assert not runner.published[-1].ok and "no scripted controller" in runner.published[-1].error


def test_unverified_scripts_refused_when_disallowed(env):
    with pytest.raises(ScriptError, match="not verified"):
        scripted_skill("open_middle_drawer", allow_unverified=False)
    assert scripted_skill("bowl_on_plate", allow_unverified=False)
    runner = make_runner(env, None, allow_unverified_scripts=False)
    runner.handle_ctl(run_skill_cmd("open_middle_drawer"))
    assert not runner.published[-1].ok and "not verified" in runner.published[-1].error
    runner.handle_ctl(run_skill_cmd("bowl_on_plate"))
    assert runner.published[-1].ok


def test_scripted_run_respects_limits(env):
    from nod_protocol import Limits

    runner = make_runner(env, None)
    tight = run_skill_cmd("bowl_on_plate", limits=Limits(eef_box=((-0.4, -0.4, 1.0), (0.2, 0.4, 1.35))))
    res = runner.run_policy_skill(tight)  # the pick descends below z=1.0
    assert res.fault == "eef_out_of_box"


# ----------------------------------------------------------------------------- goal swap helper


def test_swap_skill_on_gr00t_style_env():
    class Problem:
        parsed_problem = {"goal_state": [["on", "a", "b"]]}

    class OffScreen:
        env = Problem()

    class Gr00tLiberoEnv:
        _env = OffScreen()
        _task_description = "old"

    e = Gr00tLiberoEnv()
    swap_skill(e, "put the bowl on the plate", [["on", "akita_black_bowl_1", "plate_1"]])
    assert e._task_description == "put the bowl on the plate"
    assert e._env.env.parsed_problem["goal_state"] == [["on", "akita_black_bowl_1", "plate_1"]]


# ----------------------------------------------------------------------------- scene helpers


BDDL = """(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language Put the bowl on the plate)
    (:regions
      (plate_region
          (:target main_table)
          (:ranges ((0.04 -0.03 0.06 -0.01)))
      )
      (handover_region
          (:target main_table)
          (:ranges ((0.08 0.06 0.16 0.14)))
      )
    )
  (:obj_of_interest
    akita_black_bowl_1
  )
  (:goal
    (And (On akita_black_bowl_1 plate_1))
  )
)
"""


def test_fingerprint_ignores_language_goal_and_interest():
    other = BDDL.replace("Put the bowl on the plate", "Turn on the stove").replace(
        "(On akita_black_bowl_1 plate_1)", "(Turnon flat_stove_1)").replace("akita_black_bowl_1\n  )", "flat_stove_1\n  )")
    assert scene_fingerprint(BDDL) == scene_fingerprint(other)
    assert "(:goal" not in strip_blocks(BDDL) and "(:language" not in strip_blocks(BDDL)
    moved = BDDL.replace("0.04 -0.03", "0.05 -0.03")
    assert scene_fingerprint(moved) != scene_fingerprint(BDDL)
    no_region = scene_fingerprint(BDDL, drop_regions=("handover_region",))
    assert no_region != scene_fingerprint(BDDL)
    assert "handover_region" not in strip_blocks(BDDL.replace(BDDL[BDDL.index("      (handover_region"):BDDL.index("    )\n  (:obj")], ""))


def test_goal_refs_cover_the_registry():
    refs = {s.goal_ref for s in load_skills(REGISTRY) if s.kind == "policy"}
    assert refs <= set(GOAL_REFS)
    assert GOAL_REFS["libero_goal/put_the_bowl_on_the_plate"][0] == "put the bowl on the plate"
    assert instruction_from_name("turn_on_the_stove.bddl") == "turn on the stove"
    for s in load_skills(REGISTRY):
        if s.kind == "policy":
            assert GOAL_REFS[s.goal_ref][0] == s.policy.instruction, s.skill_id


def test_to_robosuite_gripper_convention():
    a = np.array([0.1, 0, 0, 0, 0, 0, 1.0])
    assert to_robosuite(a)[-1] == -1.0  # policy 1.0 (open) -> robosuite -1 (open)
    a[-1] = 0.0
    assert to_robosuite(a)[-1] == 1.0
    a[-1] = 0.3
    assert to_robosuite(a)[-1] == 1.0  # binarised as LiberoEnv does


# ----------------------------------------------------------------------------- capture


def test_capture_frames_jpeg():
    frames = capture_frames(FakeEnv(), ["agentview"], 40, 30, step=5)
    assert frames[0].w == 40 and frames[0].h == 30 and frames[0].step == 5
    from PIL import Image
    import io

    assert Image.open(io.BytesIO(frames[0].jpeg)).size == (40, 30)
    assert encode_jpeg(np.zeros((8, 8, 3), np.uint8))[:2] == b"\xff\xd8"


# ----------------------------------------------------------------------------- GR00T client


class FakeGr00tServer:
    """REP server speaking Isaac-GR00T's msgpack-numpy wire format (server_client.py)."""

    def __init__(self, ctx, horizon=16, delay_s=0.0, error=None):
        self.sock = ctx.socket(zmq.REP)
        self.port = self.sock.bind_to_random_port("tcp://127.0.0.1")
        self.horizon, self.delay_s, self.error = horizon, delay_s, error
        self.requests = []
        self.t = threading.Thread(target=self.loop, daemon=True)
        self.stop = False
        self.t.start()

    def loop(self):
        import time

        while not self.stop:
            if not self.sock.poll(50):
                continue
            req = unpack(self.sock.recv())
            self.requests.append(req)
            if self.delay_s:
                time.sleep(self.delay_s)
            if req["endpoint"] == "ping":
                self.sock.send(pack({"status": "ok", "message": "Server is running"}))
            elif self.error:
                self.sock.send(pack({"error": self.error}))
            else:
                T = self.horizon
                act = {f"action.{k}": np.full((1, T, 1), i / 10, np.float32)
                       for i, k in enumerate(["x", "y", "z", "roll", "pitch", "yaw", "gripper"])}
                self.sock.send(pack([act, {}]))

    def close(self):
        self.stop = True
        self.t.join(1)
        self.sock.close(0)


def test_gr00t_client_round_trip():
    ctx = zmq.Context()
    srv = FakeGr00tServer(ctx)
    try:
        pol = Gr00tPolicy("127.0.0.1", srv.port, timeout_ms=2000, context=ctx)
        assert pol.ping()
        obs = FakeEnv().observation()
        obs["video.image"] = np.zeros((256, 256, 3), np.uint8)
        chunk = pol.get_action(obs)
        assert chunk.shape == (16, 7) and chunk.dtype == np.float32
        assert np.allclose(chunk[0], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        req = srv.requests[-1]
        o = req["data"]["observation"]
        assert req["endpoint"] == "get_action" and req["data"]["options"] is None
        assert o["video.image"].shape == (1, 1, 256, 256, 3) and o["video.image"].dtype == np.uint8
        assert o["state.x"].shape == (1, 1, 1) and o["state.x"].dtype == np.float32
        assert o["state.gripper"].shape == (1, 1, 2)
        assert list(o["annotation.human.action.task_description"]) == [""]
        pol.close()
    finally:
        srv.close()
        ctx.term()


def test_gr00t_client_timeout_and_error():
    ctx = zmq.Context()
    slow = FakeGr00tServer(ctx, delay_s=0.5)
    err = FakeGr00tServer(ctx, error="CUDA out of memory")
    try:
        pol = Gr00tPolicy("127.0.0.1", slow.port, timeout_ms=100, context=ctx)
        with pytest.raises(PolicyTimeout):
            pol.get_action(FakeEnv().observation())
        pol.close()
        pol = Gr00tPolicy("127.0.0.1", err.port, timeout_ms=2000, context=ctx)
        with pytest.raises(PolicyError, match="CUDA out of memory"):
            pol.get_action(FakeEnv().observation())
        pol.close()
    finally:
        slow.close()
        err.close()
        ctx.term()


def test_gr00t_client_with_runner(env):
    ctx = zmq.Context()
    srv = FakeGr00tServer(ctx, horizon=16)
    try:
        pol = Gr00tPolicy("127.0.0.1", srv.port, timeout_ms=2000, context=ctx)
        runner = make_runner(env, pol)
        res = runner.run_policy_skill(run_skill_cmd(max_steps=16))
        assert res.fault == "budget" and res.steps == 16 and len(res.policy_ms) == 2  # 8 of 16 per chunk
        pol.close()
    finally:
        srv.close()
        ctx.term()


def test_chunk_from_response_validation():
    with pytest.raises(PolicyError):
        chunk_from_response({"action.x": np.zeros((1, 4, 1))})
    b = batch_observation({"video.image": np.zeros((2, 2, 3)), "state.x": [0.1], "lang": "hi"})
    assert b["video.image"].dtype == np.uint8 and b["lang"] == ("hi",)


def test_from_url():
    p = Gr00tPolicy.from_url("tcp://policy:5599", timeout_ms=10)
    assert (p.host, p.port) == ("policy", 5599)
    p.close()
