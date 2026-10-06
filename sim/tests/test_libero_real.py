"""Tests against the real LIBERO-Goal scene. Skipped unless LIBERO, robosuite and MuJoCo import.

Run them in the LIBERO venv (sim/README.md):
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_CONFIG_PATH=... .venv-libero/bin/python -m pytest tests/test_libero_real.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
pytest.importorskip("robosuite")
pytest.importorskip("libero.libero")

from nod_protocol import load_skills, registry_atoms  # noqa: E402

from conftest import REGISTRY, make_runner, run_scripted_cmd, run_skill_cmd  # noqa: E402
from nod_sim.oracle import world_state  # noqa: E402
from nod_sim.recorder import load_recording  # noqa: E402
from nod_sim.scene import LIBERO_GOAL_GOALS, load_goals, make_session_env, verify_single_scene  # noqa: E402


@pytest.fixture(scope="module")
def session():
    env = make_session_env(init_index=0)
    yield env
    env.close()


def test_single_scene_claim():
    r = verify_single_scene()
    assert r["n_tasks"] == 10 and r["identical"], r["task_hashes"]
    assert r["nod_matches"], "the Nod session BDDL differs from LIBERO-Goal beyond the handover region"


def test_goal_table_matches_bddl_files():
    g = load_goals()
    for name, goal in LIBERO_GOAL_GOALS.items():
        assert g[f"libero_goal/{name}"] == goal
    assert g["nod/bring_the_bowl_to_me"] == [["on", "akita_black_bowl_1", "main_table_handover_region"]]


def test_initial_world_and_observation(session):
    session.reset(0)
    atoms = registry_atoms(load_skills(REGISTRY))
    w = world_state(session, atoms)
    assert set(w.atoms) == set(atoms), "an atom the registry uses is not evaluable in the session"
    assert not any(w.atoms.values())
    obs = session.observation()
    assert obs["video.image"].shape == (256, 256, 3) and obs["video.image"].dtype == np.uint8
    assert len(obs["state.gripper"]) == 2
    img = session.render("frontview", 64, 48)
    assert img.shape == (48, 64, 3) and img.std() > 1


def test_scripted_chain_with_goal_swap_and_estop(session, tmp_path):
    session.reset(0)
    runner = make_runner(session, None, record_dir=tmp_path)
    r1 = runner.run_policy_skill(run_skill_cmd("bowl_on_plate"))
    assert r1.success, r1
    r2 = runner.run_scripted(run_scripted_cmd("return_home"))
    assert r2.success
    r3 = runner.run_policy_skill(run_skill_cmd("cheese_in_bowl"))
    assert r3.success, r3
    assert session.eval_atom("on(akita_black_bowl_1,plate_1)")  # still true after the swap
    man, arr = load_recording(r3.record_dir)
    assert man["success"] and arr["xpos"].shape[1] == session.sim.model.nbody
    # e-stop while carrying: the controlled stop holds the gripper closed
    session.reset(1)
    runner2 = make_runner(session, None)
    session_step = session.step

    def step(a):
        out = session_step(a)
        if session.step_count == 60:  # after the grasp, while lifting
            runner2.request_estop(1)
        return out

    session.step = step
    try:
        res = runner2.run_policy_skill(run_skill_cmd("bowl_on_plate"))
    finally:
        session.step = session_step
    assert res.phase == "aborted"
    ack = [m for m in runner2.published if m.type == "sim.estop_ack"][-1]
    assert ack.stopped_at_step == res.stopped_at_step and ack.t_halt - ack.t_rx <= 200
    assert session.gripper_is_closed(), "the controlled stop must hold the gripper command"
    assert session.object_pos("akita_black_bowl_1")[2] > 0.93, "the bowl was dropped"


def test_episode_header_has_fixture_patch(session):
    info = session.episode_info()
    assert info.body_names[0] == "world" and info.mj_ver.startswith("3.3")
    assert "wooden_cabinet_1_main" in info.model_patch["body_pos"]
    assert any(info.body_names[i] == "akita_black_bowl_1_main" for i in info.dyn_body_ids)


def test_rerender_small(session, tmp_path):
    from nod_sim.rerender import rerender

    session.reset(0)
    runner = make_runner(session, None, record_dir=tmp_path)
    res = runner.run_scripted(run_scripted_cmd("return_home"))
    out = rerender(res.record_dir, tmp_path / "frames", camera="frontview", w=96, h=64, every=10, env=session)
    assert len(list(out.glob("*.png"))) >= 1
