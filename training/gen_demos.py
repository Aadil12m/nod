#!/usr/bin/env python
"""Scripted demonstrations for a custom skill (design 4.2.5, 9.1 week 2): HDF5 episodes.

    MUJOCO_GL=egl python training/gen_demos.py --skill bowl_to_user --n 80 --keep-success \
        --out training/data/bowl_to_user.hdf5

Each attempt: `reset(init_index)`, optionally set up a start state with unrecorded scripted skills
(`--start table|plate|plate_cheese|mix`; registry note: bowl_to_user is trained from the table
and from the plate, with or without cheese in the bowl), then `return_home` (unrecorded, as the
live product runs it before every skill), then the skill's scripted waypoint controller
(`nod_sim.scripted`, privileged poses) is recorded step by step until it finishes or
`--max-steps`. Success is the oracle (`check_success` on the skill's BDDL goal).

The file follows the LIBERO/robomimic HDF5 layout so LIBERO tools can read it:

    data                        attrs: env_name, problem_info (json), bddl_file_name, env_args (json), nod (json)
    data/demo_<i>               attrs: num_samples, init_index, start, success, instruction
        actions        [T, 7]   robosuite units (LIBERO convention: gripper -1 open, +1 close)
        actions_policy [T, 7]   GR00T policy units (what to_lerobot.py writes): xyz/rot deltas
                                in [-1, 1], gripper in [0, 1] with 1 = open
        obs/agentview_rgb    [T, H, W, 3] uint8, raw renderer order like LIBERO's own demos
                                (upside down; to_lerobot.py rotates 180 degrees)
        obs/eye_in_hand_rgb  [T, H, W, 3] uint8, same convention
        obs/ee_pos [T, 3], obs/ee_ori [T, 3] (axis-angle), obs/gripper_states [T, 2]
        states [T, S]          flattened sim state (replay with set_state)
        rewards [T], dones [T]  1 on the last frame of a successful demo

The observation at frame t is taken before action t is applied (the pairing GR00T trains on).
Runs on `nod_sim.scene.LiberoSession` (LIBERO venv) or `nod_sim.fakes.FakeEnv` (`--fake`, tests).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]

STARTS = ("table", "plate", "plate_cheese")
# Unrecorded set-up per start state (scripted skills, verified in LIBERO for chain B).
SETUP = {"table": [], "plate": ["bowl_on_plate"], "plate_cheese": ["bowl_on_plate", "cheese_in_bowl"]}


def skill_task(skill_id: str, registry: Path) -> tuple[str, str]:
    """(instruction, goal_ref) of a policy skill from the registry."""
    from nod_protocol import load_skills

    for s in load_skills(registry):
        if s.skill_id == skill_id:
            if s.kind != "policy":
                raise SystemExit(f"{skill_id} is not a policy skill")
            return s.policy.instruction, s.goal_ref
    raise SystemExit(f"{skill_id} is not in {registry}")


def run_unrecorded(env: Any, gen, max_steps: int = 600) -> None:
    for i, a in enumerate(gen):
        env.step(a)
        if i >= max_steps:
            break


def setup_start(env: Any, start: str, registry: Path) -> bool:
    """Bring the scene to a start state with scripted skills, then return home. False if it failed."""
    from nod_sim.scripted import make_script, scripted_skill

    for sid in SETUP[start]:
        instr, ref = skill_task(sid, registry)
        env.set_skill(instr, ref)
        run_unrecorded(env, scripted_skill(sid)(env))
        if not env.check_success():
            return False
    run_unrecorded(env, make_script("return_home", env))
    return True


def record_attempt(env: Any, skill_id: str, max_steps: int, noise: float, rng: np.random.Generator) -> dict:
    """Run the skill's scripted controller, recording observation-before-action pairs."""
    from nod_sim.env_api import to_robosuite
    from nod_sim.scripted import scripted_skill

    gen = scripted_skill(skill_id)(env)
    rec: dict[str, list] = {k: [] for k in ("agentview_rgb", "eye_in_hand_rgb", "ee_pos", "ee_ori", "gripper_states",
                                            "states", "actions", "actions_policy")}
    success = False
    for _ in range(max_steps):
        obs = env.observation()
        try:
            a = np.asarray(next(gen), dtype=np.float64).reshape(7).copy()
        except StopIteration:
            break
        if noise > 0:  # diversity: jitter the commanded translation, as executed and recorded
            a[:3] = np.clip(a[:3] + rng.normal(0.0, noise, 3), -1.0, 1.0)
        # raw renderer order (LIBERO HDF5 convention): undo the 180 degree turn of the policy view
        rec["agentview_rgb"].append(np.ascontiguousarray(np.asarray(obs["video.image"])[::-1, ::-1]))
        rec["eye_in_hand_rgb"].append(np.ascontiguousarray(np.asarray(obs["video.wrist_image"])[::-1, ::-1]))
        rec["ee_pos"].append([obs["state.x"][0], obs["state.y"][0], obs["state.z"][0]])
        rec["ee_ori"].append([obs["state.roll"][0], obs["state.pitch"][0], obs["state.yaw"][0]])
        rec["gripper_states"].append(np.asarray(obs["state.gripper"], dtype=np.float64).reshape(2))
        rec["states"].append(np.asarray(env.get_state(), dtype=np.float64).ravel())
        rec["actions_policy"].append(a.astype(np.float32))
        rec["actions"].append(np.asarray(to_robosuite(a), dtype=np.float32))
        success = bool(env.step(a))
    success = success or bool(env.check_success())
    return {"success": success, **rec}


def write_demo(grp, i: int, rec: dict, meta: dict) -> None:
    d = grp.create_group(f"demo_{i}")
    t = len(rec["actions"])
    d.attrs["num_samples"] = t
    for k, v in meta.items():
        d.attrs[k] = v
    d.create_dataset("actions", data=np.asarray(rec["actions"], np.float32))
    d.create_dataset("actions_policy", data=np.asarray(rec["actions_policy"], np.float32))
    d.create_dataset("states", data=np.asarray(rec["states"], np.float64))
    rewards = np.zeros(t, np.uint8)
    dones = np.zeros(t, np.uint8)
    if rec["success"] and t:
        rewards[-1] = dones[-1] = 1
    d.create_dataset("rewards", data=rewards)
    d.create_dataset("dones", data=dones)
    o = d.create_group("obs")
    for k in ("agentview_rgb", "eye_in_hand_rgb"):
        o.create_dataset(k, data=np.asarray(rec[k], np.uint8), compression="gzip", compression_opts=4)
    for k in ("ee_pos", "ee_ori", "gripper_states"):
        o.create_dataset(k, data=np.asarray(rec[k], np.float64))


def generate(env: Any, out: Path, *, skill_id: str, n: int, keep_success: bool, start: str = "mix",
             registry: Path = REPO / "registry" / "skills.yaml", max_steps: int = 400, noise: float = 0.0,
             seed: int = 0, init_offset: int = 0, log=print) -> dict:
    """Generate `n` attempts into `out` (HDF5). Returns a summary (also written next to it as .json)."""
    import h5py

    instr, goal_ref = skill_task(skill_id, registry)
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"skill_id": skill_id, "instruction": instr, "goal_ref": goal_ref, "attempts": 0, "setup_failed": 0,
               "successes": 0, "kept": 0, "by_start": {}, "frames": 0, "seed": seed, "noise": noise,
               "env": type(env).__name__}
    t0 = time.time()
    with h5py.File(out, "w") as f:
        grp = f.create_group("data")
        grp.attrs["env_name"] = "libero_goal_v1"
        from nod_sim.scene import SESSION_BDDL

        grp.attrs["bddl_file_name"] = str(SESSION_BDDL)
        grp.attrs["problem_info"] = json.dumps({"language_instruction": instr, "problem_name": "nod_libero_goal",
                                                "domain_name": "robosuite"})
        grp.attrs["env_args"] = json.dumps({"control_freq": 20, "controller": "OSC_POSE",
                                            "camera_px": int(getattr(env, "camera_px", 256))})
        grp.attrs["nod"] = json.dumps({"skill_id": skill_id, "goal_ref": goal_ref, "generator": "nod_sim.scripted",
                                       "gripper_policy_units": "1 = open"})
        kept = 0
        for i in range(n):
            ii = (init_offset + i) % 50
            st = pyrng.choice(STARTS) if start == "mix" else start
            env.reset(ii)
            summary["attempts"] += 1
            by = summary["by_start"].setdefault(st, {"attempts": 0, "successes": 0})
            by["attempts"] += 1
            if not setup_start(env, st, registry):
                summary["setup_failed"] += 1
                log(f"attempt {i}: set-up {st} failed (init {ii})")
                continue
            env.set_skill(instr, goal_ref)
            rec = record_attempt(env, skill_id, max_steps, noise, rng)
            ok = rec["success"]
            summary["successes"] += ok
            by["successes"] += ok
            log(f"attempt {i}: start={st} init={ii} steps={len(rec['actions'])} {'success' if ok else 'FAIL'}")
            if keep_success and not ok:
                continue
            write_demo(grp, kept, rec, {"init_index": ii, "start": st, "success": ok, "instruction": instr})
            kept += 1
            summary["frames"] += len(rec["actions"])
        grp.attrs["total"] = summary["frames"]
        summary["kept"] = kept
    summary["success_rate"] = round(summary["successes"] / max(1, summary["attempts"] - summary["setup_failed"]), 3)
    summary["wall_s"] = round(time.time() - t0, 1)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Scripted demos (HDF5) for fine-tuning a custom skill")
    ap.add_argument("--skill", default="bowl_to_user")
    ap.add_argument("--n", type=int, default=80, help="attempts")
    ap.add_argument("--keep-success", action="store_true", help="write only successful attempts")
    ap.add_argument("--start", choices=[*STARTS, "mix"], default="mix")
    ap.add_argument("--out", default="training/data/bowl_to_user.hdf5")
    ap.add_argument("--registry", default=str(REPO / "registry" / "skills.yaml"))
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--noise", type=float, default=0.0, help="std of translation jitter, policy units")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-offset", type=int, default=0)
    ap.add_argument("--camera-px", type=int, default=256, help="policy camera size (GR00T LIBERO: 256)")
    ap.add_argument("--fake", action="store_true", help="CPU kinematic FakeEnv instead of LIBERO (tests)")
    a = ap.parse_args(argv)
    if a.fake:
        from nod_sim.fakes import FakeEnv

        env = FakeEnv()
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
        from nod_sim.scene import make_session_env

        env = make_session_env(lazy_cameras=True, camera_px=a.camera_px)
    try:
        s = generate(env, Path(a.out), skill_id=a.skill, n=a.n, keep_success=a.keep_success, start=a.start,
                     registry=Path(a.registry), max_steps=a.max_steps, noise=a.noise, seed=a.seed,
                     init_offset=a.init_offset)
    finally:
        if hasattr(env, "close"):
            env.close()
    print(json.dumps(s, indent=1))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
