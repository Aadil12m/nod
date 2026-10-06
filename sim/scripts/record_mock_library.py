#!/usr/bin/env python
"""Record a small mock library from the real LIBERO-Goal scene with the scripted actor (tier T4).

Every run happens in ONE session with no reset (as the live product does), so a single fixture
placement (model_patch) is valid for every recording and for the exported scene. Writes:

    <out>/<NN>_<skill>_<outcome>/{manifest.json, poses.npz, model_patch.json, events.jsonl}
    <scene-dir>/{scene.xml, layout.json}   input for `nod_viewer.export_scene compile`

These recordings come from the scripted waypoint controller, not GR00T (no GPU here); their
manifests say `"actor": "scripted"`. Run in the LIBERO venv:

    MUJOCO_GL=osmesa python sim/scripts/record_mock_library.py --out recordings --scene-dir /tmp/scene
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "viewer"))

from nod_protocol import RunScripted, RunSkill, load_skills, new_id  # noqa: E402

from nod_sim.runner import SimRunner  # noqa: E402
from nod_sim.scene import make_session_env  # noqa: E402

PLAN = [  # (skill or script, max_steps, estop_at_step)
    ("bowl_on_plate", 400, None),
    ("return_home", 160, None),
    ("cheese_in_bowl", 400, None),
    ("return_home", 160, 12),         # e-stop mid-motion: aborted, never resumed
    ("return_home", 160, None),
    ("bowl_to_user", 400, None),      # near user: action scale 0.6 (R14)
    ("handover_retreat", 160, None),  # fails in LIBERO from the handover pose (sim/README.md)
]
SCRIPTS = {"return_home", "handover_retreat", "close_middle_drawer", "reset_scene"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "recordings")
    ap.add_argument("--scene-dir", type=Path, default=Path("/tmp/nod_scene"))
    ap.add_argument("--init-index", type=int, default=0)
    args = ap.parse_args()
    specs = {s.skill_id: s for s in load_skills(REPO / "registry" / "skills.yaml")}
    env = make_session_env(init_index=args.init_index)
    tmp = args.out / "_tmp_runs"
    runner = SimRunner(env, None, None, hz=None, registry=REPO / "registry" / "skills.yaml", record_dir=tmp,
                       actor="scripted", allow_unverified_scripts=False, hb_timeout_ms=None)
    summary = []
    for i, (name, max_steps, estop_at) in enumerate(PLAN):
        if name in SCRIPTS:
            cmd = RunScripted(req_id=new_id("q"), run_id=new_id("r"), epoch=runner.epoch, script=name, max_steps=max_steps)
        else:
            s = specs[name]
            cmd = RunSkill(req_id=new_id("q"), run_id=new_id("r"), epoch=runner.epoch, skill_id=name,
                           instruction=s.policy.instruction, goal_ref=s.goal_ref, max_steps=max_steps,
                           action_scale=0.6 if s.near_user else 1.0)
        hook = None
        if estop_at is not None:
            start = env.step_count
            orig = env.step

            def hook(a, _orig=orig, _start=start, _seq=i + 1):
                out = _orig(a)
                if env.step_count - _start == estop_at:
                    runner.request_estop(_seq)
                return out

            env.step = hook
        try:
            res = runner.run_scripted(cmd) if isinstance(cmd, RunScripted) else runner.run_policy_skill(cmd)
        finally:
            if hook is not None:
                env.step = orig
        outcome = "success" if res.success else res.phase
        dst = args.out / f"{i:02d}_{res.skill_id}_{outcome}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(res.record_dir, dst)
        summary.append({"dir": dst.name, "skill_id": res.skill_id, "phase": res.phase, "success": res.success,
                        "fault": res.fault, "steps": res.steps})
        print(json.dumps(summary[-1]), flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    from nod_viewer.export_scene import dump

    dump(args.scene_dir, env=env)
    (args.out / "library.json").write_text(json.dumps({"scene_id": "libero_goal_v1", "actor": "scripted",
                                                       "init_index": args.init_index, "runs": summary}, indent=1))


if __name__ == "__main__":
    main()
