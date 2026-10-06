"""Nightly evaluation driver (design 4.2.5): every skill alone, then the chains, in one process.

    python -m nod_eval.run --registry registry/skills.yaml --episodes 20 --chains A,B --out /mnt/results/<date>

It reuses `nod_sim.runner.SimRunner` without the gateway: the same interruptible loop, limits,
goal swapping and recorder the live product uses, called in-process (`run_policy_skill`,
`run_scripted`) with no sockets and no pacing (`hz=None`; the policy round trip is the only wait).

Episodes
* alone: `env.reset(init_index)`, run the skill once, judged by the oracle (BDDL goal).
* chain: `env.reset(init_index)`, then for each step: scripted `return_home` (not before the
  first step), the skill. The chain stops at the first failed step, as the live product would.
  A step's in-chain outcome is counted for every step after the first (the first step starts
  from a fresh table, which is what "alone" measures).
* `init_index` is `(init_offset + episode) % 50`, the LIBERO-Goal init states.

Output (`--out`):
    run_meta.json      configuration, skipped skills and why, timing
    episodes.jsonl     one row per skill run (alone or chain step) and one per chain episode
    episodes/<label>/<run_id>/...   recorder output (manifest.json, poses.npz, ...) unless --no-record
    frames/<label>/{pre,post}_<camera>.jpg   captures for cosmos_agree.py unless --frames none

Actors: `--actor policy` (default) talks to Isaac-GR00T policy servers (`--policy URL` for the
LIBERO-Goal checkpoint, `--policy-for CHECKPOINT=URL` for others such as our custom skill).
Skills whose checkpoint has no server are skipped and listed in run_meta.json. `--actor
scripted` evaluates the T4 waypoint controllers instead; the published stats then say so.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nod_protocol import RunScripted, RunSkill, SkillSpec, holds, load_skills, new_id, registry_atoms

from .chains import CHAINS, available_prefix, parse_chains

log = logging.getLogger("nod_eval")

N_INIT_STATES = 50
DEFAULT_FRAME_CAMERAS = ("frontview",)  # what the live Cosmos check sees (gateway cosmos_cameras)


@dataclass
class EvalConfig:
    episodes: int = 20
    chains: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in CHAINS.items()})
    skills: list[str] | None = None  # None: every enabled policy skill
    alone: bool = True
    actor: str = "policy"
    home_between: bool = True
    init_offset: int = 0
    record: bool = True
    frames: str = "pre+post"  # "none" | "post" | "pre+post"
    frame_cameras: tuple[str, ...] = DEFAULT_FRAME_CAMERAS
    frame_w: int = 640
    frame_h: int = 480
    max_steps: int | None = None  # overrides the registry budget (smoke runs)
    hz: float | None = None  # None: no pacing


@dataclass
class EvalContext:
    env: Any
    runner: Any
    specs: dict[str, SkillSpec]
    out: Path
    cfg: EvalConfig
    policies: dict[str, Any]  # checkpoint -> policy client (empty for the scripted actor)
    rows: list[dict] = field(default_factory=list)
    atoms: list[str] = field(default_factory=list)

    def write(self, row: dict) -> None:
        self.rows.append(row)
        with open(self.out / "episodes.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------- planning


def evaluable_skills(specs: dict[str, SkillSpec], cfg: EvalConfig, policies: dict[str, Any],
                     script_available: Callable[[str], bool]) -> tuple[list[str], dict[str, str]]:
    """Which policy skills can run with this actor, and why the others cannot."""
    ok, skipped = [], {}
    wanted = cfg.skills or [s.skill_id for s in specs.values() if s.kind == "policy" and s.enabled]
    for sid in wanted:
        spec = specs.get(sid)
        if spec is None:
            skipped[sid] = "not in the registry"
        elif spec.kind != "policy":
            skipped[sid] = f"kind {spec.kind}: only policy skills are evaluated"
        elif cfg.actor == "policy" and spec.policy.checkpoint not in policies:
            skipped[sid] = f"no policy server for checkpoint {spec.policy.checkpoint}"
        elif cfg.actor == "scripted" and not script_available(sid):
            skipped[sid] = "no scripted controller (or not allowed)"
        else:
            ok.append(sid)
    return ok, skipped


def run_skill_cmd(spec: SkillSpec, max_steps: int | None = None) -> RunSkill:
    p = spec.policy
    return RunSkill(req_id=new_id("q"), run_id=new_id("r"), epoch=0, skill_id=spec.skill_id,
                    instruction=p.instruction, goal_ref=spec.goal_ref, n_action_steps=p.n_action_steps,
                    max_steps=max_steps or p.max_steps, action_scale=0.6 if spec.near_user else 1.0)


# ---------------------------------------------------------------------------- episodes


def _atoms(ctx: EvalContext) -> dict[str, bool]:
    from nod_sim.oracle import world_state

    return dict(world_state(ctx.env, ctx.atoms).atoms)


def _capture(ctx: EvalContext, label: str, when: str) -> dict[str, str]:
    if ctx.cfg.frames == "none" or (when == "pre" and ctx.cfg.frames != "pre+post"):
        return {}
    from nod_sim.capture import capture_frames

    d = ctx.out / "frames" / label
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for f in capture_frames(ctx.env, list(ctx.cfg.frame_cameras), ctx.cfg.frame_w, ctx.cfg.frame_h):
        p = d / f"{when}_{f.camera}.jpg"
        p.write_bytes(f.jpeg)
        out[f.camera] = str(p.relative_to(ctx.out))
    return out


def _set_record(ctx: EvalContext, label: str) -> None:
    ctx.runner.record_dir = (ctx.out / "episodes" / label) if ctx.cfg.record else None


def _rel(ctx: EvalContext, p: str | None) -> str | None:
    if not p:
        return None
    try:
        return str(Path(p).relative_to(ctx.out))
    except ValueError:
        return p


def run_one(ctx: EvalContext, skill_id: str, label: str, **row_extra: Any) -> dict:
    """Run one skill from the current state; returns (and writes) its episode row."""
    spec = ctx.specs[skill_id]
    if ctx.cfg.actor == "policy":
        ctx.runner.policy = ctx.policies[spec.policy.checkpoint]
    atoms_pre = _atoms(ctx)
    pre_ok = all(holds(t, atoms_pre) for t in spec.pre)
    frames = {"pre": _capture(ctx, label, "pre")}
    _set_record(ctx, label)
    t0 = time.monotonic()
    res = ctx.runner.run_policy_skill(run_skill_cmd(spec, ctx.cfg.max_steps))
    wall = time.monotonic() - t0
    ctx.runner.published.clear()
    atoms_post = _atoms(ctx)
    frames["post"] = _capture(ctx, label, "post")
    effects_ok = all(holds(t, atoms_post) for t in spec.effects)
    hz = getattr(ctx.env, "hz", 20) or 20
    row = {
        "label": label,
        "skill_id": skill_id,
        "actor": res.actor,
        "success": bool(res.success),
        "phase": res.phase,
        "fault": res.fault,
        "error": res.error,
        "steps": res.steps,
        "sim_s": round(res.steps / hz, 3),
        "wall_s": round(wall, 3),
        "already_true": res.already_true,
        "pre_ok": pre_ok,
        "effects_ok": effects_ok,
        "atoms_pre": atoms_pre,
        "atoms_post": atoms_post,
        "policy_ms_p50": _pct(res.policy_ms, 50),
        "policy_ms_p95": _pct(res.policy_ms, 95),
        "frames": frames,
        "record_dir": _rel(ctx, res.record_dir),
        **row_extra,
    }
    ctx.write(row)
    log.info("%s %s: %s in %d steps%s", label, skill_id, "ok" if row["success"] else "FAIL", res.steps,
             f" ({res.fault})" if res.fault else "")
    return row


def run_home(ctx: EvalContext, label: str) -> dict:
    _set_record(ctx, label)
    res = ctx.runner.run_scripted(RunScripted(req_id=new_id("q"), run_id=new_id("r"), epoch=0, script="return_home"))
    ctx.runner.published.clear()
    return {"success": bool(res.success), "steps": res.steps, "fault": res.fault}


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    import numpy as np

    return round(float(np.percentile(np.asarray(xs), q)), 1)


def init_index(cfg: EvalConfig, episode: int) -> int:
    return (cfg.init_offset + episode) % N_INIT_STATES


def run_alone(ctx: EvalContext, skill_id: str, episode: int) -> dict:
    ii = init_index(ctx.cfg, episode)
    ctx.env.reset(ii)
    return run_one(ctx, skill_id, f"alone_{skill_id}_{episode:03d}", kind="alone", chain=None,
                   episode=episode, step=0, init_index=ii)


def run_chain(ctx: EvalContext, name: str, steps: list[str], episode: int) -> dict:
    ii = init_index(ctx.cfg, episode)
    ctx.env.reset(ii)
    results, sim_s = [], 0.0
    hz = getattr(ctx.env, "hz", 20) or 20
    for i, sid in enumerate(steps):
        label = f"chain{name}_{episode:03d}_{i}_{sid}"
        home = None
        if i > 0 and ctx.cfg.home_between:
            home = run_home(ctx, label + "_home")
            sim_s += home["steps"] / hz
        row = run_one(ctx, sid, label, kind="chain", chain=name, episode=episode, step=i, init_index=ii, home=home)
        sim_s += row["sim_s"]
        results.append(row["success"])
        if not row["success"]:
            break
    ok = len(results) == len(steps) and all(results)
    summary = {"kind": "chain_summary", "chain": name, "episode": episode, "init_index": ii, "skill_ids": steps,
               "reached": len(results), "step_success": results, "success": ok, "sim_s": round(sim_s, 3)}
    ctx.write(summary)
    log.info("chain %s episode %d: %s (%d/%d steps)", name, episode, "ok" if ok else "FAIL", sum(results), len(steps))
    return summary


def evaluate(env: Any, cfg: EvalConfig, out: Path, *, registry: str | Path, policies: dict[str, Any] | None = None,
             runner_kw: dict | None = None) -> dict:
    """Run the whole evaluation; returns the run_meta dict (also written to out/run_meta.json)."""
    from nod_sim.runner import SimRunner
    from nod_sim.scripted import ScriptError, scripted_skill

    policies = policies or {}
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes.jsonl").write_text("")
    kw = dict(hz=cfg.hz, registry=registry, hb_timeout_ms=None, publish_poses=False,
              actor=cfg.actor, allow_unverified_scripts=True)
    kw.update(runner_kw or {})
    runner = SimRunner(env, None, None, **kw)
    specs = {s.skill_id: s for s in load_skills(registry)}

    def script_ok(sid: str) -> bool:
        try:
            scripted_skill(sid, runner.allow_unverified_scripts)
            return True
        except ScriptError:
            return False

    skills, skipped = evaluable_skills(specs, cfg, policies, script_ok)
    chains: dict[str, list[str]] = {}
    chain_notes: dict[str, str] = {}
    for name, steps in cfg.chains.items():
        runnable, _ = evaluable_skills(specs, EvalConfig(skills=steps, actor=cfg.actor), policies, script_ok)
        prefix = available_prefix(steps, set(runnable))
        if len(prefix) < len(steps):
            chain_notes[name] = f"truncated to {prefix}: {steps[len(prefix)]} cannot run with this actor"
        if prefix:
            chains[name] = prefix
    ctx = EvalContext(env=env, runner=runner, specs=specs, out=out, cfg=cfg, policies=policies,
                      atoms=registry_atoms(list(specs.values())))
    meta = {
        "format": "nod.eval/1",
        "actor": cfg.actor,
        "episodes": cfg.episodes,
        "skills": skills,
        "skipped": skipped,
        "chains": chains,
        "chain_notes": chain_notes,
        "home_between": cfg.home_between,
        "init_offset": cfg.init_offset,
        "max_steps": cfg.max_steps,
        "registry": str(registry),
        "job_id": os.environ.get("NOD_JOB_ID"),
        "git_sha": os.environ.get("NOD_GIT_SHA"),
        "host": platform.node(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    for sid, why in skipped.items():
        log.warning("skipping %s: %s", sid, why)
    t0 = time.monotonic()
    try:
        if cfg.alone:
            for sid in skills:
                for ep in range(cfg.episodes):
                    run_alone(ctx, sid, ep)
        for name, steps in chains.items():
            for ep in range(cfg.episodes):
                run_chain(ctx, name, steps, ep)
    finally:
        runner.close()
        meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta["wall_s"] = round(time.monotonic() - t0, 1)
        meta["rows"] = len(ctx.rows)
        (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


# ---------------------------------------------------------------------------- CLI


def _policies(args, specs: list[SkillSpec]) -> dict[str, Any]:
    """checkpoint -> Gr00tPolicy. --policy serves every LIBERO-Goal checkpoint id in the registry."""
    from nod_sim.policy import Gr00tPolicy

    urls: dict[str, str] = {}
    for s in specs:
        if s.kind == "policy" and not s.custom and args.policy:
            urls[s.policy.checkpoint] = args.policy
    for item in args.policy_for or []:
        ck, _, url = item.partition("=")
        if not url:
            raise SystemExit(f"--policy-for needs CHECKPOINT=URL, got {item!r}")
        urls[ck] = url
    clients: dict[str, Any] = {}
    by_url: dict[str, Any] = {}
    for ck, url in urls.items():
        if url not in by_url:
            c = Gr00tPolicy.from_url(url, timeout_ms=args.policy_timeout_ms)
            if not c.ping():
                log.error("policy server %s (%s) does not answer; its skills are skipped", url, ck)
                by_url[url] = None
            else:
                by_url[url] = c
        if by_url[url] is not None:
            clients[ck] = by_url[url]
    return clients


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Nod nightly eval: skills alone and in chains")
    ap.add_argument("--registry", default="registry/skills.yaml")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--chains", default="A,B", help="'A,B', '' for none, or name=skill+skill")
    ap.add_argument("--skills", default=None, help="comma list of skill ids to run alone (default: all policy skills)")
    ap.add_argument("--no-alone", action="store_true", help="only run the chains")
    ap.add_argument("--out", required=True)
    ap.add_argument("--actor", choices=["policy", "scripted"], default="policy")
    ap.add_argument("--policy", default=os.environ.get("NOD_POLICY", "tcp://127.0.0.1:5555"),
                    help="policy server for the LIBERO-Goal checkpoint")
    ap.add_argument("--policy-for", action="append", metavar="CHECKPOINT=URL",
                    help="policy server for another checkpoint (e.g. the custom bowl_to_user)")
    ap.add_argument("--policy-timeout-ms", type=int, default=10000)
    ap.add_argument("--no-home", action="store_true", help="no return_home between chain steps")
    ap.add_argument("--init-offset", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--frames", choices=["none", "post", "pre+post"], default="pre+post")
    ap.add_argument("--frame-size", default="640x480")
    ap.add_argument("--lazy-cameras", action="store_true", help="render policy cameras only on chunk requests")
    ap.add_argument("--allow-unverified-scripts", action="store_true",
                    help="scripted actor: also run stand-ins not verified in LIBERO")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    w, h = (int(x) for x in a.frame_size.lower().split("x"))
    cfg = EvalConfig(episodes=a.episodes, chains=parse_chains(a.chains),
                     skills=[s.strip() for s in a.skills.split(",") if s.strip()] if a.skills else None,
                     alone=not a.no_alone, actor=a.actor, home_between=not a.no_home, init_offset=a.init_offset,
                     record=not a.no_record, frames=a.frames, frame_w=w, frame_h=h, max_steps=a.max_steps)
    specs = load_skills(a.registry)
    policies = _policies(a, specs) if a.actor == "policy" else {}
    from nod_sim.scene import make_session_env

    env = make_session_env(init_index=a.init_offset, lazy_cameras=a.lazy_cameras)
    try:
        meta = evaluate(env, cfg, Path(a.out), registry=a.registry, policies=policies,
                        runner_kw={"policy_timeout_ms": a.policy_timeout_ms,
                                   "allow_unverified_scripts": a.allow_unverified_scripts or a.actor == "policy"})
    finally:
        env.close()
    print(json.dumps({k: meta[k] for k in ("skills", "skipped", "chains", "rows", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()
