#!/usr/bin/env python
"""Nod Day-1 spike: chain LIBERO-Goal skills in ONE env with no reset (return-home between skills).
Run from ~/Isaac-GR00T with the LIBERO venv python. The policy server must be running on 127.0.0.1:5555."""
# ruff: noqa: E401, E741
import argparse, json, os, time
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import torch
from libero.libero import benchmark
from libero.libero.utils import get_libero_path
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv
from gr00t.policy.server_client import PolicyClient

SUITE = benchmark.get_benchmark_dict()["libero_goal"]()
TASKS = {SUITE.get_task(i).name: SUITE.get_task(i) for i in range(SUITE.get_num_tasks())}
def bddl(t): return os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
GOALS = {n: robosuite_parse_problem(bddl(t))["goal_state"] for n, t in TASKS.items()}
CHAINS = {
    "A": ["put_the_wine_bottle_on_the_rack", "open_the_middle_drawer_of_the_cabinet",
          "put_the_cream_cheese_in_the_bowl", "push_the_plate_to_the_front_of_the_stove"],
    "B": ["put_the_bowl_on_the_plate", "put_the_cream_cheese_in_the_bowl"],
}
KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
LANG = "annotation.human.action.task_description"
FRAMES, STEP_MS = [], []

def act(xyz=(0.0, 0.0, 0.0), grip=1.0):
    a = {f"action.{k}": np.zeros(1, dtype=np.float32) for k in KEYS}
    for k, v in zip("xyz", xyz):
        a[f"action.{k}"][0] = v
    a["action.gripper"][0] = grip          # policy units: 1.0 = open (LiberoEnv rescales, binarizes, inverts)
    return a

def step(env, a):
    t0 = time.perf_counter()
    obs, _, _, _, info = env.step(a)
    STEP_MS.append((time.perf_counter() - t0) * 1000)
    if FRAMES is not None:
        FRAMES.append(obs["video.image"])
    return obs, info

def batch(obs):
    b = {}
    for k, v in obs.items():
        if k.startswith("video."):
            b[k] = np.ascontiguousarray(v, dtype=np.uint8)[None, None]
        elif k.startswith("state."):
            b[k] = np.asarray(v, dtype=np.float32).reshape(1, 1, -1)
        else:
            b[k] = (v,)
    return b

def eef(obs):
    return np.array([obs["state.x"][0], obs["state.y"][0], obs["state.z"][0]], dtype=np.float64)

def set_skill(env, name, obs):
    env._task_description = TASKS[name].language
    env._env.env.parsed_problem["goal_state"] = GOALS[name]
    obs[LANG] = TASKS[name].language

def run_skill(env, client, name, obs, budget, n_steps, lat):
    set_skill(env, name, obs)
    if env._env.check_success():
        return "already_true", 0, obs
    steps = 0
    while steps < budget:
        t0 = time.perf_counter()
        chunk, _ = client.get_action(batch(obs))
        lat.append((time.perf_counter() - t0) * 1000)
        for t in range(n_steps):
            a = {k: np.asarray(v[0, t], dtype=np.float32).reshape(-1) for k, v in chunk.items()}
            obs, info = step(env, a)
            steps += 1
            if info["success"]:
                return "success", steps, obs
            if steps >= budget:
                break
    return "timeout", steps, obs

def return_home(env, obs, home, max_steps=80):
    """Scripted (not GR00T): open gripper, lift to home height, move to home xyz. Rotation untouched."""
    for _ in range(5):
        obs, _ = step(env, act(grip=1.0))
    for phase in ("lift", "move"):
        for _ in range(max_steps):
            p = eef(obs)
            tgt = np.array([p[0], p[1], max(home[2], p[2])]) if phase == "lift" else home
            d = tgt - p
            if np.linalg.norm(d) < 0.01:
                break
            obs, _ = step(env, act(np.clip(d / 0.05, -1.0, 1.0), grip=1.0))
    return obs

def save_video(path, frames):
    import cv2
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    for f in frames:
        vw.write(np.ascontiguousarray(f[..., ::-1]))
    vw.release()

def main():
    global FRAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="A", help="A, B, or single:<task_name>")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--n-action-steps", type=int, default=8)
    ap.add_argument("--no-home", action="store_true")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--video-dir", default="")
    ap.add_argument("--out", default=os.path.expanduser("~/nod_spike/chain_results.jsonl"))
    args = ap.parse_args()
    chain = [args.chain.split(":", 1)[1]] if args.chain.startswith("single:") else CHAINS[args.chain]
    first = TASKS[chain[0]]
    init_states = torch.load(os.path.join(get_libero_path("init_states"), first.problem_folder,
                                          first.init_states_file), weights_only=False)
    env = LiberoEnv(task_bddl_file=bddl(first), task_description=first.language)
    client = PolicyClient(host="127.0.0.1", port=args.port, timeout_ms=5000)
    assert client.ping(), "policy server not reachable on 127.0.0.1:%d" % args.port
    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)
    lat, rows = [], []
    for trial in range(args.trials):
        FRAMES = [] if args.video_dir else None
        env._env.reset()
        obs = env._process_observation(env._env.set_init_state(init_states[trial % len(init_states)]))
        for _ in range(10):                                   # settle, as LeRobot's LIBERO wrapper does
            obs, _ = step(env, act(grip=1.0))
        home, res = eef(obs), []
        for i, name in enumerate(chain):
            if i > 0 and not args.no_home:
                obs = return_home(env, obs, home)
            t0 = time.time()
            status, steps, obs = run_skill(env, client, name, obs, args.budget, args.n_action_steps, lat)
            res.append({"skill": name, "status": status, "steps": steps, "wall_s": round(time.time() - t0, 2)})
            if status != "success":
                break
        ok = len(res) == len(chain) and all(r["status"] == "success" for r in res)
        row = {"trial": trial, "chain": args.chain, "home": not args.no_home, "results": res, "chain_ok": ok}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if args.video_dir and FRAMES:
            save_video(os.path.join(args.video_dir, f"{args.chain.replace(':', '_')}_{trial:02d}.mp4"), FRAMES)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("\n=== SUMMARY chain=%s trials=%d home=%s ===" % (args.chain, len(rows), not args.no_home))
    for i, name in enumerate(chain):
        att = sum(1 for r in rows if len(r["results"]) > i)
        okc = sum(1 for r in rows if len(r["results"]) > i and r["results"][i]["status"] == "success")
        stp = [r["results"][i]["steps"] for r in rows if len(r["results"]) > i and r["results"][i]["status"] == "success"]
        print(f"step {i+1} {name}: {okc}/{att} = {okc/max(att,1):.0%}  median steps {np.median(stp) if stp else float('nan'):.0f}")
    full = sum(r["chain_ok"] for r in rows)
    print(f"full chain: {full}/{len(rows)} = {full/max(len(rows),1):.0%}")
    l, s = np.array(lat), np.array(STEP_MS)
    print(f"policy call ms: p50 {np.percentile(l,50):.1f} p95 {np.percentile(l,95):.1f} max {l.max():.1f} (n={len(l)})")
    print(f"env.step ms:    p50 {np.percentile(s,50):.1f} p95 {np.percentile(s,95):.1f} (n={len(s)})")

if __name__ == "__main__":
    main()
