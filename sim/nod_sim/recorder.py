"""Per-run recordings (design 4.2.4 recorder.py; mock library format in 4.4).

One directory per run:

    manifest.json     summary: skill, actor, outcome, steps, fault, timings, scene ids, final atoms
    poses.npz         xpos [T,nb,3] f32, xquat [T,nb,4] f32 (w,x,y,z), qpos [T,nq] f32,
                      action [T,7] f32 (action applied to reach frame t; zeros at t=0),
                      sim_state [T,ns] f64 (flattened MuJoCo state, for rerender.py),
                      step [T] i64 (global sim step), stop [T] bool (controlled-stop steps),
                      chunk_starts [K] i64 (frame index where each policy chunk began)
    model_patch.json  fixture body_pos/body_quat LIBERO wrote at reset (the viewer applies it)
    events.jsonl      one JSON object per line: start, chunk, pause, resume, fault, estop, end
    policy_cam.mp4    optional 256 px agentview, only when video=True (needs imageio-ffmpeg)

Frame 0 is the state before the first action. The gateway MockRunner replays frame `run.step`
(it reads `skill_id`, `success`/`outcome` and `steps` from the manifest).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "nod.recording/1"


def _stats(xs: list[float]) -> dict[str, float] | None:
    if not xs:
        return None
    a = np.asarray(xs, dtype=np.float64)
    return {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2), "max": round(float(a.max()), 2)}


class EpisodeRecorder:
    def __init__(
        self,
        out_dir: str | Path,
        *,
        run_id: str,
        skill_id: str,
        kind: str,
        actor: str,
        env: Any,
        epoch: int = 0,
        meta: dict[str, Any] | None = None,
        video: bool = False,
        record_state: bool = True,
    ) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id, self.skill_id, self.kind, self.actor = run_id, skill_id, kind, actor
        self.env = env
        self.meta = dict(meta or {})
        self.video = video
        self.record_state = record_state
        self.t0 = time.time()
        self.info = env.episode_info()
        self._xpos: list[np.ndarray] = []
        self._xquat: list[np.ndarray] = []
        self._qpos: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._state: list[np.ndarray] = []
        self._step: list[int] = []
        self._stop: list[bool] = []
        self._chunks: list[int] = []
        self._frames: list[np.ndarray] = []
        self._events = open(self.dir / "events.jsonl", "w")
        (self.dir / "model_patch.json").write_text(json.dumps(self.info.model_patch, indent=1) + "\n")
        self.event("start", run_id=run_id, skill_id=skill_id, kind=kind, actor=actor, epoch=epoch, **self.meta)
        self._frame(None, np.zeros(7, np.float32), stop=False)

    # ------------------------------------------------------------------ during the run

    def event(self, name: str, **data: Any) -> None:
        rec = {"t_ms": int(time.time() * 1000), "event": name, **data}
        self._events.write(json.dumps(rec, default=_jsonable) + "\n")
        self._events.flush()

    def chunk(self, run_step: int, policy_ms: float, epoch: int) -> None:
        self._chunks.append(len(self._xpos))
        self.event("chunk", step=run_step, policy_ms=round(policy_ms, 1), epoch=epoch)

    def step(self, sim_step: int, action: np.ndarray, env: Any, stop: bool = False) -> None:
        self._frame(sim_step, np.asarray(action, np.float32).reshape(7), stop=stop)

    def _frame(self, sim_step: int | None, action: np.ndarray, stop: bool) -> None:
        xpos, xquat, qpos = self.env.body_poses()
        self._xpos.append(np.asarray(xpos, np.float32))
        self._xquat.append(np.asarray(xquat, np.float32))
        self._qpos.append(np.asarray(qpos, np.float32))
        self._action.append(action)
        if self.record_state:
            self._state.append(np.asarray(self.env.get_state(), np.float64))
        self._step.append(-1 if sim_step is None else int(sim_step))
        self._stop.append(stop)
        if self.video:
            self._frames.append(self.env.render("agentview", 256, 256))

    # ------------------------------------------------------------------ end

    def finish(self, result: Any, world: Any = None) -> Path:
        res = asdict(result) if is_dataclass(result) else dict(result)
        arrays: dict[str, np.ndarray] = {
            "xpos": np.stack(self._xpos),
            "xquat": np.stack(self._xquat),
            "qpos": np.stack(self._qpos),
            "action": np.stack(self._action),
            "step": np.asarray(self._step, np.int64),
            "stop": np.asarray(self._stop, bool),
            "chunk_starts": np.asarray(self._chunks, np.int64),
        }
        if self._state:
            arrays["sim_state"] = np.stack(self._state)
        np.savez_compressed(self.dir / "poses.npz", **arrays)
        files = ["manifest.json", "poses.npz", "model_patch.json", "events.jsonl"]
        if self.video and self._frames:
            if _write_mp4(self.dir / "policy_cam.mp4", self._frames):
                files.append("policy_cam.mp4")
        outcome = "success" if res.get("success") else ("aborted" if res.get("phase") == "aborted" else "failure")
        manifest = {
            "format": FORMAT,
            "run_id": self.run_id,
            "skill_id": self.skill_id,
            "kind": self.kind,
            "actor": self.actor,  # "policy" = GR00T closed loop, "scripted" = waypoint controller
            "outcome": outcome,
            "success": bool(res.get("success")),
            "phase": res.get("phase"),
            "fault": res.get("fault"),
            "error": res.get("error"),
            "steps": int(res.get("steps", 0)),
            "frames": int(arrays["xpos"].shape[0]),
            "stopped_at_step": res.get("stopped_at_step"),
            "already_true": bool(res.get("already_true")),
            "discarded_chunks": int(res.get("discarded_chunks", 0)),
            "policy_ms": _stats(res.get("policy_ms") or []),
            "step_ms": _stats(res.get("step_ms") or []),
            "wall_s": res.get("wall_s"),
            "hz": 20,
            "scene_id": self.info.scene_id,
            "model_hash": self.info.model_hash,
            "mj_ver": self.info.mj_ver,
            "body_names": self.info.body_names,
            "dyn_body_ids": self.info.dyn_body_ids,
            "created": datetime.fromtimestamp(self.t0, timezone.utc).isoformat(timespec="seconds"),
            "meta": self.meta,
            "world_end": None if world is None else {"atoms": dict(world.atoms), "t_sim": world.t_sim},
            "files": files,
        }
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=_jsonable) + "\n")
        self.event("end", outcome=outcome, phase=res.get("phase"), fault=res.get("fault"), steps=res.get("steps"))
        self._events.close()
        return self.dir


def load_recording(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read a recording directory back: (manifest, arrays)."""
    p = Path(path)
    manifest = json.loads((p / "manifest.json").read_text())
    with np.load(p / "poses.npz") as z:
        arrays = {k: z[k] for k in z.files}
    return manifest, arrays


def _write_mp4(path: Path, frames: list[np.ndarray], fps: int = 20) -> bool:
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1) as w:
            for f in frames:
                w.append_data(np.ascontiguousarray(f, dtype=np.uint8))
        return True
    except Exception:  # imageio-ffmpeg missing or no encoder: the video is optional
        return False


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
