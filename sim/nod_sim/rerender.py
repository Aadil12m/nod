"""Offline re-render of a recorded run at video resolution (design 2.1 row 5, 4.2.4 rerender.py).

Replays the flattened MuJoCo states saved in `poses.npz["sim_state"]` with set_state + forward
(no physics is stepped, so the motion is exactly what happened) and renders each frame from a
chosen camera at e.g. 1920x1080. The fixture placement in `model_patch.json` is written into
the model first, because LIBERO places fixtures at reset and that placement is not part of the
state vector.

    python -m nod_sim.rerender recordings/<run_dir> --out run.mp4 --camera frontview --w 1920 --h 1080

Runs in the LIBERO venv (MUJOCO_GL=egl on the VM; osmesa works on CPU but is slow at 1080p).
Label the footage "re-rendered from logged sim state" (design 9.2).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def apply_model_patch(sim, patch: dict) -> int:
    """Write body_pos/body_quat by body name into the live model. Returns bodies patched."""
    m = sim.model
    n = 0
    for key, attr in (("body_pos", "body_pos"), ("body_quat", "body_quat")):
        for name, val in (patch.get(key) or {}).items():
            try:
                bid = m.body_name2id(name)
            except Exception:
                continue
            getattr(m, attr)[bid] = np.asarray(val, dtype=np.float64)
            n += 1
    return n


def ensure_offscreen_size(sim, w: int, h: int) -> None:
    """Grow MuJoCo's offscreen buffer (vis.global offwidth/offheight) and rebuild the context."""
    raw = sim.model._model if hasattr(sim.model, "_model") else sim.model
    if raw.vis.global_.offwidth >= w and raw.vis.global_.offheight >= h:
        return
    raw.vis.global_.offwidth = max(w, raw.vis.global_.offwidth)
    raw.vis.global_.offheight = max(h, raw.vis.global_.offheight)
    from robosuite.utils.binding_utils import MjRenderContextOffscreen

    old = getattr(sim, "_render_context_offscreen", None)
    device = getattr(old, "device_id", -1) if old is not None else -1
    MjRenderContextOffscreen(sim, device_id=device)  # registers itself as sim._render_context_offscreen


def frames(env, states: np.ndarray, camera: str, w: int, h: int, every: int = 1) -> Iterator[np.ndarray]:
    """Upright RGB frames for every `every`-th saved state. `env` is a LiberoSession."""
    for i in range(0, len(states), max(1, every)):
        env.set_state(states[i])
        yield env.render(camera, w, h)


def rerender(run_dir: str | Path, out: str | Path, camera: str = "frontview", w: int = 1920, h: int = 1080,
             every: int = 1, fps: int = 20, env=None, limit: int | None = None) -> Path:
    run = Path(run_dir)
    with np.load(run / "poses.npz") as z:
        if "sim_state" not in z.files:
            raise ValueError(f"{run} has no sim_state (recorded with record_state=False)")
        states = z["sim_state"]
    if limit:
        states = states[:limit]
    if env is None:
        from .scene import make_session_env

        env = make_session_env()
    patch_file = run / "model_patch.json"
    if patch_file.exists():
        apply_model_patch(env.sim, json.loads(patch_file.read_text()))
    ensure_offscreen_size(env.sim, w, h)
    out = Path(out)
    if out.suffix.lower() in (".mp4", ".mov", ".mkv"):
        import imageio.v2 as imageio

        with imageio.get_writer(out, fps=max(1, fps // max(1, every)), codec="libx264", quality=8,
                                macro_block_size=1) as wr:
            for f in frames(env, states, camera, w, h, every):
                wr.append_data(f)
    else:
        out.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        for k, f in enumerate(frames(env, states, camera, w, h, every)):
            Image.fromarray(f).save(out / f"{k:05d}.png")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True, help=".mp4 file, or a directory for PNG frames")
    ap.add_argument("--camera", default="frontview")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--every", type=int, default=1, help="render every n-th recorded frame")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    print(rerender(a.run_dir, a.out, a.camera, a.w, a.h, a.every, limit=a.limit))


if __name__ == "__main__":
    main()
