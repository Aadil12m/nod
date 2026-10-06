#!/usr/bin/env python
"""Convert gen_demos.py HDF5 episodes to a GR00T-ready LeRobot v2.1 dataset (design 4.2.5).

    python training/to_lerobot.py --in training/data/bowl_to_user.hdf5 --out training/data/bowl_to_user_lerobot

Layout (the same as Isaac-GR00T `demo_data/libero_demo` and the IPEC-COMMUNITY LIBERO datasets):

    meta/info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl, modality.json
    data/chunk-000/episode_000000.parquet
        observation.state [8]  ee_pos(3), ee_ori axis-angle(3), gripper qpos(2)   (float32)
        action [7]             GR00T policy units: deltas in [-1, 1], gripper [0, 1] with 1 = open
        timestamp, frame_index, episode_index, index, task_index
    videos/chunk-000/observation.images.image/episode_000000.mp4         agentview
    videos/chunk-000/observation.images.wrist_image/episode_000000.mp4   eye in hand

* Images are rotated 180 degrees (`[::-1, ::-1]`) to match what Isaac-GR00T's LiberoEnv feeds
  the policy; gen_demos stores raw renderer order like LIBERO's own HDF5 files.
* Video is H.264 (libx264, yuv420p) at 20 fps via imageio-ffmpeg. The NVIDIA LIBERO demo data
  is AV1; GR00T decodes either through torchcodec [unverified for our files on the GPU image].
* No-op frames are dropped as in the `*_no_noops` LIBERO datasets: a frame whose translation and
  rotation command is ~0 and whose gripper command equals the previous kept frame's.
* `meta/modality.json` is Isaac-GR00T `examples/LIBERO/modality.json` (embedded below, or copied
  from `--modality-json`). `meta/stats.json` is not written: GR00T's `generate_stats` computes it
  (with schema fingerprints) on first load.
* Merging with the original `libero_goal_no_noops` dataset ("old plus new data", design 9.1) is
  done at fine-tune time by passing both dataset paths; see finetune.sh.

Verified on CPU (training/tests): the parquet and videos round-trip, and Isaac-GR00T's own
`generate_stats`, `generate_rel_stats` and `LeRobotEpisodeLoader` (commit 51d4c89) read the
dataset, with video decoding swapped for imageio because GR00T's torchcodec decoder needs torch.
Not verified: torchcodec decoding our H.264 files and an actual fine-tune step (GPU VM).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np

FPS = 20
CHUNK = 1000
NOOP_EPS = 1e-4
STATE_NAMES = ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper", "gripper"]
ACTION_NAMES = ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]
VIDEO_KEYS = {"observation.images.image": "agentview_rgb", "observation.images.wrist_image": "eye_in_hand_rgb"}

# Isaac-GR00T examples/LIBERO/modality.json (commit 51d4c89), copied verbatim in content.
LIBERO_MODALITY = {
    "state": {"x": {"start": 0, "end": 1}, "y": {"start": 1, "end": 2}, "z": {"start": 2, "end": 3},
              "roll": {"start": 3, "end": 4}, "pitch": {"start": 4, "end": 5}, "yaw": {"start": 5, "end": 6},
              "gripper": {"start": 6, "end": 8}},
    "action": {"x": {"start": 0, "end": 1}, "y": {"start": 1, "end": 2}, "z": {"start": 2, "end": 3},
               "roll": {"start": 3, "end": 4}, "pitch": {"start": 4, "end": 5}, "yaw": {"start": 5, "end": 6},
               "gripper": {"start": 6, "end": 7}},
    "video": {"image": {"original_key": "observation.images.image"},
              "wrist_image": {"original_key": "observation.images.wrist_image"}},
    "annotation": {"human.action.task_description": {"original_key": "task_index"}},
}


def keep_mask(actions: np.ndarray, eps: float = NOOP_EPS) -> np.ndarray:
    """False for no-op frames (zero motion and unchanged gripper versus the previous kept frame)."""
    keep = np.ones(len(actions), dtype=bool)
    prev_grip = None
    for i, a in enumerate(actions):
        if prev_grip is not None and np.linalg.norm(a[:6]) < eps and a[6] == prev_grip:
            keep[i] = False
            continue
        prev_grip = a[6]
    return keep


def iter_demos(h5_path: Path, only_success: bool = True) -> Iterator[dict]:
    import h5py

    with h5py.File(h5_path, "r") as f:
        grp = f["data"]
        names = sorted(grp.keys(), key=lambda k: int(k.split("_")[1]))
        for name in names:
            d = grp[name]
            if only_success and not bool(d.attrs.get("success", True)):
                continue
            o = d["obs"]
            yield {
                "name": name,
                "instruction": str(d.attrs.get("instruction", "")),
                "actions": np.asarray(d["actions_policy"], np.float32),
                "state": np.concatenate([np.asarray(o["ee_pos"]), np.asarray(o["ee_ori"]),
                                         np.asarray(o["gripper_states"])], axis=1).astype(np.float32),
                "agentview_rgb": np.asarray(o["agentview_rgb"]),
                "eye_in_hand_rgb": np.asarray(o["eye_in_hand_rgb"]),
            }


def write_video(path: Path, frames: np.ndarray, fps: int = FPS) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                           ffmpeg_log_level="error", output_params=["-crf", "18"])
    try:
        for fr in frames:
            w.append_data(np.ascontiguousarray(fr, dtype=np.uint8))
    finally:
        w.close()


def _lowdim_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(), "count": [len(x)]}


def _image_stats(frames: np.ndarray, max_samples: int = 100) -> dict:
    """LeRobot v2.1 image stats: per channel, values in [0, 1], shape [3, 1, 1]."""
    idx = np.linspace(0, len(frames) - 1, min(max_samples, len(frames))).astype(int)
    x = frames[idx].astype(np.float64) / 255.0
    per = lambda v: [[[float(c)]] for c in v]  # noqa: E731
    return {"min": per(x.min((0, 1, 2))), "max": per(x.max((0, 1, 2))), "mean": per(x.mean((0, 1, 2))),
            "std": per(x.std((0, 1, 2))), "count": [len(idx)]}


def convert(h5_paths: list[Path], out: Path, *, drop_noops: bool = True, modality_json: Path | None = None,
            only_success: bool = True, overwrite: bool = False) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    tasks: dict[str, int] = {}
    episodes, ep_stats = [], []
    index = 0
    hw = None
    dropped = 0
    for h5 in h5_paths:
        for demo in iter_demos(h5, only_success):
            mask = keep_mask(demo["actions"]) if drop_noops else np.ones(len(demo["actions"]), bool)
            dropped += int((~mask).sum())
            n = int(mask.sum())
            if n == 0:
                continue
            ep = len(episodes)
            chunk = ep // CHUNK
            task_index = tasks.setdefault(demo["instruction"], len(tasks))
            state, action = demo["state"][mask], demo["actions"][mask]
            frame_index = np.arange(n, dtype=np.int64)
            ts = (frame_index / FPS).astype(np.float32)
            table = pa.table({
                "observation.state": pa.array(list(state), type=pa.list_(pa.float32())),
                "action": pa.array(list(action), type=pa.list_(pa.float32())),
                "timestamp": pa.array(ts, type=pa.float32()),
                "frame_index": pa.array(frame_index, type=pa.int64()),
                "episode_index": pa.array(np.full(n, ep, np.int64), type=pa.int64()),
                "index": pa.array(np.arange(index, index + n, dtype=np.int64), type=pa.int64()),
                "task_index": pa.array(np.full(n, task_index, np.int64), type=pa.int64()),
            })
            dp = out / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"
            dp.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dp)
            stats = {"observation.state": _lowdim_stats(state), "action": _lowdim_stats(action),
                     "timestamp": _lowdim_stats(ts), "frame_index": _lowdim_stats(frame_index),
                     "episode_index": _lowdim_stats(np.full(n, ep)), "index": _lowdim_stats(np.arange(index, index + n)),
                     "task_index": _lowdim_stats(np.full(n, task_index))}
            for key, src in VIDEO_KEYS.items():
                frames = demo[src][mask][:, ::-1, ::-1]  # rotate 180 degrees: the policy's view
                hw = frames.shape[1:3]
                write_video(out / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{ep:06d}.mp4", frames)
                stats[key] = _image_stats(frames)
            episodes.append({"episode_index": ep, "tasks": [demo["instruction"]], "length": n})
            ep_stats.append({"episode_index": ep, "stats": stats})
            index += n
    if not episodes:
        raise SystemExit("no episodes to convert")
    h, w = (int(hw[0]), int(hw[1]))
    video_feature = {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "rgb"],
                     "info": {"video.height": h, "video.width": w, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                              "video.is_depth_map": False, "video.fps": FPS, "video.channels": 3, "has_audio": False}}
    scalar = lambda dt: {"dtype": dt, "shape": [1], "names": None}  # noqa: E731
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": len(episodes),
        "total_frames": index,
        "total_tasks": len(tasks),
        "total_videos": len(episodes) * len(VIDEO_KEYS),
        "total_chunks": (len(episodes) - 1) // CHUNK + 1,
        "chunks_size": CHUNK,
        "fps": FPS,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.wrist_image": video_feature,
            "observation.images.image": video_feature,
            "observation.state": {"dtype": "float32", "shape": [8], "names": {"motors": STATE_NAMES}},
            "action": {"dtype": "float32", "shape": [7], "names": {"motors": ACTION_NAMES}},
            "timestamp": scalar("float32"),
            "frame_index": scalar("int64"),
            "episode_index": scalar("int64"),
            "index": scalar("int64"),
            "task_index": scalar("int64"),
        },
    }
    meta = out / "meta"
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (meta / "tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}) + "\n" for t, i in tasks.items()))
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ep_stats))
    if modality_json:
        shutil.copy(modality_json, meta / "modality.json")
    else:
        (meta / "modality.json").write_text(json.dumps(LIBERO_MODALITY, indent=4))
    return {"episodes": len(episodes), "frames": index, "tasks": len(tasks), "dropped_noops": dropped, "out": str(out)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="gen_demos HDF5 -> LeRobot v2.1 for Isaac-GR00T fine-tuning")
    ap.add_argument("--in", dest="inp", nargs="+", required=True, help="one or more gen_demos .hdf5 files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-noops", action="store_true")
    ap.add_argument("--include-failures", action="store_true")
    ap.add_argument("--modality-json", default=None, help="copy this file instead of the embedded LIBERO one")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    r = convert([Path(p) for p in a.inp], Path(a.out), drop_noops=not a.keep_noops,
                modality_json=Path(a.modality_json) if a.modality_json else None,
                only_success=not a.include_failures, overwrite=a.overwrite)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
