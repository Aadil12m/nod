"""gen_demos.py (on FakeEnv) and to_lerobot.py: HDF5 layout, LeRobot v2.1 output, video round trip."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

import gen_demos
import to_lerobot
from nod_sim.fakes import FakeEnv

# Optional cross-checks against an Isaac-GR00T checkout (no torch needed): ISAAC_GR00T_DIR=~/Isaac-GR00T
GR00T_DIR = Path(os.environ.get("ISAAC_GR00T_DIR", "/nonexistent"))
GR00T_MODALITY = GR00T_DIR / "examples" / "LIBERO" / "modality.json"


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    out = tmp_path_factory.mktemp("demos") / "bowl_to_user.hdf5"
    s = gen_demos.generate(FakeEnv(), out, skill_id="bowl_to_user", n=3, keep_success=True, start="mix",
                           noise=0.02, seed=1, log=lambda *_: None)
    return out, s


def test_gen_demos_hdf5_layout(demos):
    out, s = demos
    assert s["attempts"] == 3 and s["successes"] == 3 and s["kept"] == 3
    assert json.loads(out.with_suffix(".json").read_text())["kept"] == 3
    with h5py.File(out) as f:
        g = f["data"]
        assert json.loads(g.attrs["problem_info"])["language_instruction"] == "bring the bowl to me"
        d = g["demo_0"]
        t = d.attrs["num_samples"]
        assert d["actions"].shape == (t, 7) and d["actions_policy"].shape == (t, 7)
        assert d["obs/agentview_rgb"].shape[0] == t and d["obs/agentview_rgb"].dtype == np.uint8
        assert d["obs/ee_pos"].shape == (t, 3) and d["obs/gripper_states"].shape == (t, 2)
        assert d["dones"][-1] == 1 and d["rewards"][:-1].sum() == 0
        pol, rs = np.asarray(d["actions_policy"]), np.asarray(d["actions"])
        # gripper conventions: policy 1 = open <-> robosuite -1 = open
        assert set(np.unique(rs[:, 6])) <= {-1.0, 1.0}
        np.testing.assert_array_equal(rs[:, 6] == -1.0, pol[:, 6] > 0.5)
        np.testing.assert_allclose(rs[:, :6], pol[:, :6])
        assert {g[k].attrs["start"] for k in g} <= {"table", "plate", "plate_cheese"}


def test_keep_mask_drops_noops():
    a = np.zeros((6, 7), np.float32)
    a[:, 6] = 1.0
    a[1, 0] = 0.3          # motion
    a[4, 6] = 0.0          # gripper change: kept even without motion
    keep = to_lerobot.keep_mask(a)
    assert keep.tolist() == [True, True, False, False, True, True]  # 5: gripper back to open


def test_to_lerobot_layout_and_video_round_trip(demos, tmp_path):
    import imageio.v2 as imageio
    import pyarrow.parquet as pq

    h5, _ = demos
    out = tmp_path / "lerobot"
    r = to_lerobot.convert([h5], out)
    assert r["episodes"] == 3 and r["tasks"] == 1
    info = json.loads((out / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1" and info["fps"] == 20 and info["total_frames"] == r["frames"]
    assert info["features"]["observation.state"]["shape"] == [8] and info["features"]["action"]["shape"] == [7]
    modality = json.loads((out / "meta/modality.json").read_text())
    assert modality == to_lerobot.LIBERO_MODALITY
    eps = [json.loads(line) for line in (out / "meta/episodes.jsonl").read_text().splitlines()]
    tasks = [json.loads(line) for line in (out / "meta/tasks.jsonl").read_text().splitlines()]
    assert tasks == [{"task_index": 0, "task": "bring the bowl to me"}]
    stats = [json.loads(line) for line in (out / "meta/episodes_stats.jsonl").read_text().splitlines()]
    assert stats[0]["stats"]["observation.images.image"]["count"] == [min(100, eps[0]["length"])]
    t = pq.read_table(out / "data/chunk-000/episode_000000.parquet")
    assert t.column_names == ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index",
                              "task_index"]
    assert t.num_rows == eps[0]["length"]
    state = np.asarray(t.column("observation.state").to_pylist(), np.float32)
    action = np.asarray(t.column("action").to_pylist(), np.float32)
    assert state.shape == (t.num_rows, 8) and action.shape == (t.num_rows, 7)
    assert t.column("index").to_pylist()[0] == 0
    t1 = pq.read_table(out / "data/chunk-000/episode_000001.parquet")
    assert t1.column("index").to_pylist()[0] == t.num_rows
    # the video decodes back to the HDF5 frames rotated 180 degrees (H.264, so approximately)
    with h5py.File(h5) as f:
        d = f["data/demo_0"]
        mask = to_lerobot.keep_mask(np.asarray(d["actions_policy"]))
        raw = np.asarray(d["obs/agentview_rgb"])[mask]
        np.testing.assert_allclose(np.asarray(d["actions_policy"])[mask], action, atol=1e-6)
    frames = imageio.mimread(out / "videos/chunk-000/observation.images.image/episode_000000.mp4", memtest=False)
    assert len(frames) == t.num_rows
    err = np.abs(np.asarray(frames[0], np.float64) - raw[0][::-1, ::-1]).mean()
    assert err < 8.0, err


@pytest.mark.skipif(not GR00T_MODALITY.exists(), reason="Isaac-GR00T checkout not present")
def test_embedded_modality_matches_isaac_gr00t():
    assert json.loads(GR00T_MODALITY.read_text()) == to_lerobot.LIBERO_MODALITY


@pytest.mark.skipif(not (GR00T_DIR / "gr00t" / "data" / "dataset" / "lerobot_episode_loader.py").exists(),
                    reason="set ISAAC_GR00T_DIR to an Isaac-GR00T checkout")
def test_isaac_gr00t_loader_reads_the_dataset(demos, tmp_path, monkeypatch):
    """Isaac-GR00T's own stats generation and LeRobotEpisodeLoader on our output.

    Video decoding is swapped for imageio because GR00T's decoder (torchcodec) needs torch.
    Needs pandas, scipy and tqdm in the test venv.
    """
    pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    import imageio.v2 as imageio

    monkeypatch.syspath_prepend(str(GR00T_DIR))
    import gr00t.data.dataset.lerobot_episode_loader as loader_mod
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import generate_rel_stats, generate_stats

    def frames_by_indices(path, indices, decoder_kwargs=None):
        return np.asarray(imageio.mimread(path, memtest=False))[np.asarray(indices)]

    monkeypatch.setattr(loader_mod, "get_frames_by_indices", frames_by_indices)
    out = tmp_path / "lerobot"
    to_lerobot.convert([demos[0]], out)
    generate_stats(out)
    generate_rel_stats(out, EmbodimentTag("libero_sim"))
    loader = loader_mod.LeRobotEpisodeLoader(out, MODALITY_CONFIGS["libero_sim"])
    assert len(loader) == 3
    df = loader[0]
    assert df["language.annotation.human.action.task_description"].iloc[0] == "bring the bowl to me"
    assert df["state.gripper"].iloc[0].shape == (2,) and df["action.gripper"].iloc[0].shape == (1,)
    assert df["video.image"].iloc[0].shape[-1] == 3
