"""Viewer tests: model loading, patching, the ZMQ pose feed and the viser scene (CPU, no sim)."""

from __future__ import annotations

import gzip
import json
import socket
import time
import uuid
from pathlib import Path

import mujoco
import numpy as np
import pytest
import zmq
from nod_protocol import ModelPatch, Pose, SimEpisodeHeader, dumps

from nod_viewer.main import PoseFeed, Viewer, apply_patch, load_model, quat_to_mat, replay_poses

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "recordings" / "scene_libero_goal_v1.mjb"

TINY = """
<mujoco>
  <worldbody>
    <body name="table" pos="0 0 0.4"><geom type="box" size="0.4 0.4 0.02" rgba="0.6 0.5 0.4 1"/></body>
    <body name="bowl" pos="0 0 0.5"><freejoint/><geom type="cylinder" size="0.05 0.02" rgba="0.1 0.1 0.1 1"/></body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def tiny(tmp_path):
    m = mujoco.MjModel.from_xml_string(TINY)
    p = tmp_path / "tiny.mjb"
    mujoco.mj_saveModel(m, str(p), None)
    return p


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def pose_for(model, step: int, z: float = 0.6) -> Pose:
    d = mujoco.MjData(model)
    d.qpos[2] = z
    mujoco.mj_kinematics(model, d)
    return Pose(epoch=0, step=step, t_sim=step / 20, t_wall=time.time(), xpos=d.xpos.astype(np.float32),
                xquat=d.xquat.astype(np.float32), qpos=d.qpos.astype(np.float32))


def test_quat_to_mat_matches_mujoco():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(20, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    ours = quat_to_mat(q)
    for i in range(len(q)):
        ref = np.zeros(9)
        mujoco.mju_quat2Mat(ref, q[i])
        np.testing.assert_allclose(ours[i], ref.reshape(3, 3), atol=1e-12)


def test_load_model_mjb_gz_and_fallback(tiny, tmp_path):
    gz = tmp_path / "scene.mjb.gz"
    gz.write_bytes(gzip.compress(tiny.read_bytes()))
    assert load_model(tiny).nbody == 3
    assert load_model(gz).nbody == 3
    assert load_model(tmp_path / "scene.mjb").nbody == 3  # missing .mjb falls back to .mjb.gz


def test_apply_patch_by_body_name(tiny):
    m = load_model(tiny)
    n = apply_patch(m, {"body_pos": {"table": [0.1, 0.2, 0.3], "nope": [0, 0, 0]},
                        "body_quat": {"table": [0, 0, 0, 1]}})
    assert n == 2
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")
    np.testing.assert_allclose(m.body_pos[bid], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(m.body_quat[bid], [0, 0, 0, 1])


def test_pose_feed_receives_newest_pose_and_header(tiny, tmp_path):
    model = load_model(tiny)
    ctx = zmq.Context()
    addr = f"ipc://{tmp_path}/{uuid.uuid4().hex[:8]}-pose"
    pub = ctx.socket(zmq.PUB)
    pub.bind(addr)
    poses, headers = [], []
    feed = PoseFeed(addr, poses.append, headers.append, context=ctx).start()
    try:
        time.sleep(0.3)  # slow joiner
        header = SimEpisodeHeader(scene_id="libero_goal_v1", model_hash="h", mj_ver="3.3.1",
                                  body_names=["world", "table", "bowl"], dyn_body_ids=[2],
                                  model_patch=ModelPatch(body_pos={"table": (0.0, 0.0, 0.41)}))
        pub.send(dumps(header))
        for i in range(3):
            pub.send(dumps(pose_for(model, i)))
        end = time.monotonic() + 3
        while (not poses or poses[-1].step != 2 or not headers) and time.monotonic() < end:
            time.sleep(0.02)
        assert headers and headers[0].model_patch.body_pos["table"] == (0.0, 0.0, 0.41)
        assert poses[-1].step == 2
        assert len(poses) <= 3
    finally:
        feed.close()
        pub.close(0)
        ctx.term()


def test_viewer_draws_poses_and_rebuilds_on_new_patch(tiny):
    model = load_model(tiny)
    v = Viewer(model, host="127.0.0.1", port=free_port())
    try:
        v.on_pose(pose_for(model, 5, z=0.7))
        assert v.poses == 1 and v.last_step == 5 and v.view_lag_ms() is not None
        bad = Pose(epoch=0, step=6, t_sim=0, t_wall=time.time(), xpos=np.zeros((2, 3), np.float32),
                   xquat=np.tile(np.float32([1, 0, 0, 0]), (2, 1)), qpos=np.zeros(7, np.float32))
        v.on_pose(bad)  # wrong body count is ignored, not drawn
        assert v.poses == 1
        empty = Pose(epoch=0, step=6, t_sim=0, t_wall=time.time(), xpos=np.zeros((0, 3), np.float32),
                     xquat=np.zeros((0, 4), np.float32), qpos=np.zeros(0, np.float32))
        v.on_pose(empty)  # the MockRunner's "no pose yet": ignored quietly
        assert v.poses == 1
        header = SimEpisodeHeader(scene_id="libero_goal_v1", model_hash="h", mj_ver="3.3.1",
                                  body_names=["world", "table", "bowl"], dyn_body_ids=[2],
                                  model_patch=ModelPatch(body_pos={"table": (0.05, 0.0, 0.4)}))
        v.on_header(header)
        assert v.patch["body_pos"]["table"] == (0.05, 0.0, 0.4)
        assert not v.set_patch(v.patch)  # unchanged patch: no rebuild
        v.on_header(header.model_copy(update={"body_names": ["world"]}))  # wrong scene: ignored
        assert v.patch["body_pos"]["table"] == (0.05, 0.0, 0.4)
        v.on_pose(pose_for(model, 7))
        assert v.poses == 2
    finally:
        v.server.stop()


@pytest.mark.skipif(not (SCENE.exists() or Path(str(SCENE) + ".gz").exists()), reason="no exported scene")
def test_exported_scene_replays_a_recording():
    model = load_model(SCENE)
    report = json.loads(SCENE.with_suffix(".json").read_text())
    assert model.nbody == report["nbody"] and report["max_xpos_err_m"] < 1e-3
    runs = sorted(p for p in SCENE.parent.iterdir() if (p / "poses.npz").exists())
    if not runs:
        pytest.skip("no recorded runs")
    patch = json.loads((runs[0] / "model_patch.json").read_text())
    assert apply_patch(model, patch) > 0
    v = Viewer(model, host="127.0.0.1", port=free_port())
    try:
        gen = replay_poses(runs[0], hz=1000, loop=False)
        for _ in range(5):
            v.on_pose(next(gen))
        assert v.poses == 5
    finally:
        v.server.stop()
