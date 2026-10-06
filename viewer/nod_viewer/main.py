"""Nod 3D viewer: draws the sim's pose stream in the browser with mjviser/viser (design 3.2, 4.2.5).

    python -m nod_viewer.main --mjb recordings/scene_libero_goal_v1.mjb --sub tcp://sim:5563 --port 8080

* Loads the scene compiled for MuJoCo 3.14 (`export_scene.py`; `.mjb` or `.mjb.gz`).
* Subscribes to `pose` messages (3.5.6) and calls `ViserMujocoScene.update_from_arrays` with
  the body positions and rotation matrices. Newest pose wins: the SUB socket keeps a tiny
  receive queue and every wake-up drains it and draws only the last pose, so a slow viewer
  never falls behind (ZMQ_CONFLATE is not used because headers share the socket).
* Applies the fixture placement (`model_patch`) before building the scene: from `--patch`
  (a recording's model_patch.json), and again whenever a `sim.episode_header` with a different
  patch arrives (the sim publishes headers on the pose socket too; `--header-sub` can point at
  the status socket instead). Static bodies are placed once at build time, so a new patch
  rebuilds the scene.
* Draws two viewer-only overlays the policy cameras never see: a translucent handover-zone box
  and a seated-user marker.
* `--replay DIR` loops a recorded run's poses.npz without any sim (demo, smoke test).

Shows the measured view lag (viewer receive time minus the sim's `t_wall`) in the GUI; it is
only meaningful when sim and viewer clocks agree (same host in live mode).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger("nod_viewer")

# Viewer-only overlays (world frame, metres). The handover region matches
# sim/bddl/nod/bring_the_bowl_to_me.bddl (table-frame ranges 0.08..0.16 x 0.06..0.14, table top 0.895).
HANDOVER_CENTER = (0.12, 0.10, 0.90)
HANDOVER_SIZE = (0.08, 0.08, 0.01)
USER_MARKER = (0.85, 0.10, 1.10)  # where a seated user would be, across the table from the robot
INITIAL_CAMERA_POS = (1.15, -1.0, 1.65)
INITIAL_CAMERA_LOOK_AT = (-0.15, 0.0, 0.95)


def load_model(path: str | Path):
    """Load an MJB (optionally gzipped). Falls back to `<path>.gz` when `path` is missing."""
    import mujoco

    p = Path(path)
    if not p.exists() and Path(str(p) + ".gz").exists():
        p = Path(str(p) + ".gz")
    if p.suffix == ".gz":
        data = gzip.decompress(p.read_bytes())
        return mujoco.MjModel.from_binary_path(_spill(data))
    if p.suffix == ".xml":
        return mujoco.MjModel.from_xml_path(str(p))
    return mujoco.MjModel.from_binary_path(str(p))


def _spill(data: bytes) -> str:
    import tempfile

    f = tempfile.NamedTemporaryFile(suffix=".mjb", delete=False)
    f.write(data)
    f.close()
    return f.name


def apply_patch(model, patch: dict | None) -> int:
    """Write body_pos/body_quat by body name into the model. Returns how many were written."""
    import mujoco

    if not patch:
        return 0
    n = 0
    for key in ("body_pos", "body_quat"):
        for name, val in (patch.get(key) or {}).items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            getattr(model, key)[bid] = np.asarray(val, dtype=np.float64)
            n += 1
    return n


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) quaternions [n, 4] -> rotation matrices [n, 3, 3] (MuJoCo convention)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - w * z)
    m[..., 0, 2] = 2 * (x * z + w * y)
    m[..., 1, 0] = 2 * (x * y + w * z)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - w * x)
    m[..., 2, 0] = 2 * (x * z - w * y)
    m[..., 2, 1] = 2 * (y * z + w * x)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


class PoseFeed:
    """Receives `pose` / `sim.episode_header` messages and forwards them to callbacks.

    Pure ZMQ + nod_protocol (no viser), so it can be tested on its own.
    """

    def __init__(self, sub: str | None, on_pose: Callable[[Any], None], on_header: Callable[[Any], None],
                 header_sub: str | None = None, context=None) -> None:
        import zmq

        self.zmq = zmq
        self.ctx = context or zmq.Context.instance()
        self.on_pose, self.on_header = on_pose, on_header
        self.sockets = []
        if sub:
            s = self.ctx.socket(zmq.SUB)
            s.setsockopt(zmq.SUBSCRIBE, b"")
            s.setsockopt(zmq.RCVHWM, 4)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(sub)
            self.sockets.append(s)
        if header_sub:
            h = self.ctx.socket(zmq.SUB)
            h.setsockopt(zmq.SUBSCRIBE, b"")
            h.setsockopt(zmq.LINGER, 0)
            h.connect(header_sub)
            self.sockets.append(h)
        self.stop = threading.Event()
        self.received = 0
        self.thread = threading.Thread(target=self.loop, name="pose-feed", daemon=True)

    def start(self) -> "PoseFeed":
        self.thread.start()
        return self

    def loop(self) -> None:
        from nod_protocol import Pose, SimEpisodeHeader, loads

        poller = self.zmq.Poller()
        for s in self.sockets:
            poller.register(s, self.zmq.POLLIN)
        while not self.stop.is_set():
            events = dict(poller.poll(50))
            newest_pose = None
            for s in self.sockets:
                if s not in events:
                    continue
                while True:  # drain: only the newest pose is drawn
                    try:
                        data = s.recv(self.zmq.NOBLOCK)
                    except self.zmq.Again:
                        break
                    try:
                        msg = loads(data)
                    except Exception as exc:
                        log.debug("ignoring undecodable message: %s", exc)
                        continue
                    if isinstance(msg, Pose):
                        newest_pose = msg
                    elif isinstance(msg, SimEpisodeHeader):
                        self.on_header(msg)
            if newest_pose is not None:
                self.received += 1
                self.on_pose(newest_pose)

    def close(self) -> None:
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(1)
        for s in self.sockets:
            s.close(0)


class Viewer:
    """Owns the viser server and the mjviser scene; thread-safe pose updates."""

    def __init__(self, model, host: str = "0.0.0.0", port: int = 8080, overlays: bool = True,
                 server=None) -> None:
        import viser
        from mjviser import ViserMujocoScene

        self._Scene = ViserMujocoScene
        self.model = model
        self.server = server or viser.ViserServer(host=host, port=port, verbose=False)
        try:  # start looking at the table from the user's side, not from the world origin
            self.server.initial_camera.position = INITIAL_CAMERA_POS
            self.server.initial_camera.look_at = INITIAL_CAMERA_LOOK_AT
        except AttributeError:  # older viser without initial_camera
            pass
        self.overlays = overlays
        self.lock = threading.Lock()
        self.patch: dict | None = None
        self.lags: deque[float] = deque(maxlen=200)
        self.poses = 0
        self.last_step = -1
        self._build()
        self.status = self.server.gui.add_markdown("waiting for poses")
        self._last_gui = 0.0

    def _build(self) -> None:
        self.scene = self._Scene(self.server, self.model, num_envs=1)
        # mjviser tracks the first moving body by default, i.e. it shifts the whole scene so that
        # body sits at the origin. Our overlays and camera are in world coordinates: turn it off.
        self.scene.camera_tracking_enabled = False
        try:
            self.server.scene.set_up_direction("+z")
        except AttributeError:
            pass
        if self.overlays:
            self.server.scene.add_box("/nod/handover_zone", dimensions=HANDOVER_SIZE, position=HANDOVER_CENTER,
                                      color=(40, 170, 90), opacity=0.35)
            self.server.scene.add_icosphere("/nod/user_head", radius=0.09, position=USER_MARKER,
                                            color=(230, 180, 140), opacity=0.6)
            self.server.scene.add_box("/nod/user_body", dimensions=(0.2, 0.36, 0.4),
                                      position=(USER_MARKER[0] + 0.03, USER_MARKER[1], USER_MARKER[2] - 0.3),
                                      color=(90, 110, 160), opacity=0.4)
            self.server.scene.add_label("/nod/user_label", "you", position=(USER_MARKER[0], USER_MARKER[1], USER_MARKER[2] + 0.14))

    def set_patch(self, patch: dict | None) -> bool:
        """Apply a new fixture placement; rebuilds the scene if it changed."""
        if not patch or patch == self.patch:
            return False
        with self.lock:
            apply_patch(self.model, patch)
            self.patch = patch
            self.server.scene.reset()
            self._build()
        log.info("applied model_patch (%d bodies) and rebuilt the scene", len(patch.get("body_pos", {})))
        return True

    def on_header(self, header) -> None:
        if header.body_names and len(header.body_names) != self.model.nbody:
            log.warning("episode header has %d bodies, scene has %d: wrong .mjb?", len(header.body_names), self.model.nbody)
            return
        self.set_patch(header.model_patch.model_dump())

    def on_pose(self, pose) -> None:
        xpos = np.asarray(pose.xpos, dtype=np.float64)
        if xpos.shape[0] == 0:  # "no pose known yet" (the gateway's MockRunner before any replay)
            return
        if xpos.shape[0] != self.model.nbody:
            log.warning("pose has %d bodies, scene has %d", xpos.shape[0], self.model.nbody)
            return
        xmat = quat_to_mat(pose.xquat)
        with self.lock:
            self.scene.update_from_arrays(xpos[None], xmat[None])
        self.poses += 1
        self.last_step = pose.step
        self.lags.append((time.time() - pose.t_wall) * 1000.0)
        now = time.monotonic()
        if now - self._last_gui > 0.5:
            self._last_gui = now
            lag = np.percentile(self.lags, 50) if self.lags else float("nan")
            self.status.content = f"step {pose.step} | epoch {pose.epoch} | view lag p50 {lag:.0f} ms"

    def view_lag_ms(self) -> dict[str, float] | None:
        if not self.lags:
            return None
        a = np.asarray(self.lags)
        return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95))}


def replay_poses(run_dir: Path, hz: float = 20.0, loop: bool = True):
    """Yield `Pose` messages from a recorded run (poses.npz) at `hz`."""
    from nod_protocol import Pose

    with np.load(Path(run_dir) / "poses.npz") as z:
        xpos, xquat, qpos = z["xpos"], z["xquat"], z["qpos"]
    while True:
        for i in range(len(xpos)):
            yield Pose(epoch=0, step=i, t_sim=i / hz, t_wall=time.time(), xpos=xpos[i], xquat=xquat[i], qpos=qpos[i])
            time.sleep(1.0 / hz)
        if not loop:
            return


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Nod 3D viewer (mjviser)")
    ap.add_argument("--mjb", required=True, help="scene .mjb (or .mjb.gz) compiled for this MuJoCo")
    ap.add_argument("--sub", default="tcp://127.0.0.1:5563", help="pose PUB endpoint of the sim or MockRunner")
    ap.add_argument("--header-sub", default=None, help="optional status PUB endpoint carrying sim.episode_header")
    ap.add_argument("--patch", default=None, help="model_patch.json to apply before the first header arrives")
    ap.add_argument("--replay", default=None, help="loop a recorded run directory instead of subscribing")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-overlays", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    model = load_model(a.mjb)
    patch = None
    if a.patch:
        patch = json.loads(Path(a.patch).read_text())
    elif a.replay and (Path(a.replay) / "model_patch.json").exists():
        patch = json.loads((Path(a.replay) / "model_patch.json").read_text())
    if patch:
        apply_patch(model, patch)
    v = Viewer(model, a.host, a.port, overlays=not a.no_overlays)
    v.patch = patch
    log.info("viewer on http://%s:%d (%d bodies)", a.host, a.port, model.nbody)
    if a.replay:
        for pose in replay_poses(Path(a.replay)):
            v.on_pose(pose)
        return
    feed = PoseFeed(a.sub, v.on_pose, v.on_header, a.header_sub).start()
    try:
        while True:
            time.sleep(5)
            log.info("poses drawn: %d, view lag: %s", v.poses, v.view_lag_ms())
    except KeyboardInterrupt:
        pass
    finally:
        feed.close()


if __name__ == "__main__":
    main()
