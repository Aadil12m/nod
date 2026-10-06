"""MockRunner: a CPU-only stand-in for `nod_sim.runner` that speaks the SAME ZMQ protocol
(design 3.3, 3.5.4-3.5.6, 4.2.2, 4.4). Needs only numpy and pyzmq.

Behaviour:
  * ROUTER on ctl, PULL on estop, PUB on status, PUB on poses, all bound.
  * A fixed-rate loop (`hz`). Before every step it drains the e-stop socket, and the pacing
    wait polls the e-stop socket, so `ctl.estop` freezes the run within one tick.
  * Runs replay a recorded episode from `library` when one matches the skill
    (`<library>/*/manifest.json` with `skill_id` and `outcome`/`success`, plus `poses.npz`
    with `xpos [T,nb,3]`, `xquat [T,nb,4]`, `qpos [T,nq]`). Otherwise a scripted timeline runs:
    the skill "succeeds" after `steps` ticks (scripted motions after half that), or fails with
    `fault: budget` at `max_steps` with probability `failure_rate`. Success applies the
    skill's registry effects to the oracle atoms.
  * Honours ctl.pause / ctl.resume / ctl.abort and the epoch; sim-side heartbeat watchdog:
    no `ctl.hb` for `hb_timeout_ms` while executing -> controlled stop, `paused`, `hb_lost`.
  * Publishes `sim.status` (10 Hz while a run is active, and on every phase change),
    `sim.world` (2 Hz and on change), `sim.estop_ack`, and `pose` every tick. Between runs the
    last replayed pose is held, so the viewer keeps showing where the arm stopped; before any
    recording was replayed the arrays are empty (the viewer ignores those).
  * Publishes `sim.episode_header` on the status AND the pose socket at start, on
    `ctl.load_scene` and every 2 s (as `nod_sim.runner` does), so a viewer that joins late still
    gets the fixture placement. With a library the header carries the recordings' `model_patch`,
    `body_names` and `model_hash` (one library comes from one session, so one patch fits all).
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zmq
from nod_protocol import (
    CTL,
    Abort,
    Capture,
    CtlEStop,
    CtlHeartbeat,
    Eef,
    Frame,
    LoadScene,
    ObjPose,
    Pause,
    Pose,
    Resume,
    RunScripted,
    RunSkill,
    SimAck,
    SimEpisodeHeader,
    SimEStopAck,
    SimFrames,
    SimStatus,
    SimWorld,
    WorldRequest,
    apply_effects,
    dumps,
    load_skills,
    loads,
    now_ms,
    registry_atoms,
    registry_objects,
)

log = logging.getLogger(__name__)

# A valid 8x8 grey JPEG (generated with Pillow), used for ctl.capture when nothing better is recorded.
PLACEHOLDER_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300100b0c0e0c0a100e0d0e1211101318281a181616183123"
    "251d283a333d3c3933383740485c4e404457453738506d51575f626768673e4d71797064785c656763ffdb0043011112"
    "121815182f1a1a2f63423842636363636363636363636363636363636363636363636363636363636363636363636363"
    "6363636363636363636363636363ffc00011080008000803012200021101031101ffc4001f0000010501010101010100"
    "000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d010203000411051221"
    "31410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a"
    "434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a9293949596979899"
    "9aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1"
    "f2f3f4f5f6f7f8f9faffc4001f0100030101010101010101010000000000000102030405060708090a0bffc400b51100"
    "020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c109233352f015"
    "6272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a63646566676869"
    "6a737475767778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4"
    "c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda000c03010002110311003f0028"
    "a28a00ffd9"
)

# Rough object positions in the LIBERO-Goal scene (metres, world frame); display only.
_OBJ_POS = {
    "akita_black_bowl_1": (-0.09, 0.0, 0.90),
    "plate_1": (0.05, 0.10, 0.90),
    "cream_cheese_1": (-0.05, -0.20, 0.90),
    "wine_bottle_1": (-0.25, 0.05, 0.95),
    "wine_rack_1": (-0.30, -0.25, 0.95),
    "wooden_cabinet_1": (-0.30, 0.30, 0.95),
    "flat_stove_1": (-0.10, 0.25, 0.90),
}
HOME_EEF = (-0.21, 0.02, 1.12)
HEADER_PERIOD_S = 2.0  # sim.episode_header repeat, as in nod_sim.runner


@dataclass
class Recording:
    skill_id: str
    success: bool
    steps: int
    path: Path
    poses: dict[str, np.ndarray] | None = None

    def load(self) -> dict[str, np.ndarray]:
        if self.poses is None:
            with np.load(self.path / "poses.npz") as z:
                self.poses = {k: np.asarray(z[k], dtype=np.float32) for k in ("xpos", "xquat", "qpos")}
        return self.poses


def scan_library(library: Path | None) -> list[Recording]:
    """Find recorded runs: `<library>/**/manifest.json` next to a `poses.npz`."""
    if library is None or not Path(library).exists():
        return []
    out = []
    for man in sorted(Path(library).glob("**/manifest.json")):
        if not (man.parent / "poses.npz").exists():
            continue
        try:
            m = json.loads(man.read_text())
            ok = m.get("success")
            if ok is None:
                ok = m.get("outcome") in ("success", "done")
            out.append(
                Recording(skill_id=m["skill_id"], success=bool(ok), steps=int(m.get("steps", 0)), path=man.parent)
            )
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            log.warning("skipping recording %s: %s", man, exc)
    return out


def library_header(library: Path | None, scene_id: str) -> SimEpisodeHeader:
    """The episode header for a mock library: model_patch and body names of its recordings."""
    if library is not None and Path(library).exists():
        for man in sorted(Path(library).glob("**/manifest.json")):
            patch = man.parent / "model_patch.json"
            if not patch.exists():
                continue
            try:
                m = json.loads(man.read_text())
                return SimEpisodeHeader(
                    scene_id=m.get("scene_id", scene_id),
                    model_hash=m.get("model_hash", "mock"),
                    mj_ver=m.get("mj_ver", "mock"),
                    body_names=list(m.get("body_names", [])),
                    dyn_body_ids=list(m.get("dyn_body_ids", [])),
                    model_patch=json.loads(patch.read_text()),
                )
            except (ValueError, json.JSONDecodeError) as exc:
                log.warning("no episode header from %s: %s", man.parent, exc)
    return SimEpisodeHeader(scene_id=scene_id, model_hash="mock", mj_ver="mock", body_names=[], dyn_body_ids=[])


@dataclass
class _Run:
    run_id: str
    skill_id: str
    kind: str  # "policy" | "scripted"
    effects: list[str]
    max_steps: int
    finish_at: int  # step at which the run ends
    succeed: bool
    phase: str = "executing"
    step: int = 0
    fault: str | None = None
    recording: Recording | None = None
    started: float = field(default_factory=time.monotonic)


class MockRunner:
    def __init__(
        self,
        library: Path | None,
        ctl: str,
        estop: str,
        pub: str,
        pose_pub: str,
        hz: int = 20,
        *,
        registry: Path | None = None,
        scene_id: str = "libero_goal_v1",
        steps: int = 60,
        failure_rate: float = 0.0,
        seed: int = 0,
        hb_timeout_ms: int | None = 500,
    ) -> None:
        self.addrs = {"ctl": ctl, "estop": estop, "pub": pub, "pose": pose_pub}
        self.hz = hz
        self.scene_id = scene_id
        self.steps = steps
        self.failure_rate = failure_rate
        self.rng = random.Random(seed)
        self.hb_timeout_ms = hb_timeout_ms
        self.library = scan_library(library)
        self.header = library_header(library, scene_id)
        self._last_pose: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._header_at = 0.0
        specs = load_skills(registry) if registry else []
        self.specs = {s.skill_id: s for s in specs}
        self.script_effects = {s.scripted.script: s.effects for s in specs if s.scripted is not None}
        self.atom_names = registry_atoms(specs)
        self.object_names = registry_objects(specs) or sorted(_OBJ_POS)
        self.atoms: dict[str, bool] = {a: False for a in self.atom_names}
        self.epoch = 0
        self.sim_step = 0
        self.run: _Run | None = None
        self.last_hb = 0.0
        self.eef = list(HOME_EEF)
        self.gripper = "open"
        self.estops: list[int] = []  # handled estop_seq values (for tests)
        self.forced: list[bool] = []  # outcomes to use for the next runs (tests)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._world_dirty = True

    # ------------------------------------------------------------ lifecycle

    def start(self) -> MockRunner:
        self._thread = threading.Thread(target=self.serve_forever, name="mock-runner", daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise RuntimeError("MockRunner did not start")
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)

    def serve_forever(self) -> None:
        ctx = zmq.Context()
        self._ctl = ctx.socket(zmq.ROUTER)
        self._estop = ctx.socket(zmq.PULL)
        self._pub = ctx.socket(zmq.PUB)
        self._pose = ctx.socket(zmq.PUB)
        for s, key in ((self._ctl, "ctl"), (self._estop, "estop"), (self._pub, "pub"), (self._pose, "pose")):
            s.setsockopt(zmq.LINGER, 0)
            s.bind(self.addrs[key])
        poller = zmq.Poller()
        poller.register(self._estop, zmq.POLLIN)
        poller.register(self._ctl, zmq.POLLIN)
        self._ready.set()
        period = 1.0 / self.hz
        tick = 0
        try:
            next_t = time.monotonic()
            while not self._stop.is_set():
                self._drain_estop()  # e-stop first, before anything steps
                self._drain_ctl()
                self._step()
                if tick % 2 == 0 and self.run is not None:
                    self._publish_status()
                if tick % 10 == 0 or self._world_dirty:
                    self._publish_world()
                self._publish_pose()
                if time.monotonic() - self._header_at >= HEADER_PERIOD_S:
                    self._publish_header()
                tick += 1
                next_t += period
                # Pacing wait that wakes on e-stop or ctl traffic (5.5: the wait never delays a stop).
                while not self._stop.is_set():
                    remaining = next_t - time.monotonic()
                    if remaining <= 0:
                        break
                    events = dict(poller.poll(timeout=max(1, int(remaining * 1000))))
                    if self._estop in events:
                        self._drain_estop()
                    if self._ctl in events:
                        self._drain_ctl()
                if time.monotonic() - next_t > 1.0:  # fell far behind (debugger, overload): resync
                    next_t = time.monotonic()
        finally:
            for s in (self._ctl, self._estop, self._pub, self._pose):
                s.close(0)
            ctx.term()

    # ------------------------------------------------------------ publishing

    def _send_pub(self, msg) -> None:
        self._pub.send(dumps(msg))

    def world(self) -> SimWorld:
        objects = {n: ObjPose(pos=_OBJ_POS.get(n, (0.0, 0.0, 0.9))) for n in self.object_names}
        return SimWorld(
            scene_id=self.scene_id,
            step=self.sim_step,
            t_sim=round(self.sim_step / 20.0, 3),
            atoms=dict(self.atoms),
            objects=objects,
            eef=Eef(pos=tuple(self.eef), gripper=self.gripper),
        )  # type: ignore[arg-type]

    def _publish_world(self) -> None:
        self._send_pub(self.world())
        self._world_dirty = False

    def _status(self, r: _Run) -> SimStatus:
        return SimStatus(
            run_id=r.run_id,
            epoch=self.epoch,
            phase=r.phase,
            step=r.step,  # type: ignore[arg-type]
            success=r.phase == "done" and r.succeed,
            fault=r.fault,  # type: ignore[arg-type]
            policy_ms=None,
            step_ms=round(1000.0 / self.hz, 1),
        )

    def _publish_status(self) -> None:
        if self.run is not None:
            self._send_pub(self._status(self.run))

    def _publish_header(self) -> None:
        self._header_at = time.monotonic()
        data = dumps(self.header)
        self._pub.send(data)
        self._pose.send(data)

    def _publish_pose(self) -> None:
        r = self.run
        if r is not None and r.recording is not None:
            p = r.recording.load()
            i = min(r.step, len(p["xpos"]) - 1)
            self._last_pose = (p["xpos"][i], p["xquat"][i], p["qpos"][i])
        if self._last_pose is not None:
            xpos, xquat, qpos = self._last_pose
        else:
            xpos, xquat, qpos = np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        msg = Pose(
            epoch=self.epoch,
            step=self.sim_step,
            t_sim=round(self.sim_step / 20.0, 3),
            t_wall=time.time(),
            xpos=xpos,
            xquat=xquat,
            qpos=qpos,
        )
        self._pose.send(dumps(msg))

    # ------------------------------------------------------------ the run

    def _end(self, phase: str, fault: str | None = None) -> None:
        r = self.run
        assert r is not None
        r.phase, r.fault = phase, fault
        if phase == "done" and r.succeed:
            self.atoms = apply_effects(self.atoms, r.effects)
            self._world_dirty = True
        if r.skill_id in ("return_home", "handover_retreat"):
            self.eef, self.gripper = list(HOME_EEF), "open"
        self._publish_status()
        self.run = None

    def _step(self) -> None:
        r = self.run
        if r is None or r.phase != "executing":
            return
        if self.hb_timeout_ms is not None and (time.monotonic() - self.last_hb) * 1000 > self.hb_timeout_ms:
            # Sim-side watchdog (5.6): controlled stop, resumable.
            r.phase, r.fault = "paused", "hb_lost"
            self._publish_status()
            return
        r.step += 1
        self.sim_step += 1
        self.eef[2] = HOME_EEF[2] - 0.1 * abs(np.sin(r.step / 15.0))
        if r.step >= r.finish_at:
            if r.succeed:
                self._end("done")
            else:
                self._end("failed", "budget")

    # ------------------------------------------------------------ e-stop

    def _drain_estop(self) -> None:
        while True:
            try:
                data = self._estop.recv(zmq.NOBLOCK)
            except zmq.Again:
                return
            t_rx = now_ms()
            try:
                msg = loads(data, CTL)
            except Exception as exc:
                log.warning("mock runner: bad estop message: %s", exc)
                continue
            if isinstance(msg, CtlEStop):
                self._handle_estop(msg, t_rx)

    def _handle_estop(self, msg: CtlEStop, t_rx: int) -> None:
        stopped_at = None
        if msg.estop_seq in self.estops and self.run is None:
            # A repeat (the gateway resends until acked) with nothing running: ack again, no
            # second epoch bump. While a run is in flight a repeated seq stops it (nod_sim.runner).
            self._send_pub(
                SimEStopAck(estop_seq=msg.estop_seq, epoch=self.epoch, stopped_at_step=None, t_rx=t_rx, t_halt=now_ms())
            )
            return
        if self.run is not None:
            # Controlled stop (Category 1 analogue): 4 zero-motion steps holding the gripper, then halt.
            stopped_at = self.run.step
            self.sim_step += 4
            self.run.phase, self.run.fault = "aborted", None
            self._publish_status()
            self.run = None
        self.epoch += 1
        self.estops.append(msg.estop_seq)
        self._send_pub(
            SimEStopAck(
                estop_seq=msg.estop_seq, epoch=self.epoch, stopped_at_step=stopped_at, t_rx=t_rx, t_halt=now_ms()
            )
        )

    # ------------------------------------------------------------ ctl

    def _reply(self, ident: bytes, msg) -> None:
        self._ctl.send_multipart([ident, dumps(msg)])

    def _drain_ctl(self) -> None:
        while True:
            try:
                ident, data = self._ctl.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return
            except ValueError:
                continue
            try:
                msg = loads(data, CTL)
            except Exception as exc:
                log.warning("mock runner: bad ctl message: %s", exc)
                continue
            self._handle_ctl(ident, msg)

    def _ack(self, ident: bytes, req_id: str, ok: bool, error: str | None = None) -> None:
        self._reply(ident, SimAck(req_id=req_id, ok=ok, error=error))

    def _stale(self, epoch: int) -> bool:
        return epoch < self.epoch

    def _handle_ctl(self, ident: bytes, msg) -> None:
        if isinstance(msg, CtlHeartbeat):
            self.last_hb = time.monotonic()
        elif isinstance(msg, LoadScene):
            if msg.scene_id != self.scene_id:
                self._ack(ident, msg.req_id, False, f"unknown scene {msg.scene_id}")
                return
            self.epoch = max(self.epoch, msg.epoch)
            self.estops.clear()  # a (re)connecting gateway starts its e-stop counter again
            self.run = None
            self.atoms = {a: False for a in self.atom_names}
            self.eef, self.gripper = list(HOME_EEF), "open"
            self._world_dirty = True
            self._ack(ident, msg.req_id, True)
            # An idle status carries the epoch, so a restarted gateway can adopt it (as nod_sim.runner).
            self._send_pub(SimStatus(run_id=None, epoch=self.epoch, phase="idle", step=self.sim_step))
            self._publish_header()
        elif isinstance(msg, (RunSkill, RunScripted)):
            self._start_run(ident, msg)
        elif isinstance(msg, Pause):
            if self._stale(msg.epoch) or self.run is None:
                self._ack(ident, msg.req_id, False, "stale epoch" if self._stale(msg.epoch) else "no run")
                return
            if self.run.phase == "executing":
                self.run.phase = "paused"
                self._publish_status()
            self._ack(ident, msg.req_id, True)
        elif isinstance(msg, Resume):
            if self._stale(msg.epoch) or self.run is None:
                self._ack(ident, msg.req_id, False, "stale epoch" if self._stale(msg.epoch) else "no run")
                return
            if self.run.phase == "paused":
                self.run.phase, self.run.fault = "executing", None
                self.last_hb = time.monotonic()
                self._publish_status()
            self._ack(ident, msg.req_id, True)
        elif isinstance(msg, Abort):
            if self._stale(msg.epoch):
                self._ack(ident, msg.req_id, False, "stale epoch")
                return
            if self.run is not None:
                self._end("aborted")
            self._ack(ident, msg.req_id, True)
        elif isinstance(msg, Capture):
            frames = [Frame(camera=c, w=8, h=8, jpeg=PLACEHOLDER_JPEG, step=self.sim_step) for c in msg.cameras]
            self._reply(ident, SimFrames(req_id=msg.req_id, step=self.sim_step, frames=frames))
        elif isinstance(msg, WorldRequest):
            self._reply(ident, self.world())

    def _start_run(self, ident: bytes, msg: RunSkill | RunScripted) -> None:
        if self._stale(msg.epoch):
            self._ack(ident, msg.req_id, False, "stale epoch")
            return
        if self.run is not None:
            self._ack(ident, msg.req_id, False, "busy: a run is in flight")
            return
        self.epoch = max(self.epoch, msg.epoch)
        if isinstance(msg, RunSkill):
            spec = self.specs.get(msg.skill_id)
            if self.specs and spec is None:
                self._ack(ident, msg.req_id, False, f"unknown skill {msg.skill_id}")
                return
            effects = list(spec.effects) if spec else []
            succeed = self.forced.pop(0) if self.forced else self.rng.random() >= self.failure_rate
            rec = next((r for r in self.library if r.skill_id == msg.skill_id and r.success == succeed), None)
            finish = (rec.steps or self.steps) if rec else (self.steps if succeed else msg.max_steps)
            run = _Run(
                msg.run_id,
                msg.skill_id,
                "policy",
                effects,
                msg.max_steps,
                min(finish, msg.max_steps),
                succeed,
                recording=rec,
            )
        else:
            effects = list(self.script_effects.get(msg.script, []))
            rec = next((r for r in self.library if r.skill_id == msg.script and r.success), None)
            run = _Run(
                msg.run_id,
                msg.script,
                "scripted",
                effects,
                msg.max_steps,
                min((rec.steps if rec and rec.steps else max(1, self.steps // 2)), msg.max_steps),
                True,
                recording=rec,
            )
        self.run = run
        self.last_hb = time.monotonic() if self.last_hb == 0.0 else self.last_hb
        self._ack(ident, msg.req_id, True)
        self._publish_status()
