"""Gateway <-> sim messages over ZMQ msgpack (3.5.4, 3.5.5) and the pose stream (3.5.6)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .arrays import F32Array
from .base import B64Bytes, Fault, NodModel, ScriptName, ZmqMessage
from .world import WorldState

Vec3 = tuple[float, float, float]

# Placeholder workspace box from 3.5.4; fill from demo trajectory extremes on Day 1.
DEFAULT_EEF_BOX: tuple[Vec3, Vec3] = ((-0.40, -0.40, 0.85), (0.20, 0.40, 1.35))


# ---------------------------------------------------------------- gateway -> sim (3.5.4)


class Limits(NodModel):
    """Execution-time limits enforced in the sim loop (5.3)."""

    eef_box: tuple[Vec3, Vec3] = DEFAULT_EEF_BOX  # (min xyz, max xyz), sim world frame, metres
    max_abs_action: float = 1.0


class LoadScene(ZmqMessage):
    type: Literal["ctl.load_scene"] = "ctl.load_scene"
    req_id: str
    scene_id: str
    init_index: int = 0
    epoch: int


class RunSkill(ZmqMessage):
    """Run one learned policy skill (`ctl.run_skill`)."""

    type: Literal["ctl.run_skill"] = "ctl.run_skill"
    req_id: str
    run_id: str
    epoch: int
    skill_id: str
    kind: Literal["policy"] = "policy"
    instruction: str  # exact LIBERO instruction string
    goal_ref: str  # "libero_goal/<task_name>" or "nod/<task_name>"
    n_action_steps: int = 8
    max_steps: int = 400
    limits: Limits = Field(default_factory=Limits)
    # Addition (docs/DEVIATIONS.md): R14 speed scale, 0.6 for near_user skills.
    action_scale: float = Field(default=1.0, gt=0.0, le=1.0)


class RunScripted(ZmqMessage):
    """Run one scripted waypoint motion (`ctl.run_scripted`)."""

    type: Literal["ctl.run_scripted"] = "ctl.run_scripted"
    req_id: str
    run_id: str
    epoch: int
    script: ScriptName
    max_steps: int = 160
    limits: Limits = Field(default_factory=Limits)  # addition: same box applies to scripts


class Pause(ZmqMessage):
    type: Literal["ctl.pause"] = "ctl.pause"
    req_id: str
    epoch: int


class Resume(ZmqMessage):
    type: Literal["ctl.resume"] = "ctl.resume"
    req_id: str
    epoch: int


class Abort(ZmqMessage):
    type: Literal["ctl.abort"] = "ctl.abort"
    req_id: str
    epoch: int


class Capture(ZmqMessage):
    type: Literal["ctl.capture"] = "ctl.capture"
    req_id: str
    cameras: list[str]
    w: int = 640
    h: int = 480


class WorldRequest(ZmqMessage):
    type: Literal["ctl.world"] = "ctl.world"
    req_id: str


class CtlHeartbeat(ZmqMessage):
    """Relayed every 100 ms only while the phone heartbeat is fresh (5.6)."""

    type: Literal["ctl.hb"] = "ctl.hb"
    t: int
    phone_ok: bool


class CtlEStop(ZmqMessage):
    """Sent on the dedicated PUSH->PULL e-stop socket (port 5561)."""

    type: Literal["ctl.estop"] = "ctl.estop"
    estop_seq: int
    t_gw: int


# ---------------------------------------------------------------- sim -> gateway (3.5.5)


class SimAck(ZmqMessage):
    type: Literal["sim.ack"] = "sim.ack"
    req_id: str
    ok: bool
    error: str | None = None


SimPhase = Literal["idle", "executing", "paused", "done", "failed", "aborted"]


class SimStatus(ZmqMessage):
    """Run progress from the sim loop.

    phase: executing (stepping), paused (ctl.pause or hb_lost), done (finished; for a
    policy skill `success` is the goal predicate), failed (budget or fault), aborted
    (ctl.abort or e-stop), idle (no run in flight).
    """

    type: Literal["sim.status"] = "sim.status"
    run_id: str | None = None
    epoch: int
    phase: SimPhase
    step: int
    success: bool = False
    fault: Fault | None = None
    policy_ms: float | None = None
    step_ms: float | None = None


class SimEStopAck(ZmqMessage):
    type: Literal["sim.estop_ack"] = "sim.estop_ack"
    estop_seq: int
    epoch: int
    stopped_at_step: int | None = None
    t_rx: int  # sim wall clock ms when ctl.estop was received
    t_halt: int  # sim wall clock ms when physics stepping halted


class SimWorld(WorldState):
    """`sim.world`: a WorldState on the wire. Use `.to_world()` for the bare model."""

    v: int = 1
    type: Literal["sim.world"] = "sim.world"

    def to_world(self) -> WorldState:
        return WorldState.model_validate(self.model_dump(exclude={"v", "type"}))

    @classmethod
    def from_world(cls, world: WorldState) -> "SimWorld":
        return cls.model_validate(world.model_dump())


class Frame(NodModel):
    """One captured image. `jpeg` is msgpack bin on ZMQ, base64 in JSON."""

    camera: str
    w: int
    h: int
    jpeg: B64Bytes
    step: int | None = None


class SimFrames(ZmqMessage):
    type: Literal["sim.frames"] = "sim.frames"
    req_id: str
    step: int
    frames: list[Frame]


class ModelPatch(NodModel):
    """Fixture placement LIBERO writes into model.body_pos/body_quat at reset, keyed by body name."""

    body_pos: dict[str, Vec3] = Field(default_factory=dict)
    body_quat: dict[str, tuple[float, float, float, float]] = Field(default_factory=dict)


class SimEpisodeHeader(ZmqMessage):
    type: Literal["sim.episode_header"] = "sim.episode_header"
    scene_id: str
    model_hash: str
    mj_ver: str
    body_names: list[str]
    dyn_body_ids: list[int]
    model_patch: ModelPatch = Field(default_factory=ModelPatch)


# ---------------------------------------------------------------- sim -> viewer (3.5.6)


class Pose(ZmqMessage):
    """Body poses for the viewer, published on port 5563.

    Arrays are float32; see `nod_protocol.arrays` for the wire format.
    xpos is [nb, 3], xquat is [nb, 4] (w, x, y, z), qpos is [nq].
    """

    type: Literal["pose"] = "pose"
    epoch: int
    step: int
    t_sim: float
    t_wall: float  # sim wall clock, seconds since the Unix epoch
    xpos: F32Array
    xquat: F32Array
    qpos: F32Array

    @model_validator(mode="after")
    def _shapes(self) -> "Pose":
        if self.xpos.ndim != 2 or self.xpos.shape[1] != 3:
            raise ValueError(f"xpos must be [nb, 3], got {list(self.xpos.shape)}")
        if self.xquat.ndim != 2 or self.xquat.shape[1] != 4:
            raise ValueError(f"xquat must be [nb, 4], got {list(self.xquat.shape)}")
        if self.xpos.shape[0] != self.xquat.shape[0]:
            raise ValueError("xpos and xquat must have the same number of bodies")
        if self.qpos.ndim != 1:
            raise ValueError(f"qpos must be [nq], got {list(self.qpos.shape)}")
        return self

    def __eq__(self, other: object) -> bool:  # ndarray fields need array_equal
        if not isinstance(other, Pose):
            return NotImplemented
        import numpy as np

        scalars = ("v", "type", "epoch", "step", "t_sim", "t_wall")
        return all(getattr(self, f) == getattr(other, f) for f in scalars) and all(
            np.array_equal(getattr(self, f), getattr(other, f)) for f in ("xpos", "xquat", "qpos")
        )

    __hash__ = None  # type: ignore[assignment]
