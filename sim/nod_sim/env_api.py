"""The two seams of the sim runner: an environment adapter and a policy client.

`SimRunner` never touches LIBERO or GR00T directly. It drives any object that
satisfies `SimEnv` and asks any object that satisfies `Policy` for action
chunks. Production uses `nod_sim.scene.LiberoSession` and
`nod_sim.policy.Gr00tPolicy`; the unit tests use `nod_sim.fakes.FakeEnv` and
`FakePolicy`, which run on CPU without LIBERO.

Action convention (policy units, same as Isaac-GR00T's LiberoEnv input):
    a = [x, y, z, roll, pitch, yaw, gripper]
    x..yaw  in [-1, 1]: OSC_POSE delta; 1.0 on xyz is 0.05 m in one 20 Hz step
            (robosuite osc_pose.json output_max 0.05, design 1.3)
    gripper in [0, 1]: 1.0 = open, 0.0 = closed. The adapter converts this to
            robosuite's convention (rescale to [-1, 1], binarise, invert), exactly
            as LiberoEnv.step does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
ACTION_DIM = 7
OSC_MAX_DELTA = 0.05  # metres per step at |action| = 1 (osc_pose.json output_max)
CONTROL_HZ = 20  # LIBERO control_freq
GRIP_OPEN = 1.0
GRIP_CLOSED = 0.0
LANG_KEY = "annotation.human.action.task_description"  # GR00T libero_sim language key


def hold_action(gripper: float) -> np.ndarray:
    """Zero translation and rotation, gripper command held (the controlled-stop action, 5.5)."""
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    a[6] = gripper
    return a


def to_robosuite(action: np.ndarray) -> np.ndarray:
    """Policy units -> the 7-D vector robosuite's OSC_POSE env takes.

    Mirrors Isaac-GR00T LiberoEnv.step: gripper [0,1] -> [-1,1], sign(), then invert
    (robosuite: -1 open, +1 closed).
    """
    v = np.asarray(action, dtype=np.float64).copy()
    v[-1] = np.sign(2.0 * v[-1] - 1.0)
    v[-1] = -v[-1]
    return v


@dataclass
class EpisodeInfo:
    """What the viewer needs to rebuild the scene (`sim.episode_header`)."""

    scene_id: str
    model_hash: str
    mj_ver: str
    body_names: list[str]
    dyn_body_ids: list[int]
    model_patch: dict[str, dict[str, list[float]]] = field(default_factory=lambda: {"body_pos": {}, "body_quat": {}})


@runtime_checkable
class SimEnv(Protocol):
    """One continuous simulation session (design 1.3, 4.2.4).

    Implementations: `nod_sim.scene.LiberoSession` (real) and `nod_sim.fakes.FakeEnv`.
    """

    scene_id: str
    step_count: int  # env.step calls since the last reset

    def reset(self, init_index: int = 0) -> None:
        """Restore fixed initial state `init_index` and run the settle steps."""

    def step(self, action: np.ndarray) -> bool:
        """Apply one 7-D policy-unit action (one 50 ms control step). Returns the active goal's success."""

    def observation(self) -> dict[str, Any]:
        """Current observation in GR00T sim format (flat `video.*`, `state.*`, language key)."""

    def set_skill(self, instruction: str, goal_ref: str) -> None:
        """Goal swap: set the instruction the policy sees and the goal `check_success` evaluates."""

    def check_success(self) -> bool: ...

    @property
    def t_sim(self) -> float: ...

    def eef_pos(self) -> np.ndarray: ...

    def eef_quat(self) -> np.ndarray:
        """(w, x, y, z)."""

    def gripper_is_closed(self) -> bool: ...

    def object_pos(self, name: str) -> np.ndarray | None:
        """World position of a movable object or fixture body, None if unknown."""

    def eval_atom(self, atom: str) -> bool | None:
        """Evaluate one registry atom ("on(a,b)") with the simulator's predicate code; None if not evaluable."""

    def render(self, camera: str, w: int, h: int) -> np.ndarray:
        """uint8 [h, w, 3], upright."""

    def body_poses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(xpos [nb,3], xquat [nb,4] wxyz, qpos [nq]) as float32."""

    def episode_info(self) -> EpisodeInfo: ...

    def get_state(self) -> np.ndarray:
        """Flattened physics state (time, qpos, qvel) for recording and re-rendering."""

    def set_state(self, state: np.ndarray) -> None: ...


class PolicyError(RuntimeError):
    """The policy could not produce a chunk (server error, bad reply, or timeout)."""


class PolicyTimeout(PolicyError):
    """The policy did not answer within its timeout."""


@runtime_checkable
class Policy(Protocol):
    """Anything that maps an observation to an action chunk.

    `get_action` returns float32 [T, 7] in policy units. It may block; the runner
    calls it from a worker thread and waits interruptibly (5.5).
    """

    def get_action(self, obs: dict[str, Any]) -> np.ndarray: ...

    def ping(self) -> bool: ...
