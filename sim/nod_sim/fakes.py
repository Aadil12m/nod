"""CPU stand-ins for LIBERO and the GR00T policy server, for unit tests and demos without a GPU.

`FakeEnv` is a small kinematic model of the LIBERO-Goal table: the end effector moves exactly
0.05 m per unit action per step (an ideal OSC controller), a closing gripper picks up the
nearest object within reach, released objects drop onto whatever is under them, the middle and
top drawers slide when the hand pushes or pulls their handle, and the plate is pushed by a low
hand. Object and region positions are the ones measured in the real scene (init state 0), so
the scripted skills in `nod_sim.scripted` run unchanged on both.

It evaluates the same registry atoms as the real oracle, with simple geometric rules. It is NOT
a physics simulation; it exists so the runner's control logic (e-stop, limits, epochs, goal
swap, watchdog) can be tested deterministically on CPU.

`FakePolicy` returns action chunks from a function of the observation, can block or sleep to
emulate inference latency, and records every observation it was given.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from .env_api import ACTION_DIM, CONTROL_HZ, LANG_KEY, OSC_MAX_DELTA, EpisodeInfo, PolicyError
from .oracle import atom_to_state
from .scene import GOAL_REFS, SCENE_ID

TABLE_Z = 0.895
HOME = np.array([-0.207, 0.002, 1.179])
HOME_QUAT = np.array([1.0, 0.0, -0.028, 0.0])

# Initial layout (body origins, world frame), measured in LIBERO-Goal init state 0.
INIT_OBJECTS = {
    "akita_black_bowl_1": (-0.098, -0.009, 0.898),
    "cream_cheese_1": (-0.046, 0.134, 0.909),
    "wine_bottle_1": (-0.204, -0.062, 0.899),
    "plate_1": (0.062, -0.009, 0.902),
}
FIXTURES = {
    "wooden_cabinet_1": (0.037, -0.248, 0.905),
    "flat_stove_1": (-0.411, 0.209, 0.905),
    "wine_rack_1": (-0.27, -0.256, 0.92),
}
# Region sites: (centre, horizontal half extent).
SITES = {
    "main_table_handover_region": ((0.12, 0.10, TABLE_Z), 0.04),
    "main_table_stove_front_region": ((-0.05, 0.21, TABLE_Z), 0.05),
    "wine_rack_1_top_region": ((-0.27, -0.176, 1.137), 0.05),
    "wooden_cabinet_1_top_side": ((0.034, -0.264, 1.127), 0.08),
    "flat_stove_1_cook_region": ((-0.261, 0.209, 0.905), 0.06),
    "wooden_cabinet_1_top_region": ((0.034, -0.259, 1.091), 0.08),
    "wooden_cabinet_1_middle_region": ((0.034, -0.259, 1.022), 0.08),
    "wooden_cabinet_1_bottom_region": ((0.034, -0.259, 0.952), 0.08),
}
DRAWERS = ("wooden_cabinet_1_top_region", "wooden_cabinet_1_middle_region")
DRAWER_AXIS = np.array([0.0, 1.0, 0.0])
DRAWER_TRAVEL = 0.16
DRAWER_FRONT_Y = 0.105  # closed handle y offset from the region site
# Height of each object's origin above the surface it rests on, and of the surface it offers.
REST = {"akita_black_bowl_1": 0.003, "cream_cheese_1": 0.014, "wine_bottle_1": 0.004, "plate_1": 0.007}
SURFACE = {"plate_1": 0.006, "akita_black_bowl_1": 0.008}
GRASP_XY = 0.08
GRASP_Z = (-0.03, 0.22)


class FakeEnv:
    """Kinematic LIBERO-Goal stand-in implementing `nod_sim.env_api.SimEnv`."""

    def __init__(self, scene_id: str = SCENE_ID, step_delay_s: float = 0.0, image_hw: int = 32) -> None:
        self.scene_id = scene_id
        self.step_delay_s = step_delay_s
        self.image_hw = image_hw
        self.goals: dict[str, list[list[str]]] = {ref: goal for ref, (_instr, goal) in GOAL_REFS.items()}
        self.instruction = ""
        self.goal: list[list[str]] = []
        self.home_eef = HOME.copy()
        self.home_quat = HOME_QUAT.copy()
        self.actions: list[np.ndarray] = []  # every action applied (tests inspect stops)
        self.step_count = 0
        self.resets = 0
        self._lock = threading.Lock()
        self.reset(0)

    # ------------------------------------------------------------------ session

    def reset(self, init_index: int = 0) -> None:
        rng = np.random.default_rng(init_index)
        jitter = (lambda: rng.uniform(-0.005, 0.005, 3) * np.array([1, 1, 0])) if init_index else (lambda: 0.0)
        self.obj = {k: np.array(v, dtype=np.float64) + jitter() for k, v in INIT_OBJECTS.items()}
        self.eef = HOME.copy()
        self.grip_cmd = 1.0
        self.held: str | None = None
        self.held_off = np.zeros(3)
        self._riders: list[str] = []
        self.drawer = {d: 0.0 for d in DRAWERS}
        self.stove_on = False
        self.t = 0.0
        self.step_count = 0
        self.resets += 1

    def set_skill(self, instruction: str, goal_ref: str) -> None:
        if goal_ref not in self.goals:
            raise KeyError(f"unknown goal_ref {goal_ref}")
        self.instruction = instruction
        self.goal = self.goals[goal_ref]

    def check_success(self) -> bool:
        return bool(self.goal) and all(self._eval(s) for s in self.goal)

    @property
    def t_sim(self) -> float:
        return self.t

    # ------------------------------------------------------------------ stepping

    def step(self, action: np.ndarray) -> bool:
        a = np.asarray(action, dtype=np.float64).reshape(ACTION_DIM)
        if not np.all(np.isfinite(a)):
            raise ValueError("FakeEnv got a non-finite action (the runner must have caught it)")
        self.actions.append(a.astype(np.float32))
        if self.step_delay_s:
            time.sleep(self.step_delay_s)
        prev_closed = self.grip_cmd < 0.5
        self.grip_cmd = float(a[6])
        closed = self.grip_cmd < 0.5
        self.eef = self.eef + OSC_MAX_DELTA * np.clip(a[:3], -1, 1)
        self.eef[2] = max(self.eef[2], TABLE_Z + 0.005)  # the table stops the hand
        if closed and not prev_closed:
            self._try_grasp()
        elif not closed and self.held is not None:
            self._release()
        if self.held is not None:
            delta = self.eef + self.held_off - self.obj[self.held]
            self.obj[self.held] = self.obj[self.held] + delta
            for rider in self._riders:  # cheese in a carried bowl comes along
                self.obj[rider] = self.obj[rider] + delta
        self._drawers()
        self._push_plate()
        self.t += 1.0 / CONTROL_HZ
        self.step_count += 1
        return self.check_success()

    def _try_grasp(self) -> None:
        best, best_d = None, GRASP_XY
        for name, p in self.obj.items():
            dz = self.eef[2] - p[2]
            dxy = float(np.linalg.norm(self.eef[:2] - p[:2]))
            if dxy < best_d and GRASP_Z[0] < dz < GRASP_Z[1] and name != "plate_1":
                best, best_d = name, dxy
        if best is not None:
            self.held = best
            self.held_off = self.obj[best] - self.eef
            # Anything resting in the bowl comes along.
            self._riders = [n for n in self.obj if n != best and self._on_obj(n, best)]
        else:
            self._riders = []

    def _release(self) -> None:
        name = self.held
        self.held = None
        p = self.obj[name]
        support_z = TABLE_Z
        for other, q in self.obj.items():
            if other == name or other == "wine_bottle_1":
                continue
            if np.linalg.norm(p[:2] - q[:2]) < 0.06 and q[2] < p[2]:
                support_z = max(support_z, q[2] + SURFACE.get(other, 0.02))
        for site, (c, half) in SITES.items():
            c = np.asarray(c)
            if site in DRAWERS:
                c = c + DRAWER_AXIS * self.drawer[site] * DRAWER_TRAVEL
                if self.drawer[site] < 0.5:
                    continue
            if np.all(np.abs(p[:2] - c[:2]) <= half) and c[2] <= p[2] + 0.02:
                support_z = max(support_z, c[2])
        new_z = support_z + REST.get(name, 0.01)
        for rider in self._riders:
            self.obj[rider][2] += new_z - p[2]
        p[2] = new_z
        self._riders = []

    def _drawers(self) -> None:
        for d in DRAWERS:
            h = self.drawer_handle(d)
            if abs(self.eef[0] - h[0]) < 0.05 and abs(self.eef[2] - h[2]) < 0.035 and abs(self.eef[1] - h[1]) < 0.03:
                closed_y = self.drawer_closed_handle(d)[1]
                self.drawer[d] = float(np.clip((self.eef[1] - closed_y) / DRAWER_TRAVEL, 0.0, 1.0))

    def _push_plate(self) -> None:
        p = self.obj["plate_1"]
        if self.held == "plate_1" or self.eef[2] > p[2] + 0.04:
            return
        d = p[:2] - self.eef[:2]
        n = float(np.linalg.norm(d))
        if 1e-9 < n < 0.10:
            new_xy = self.eef[:2] + d / n * 0.10
            delta = new_xy - p[:2]
            for other in list(self.obj):
                if other != "plate_1" and other != self.held and self._on_obj(other, "plate_1"):
                    self.obj[other][:2] += delta
            p[:2] = new_xy

    # ------------------------------------------------------------------ geometry for scripts

    def drawer_handle(self, region: str) -> np.ndarray | None:
        if region not in DRAWERS:
            return None
        return self.drawer_closed_handle(region) + DRAWER_AXIS * self.drawer[region] * DRAWER_TRAVEL

    def drawer_closed_handle(self, region: str) -> np.ndarray:
        c = np.asarray(SITES[region][0])
        return np.array([c[0], c[1] + DRAWER_FRONT_Y, c[2]])

    def drawer_axis(self, region: str) -> np.ndarray:
        return DRAWER_AXIS.copy()

    def rest_height(self, obj: str) -> float | None:
        return REST.get(obj)

    # ------------------------------------------------------------------ oracle

    def _on_obj(self, a: str, b: str) -> bool:
        pa, pb = self.obj[a], self.obj[b]
        return (np.linalg.norm(pa[:2] - pb[:2]) < 0.06 and pb[2] - 0.005 < pa[2] < pb[2] + 0.08
                and self.held != a)

    def _eval(self, state: list[str]) -> bool | None:
        pred, *args = [s.lower() for s in state]
        known = set(self.obj) | set(FIXTURES) | set(SITES)
        if any(x not in known for x in args):
            return None
        if pred == "open":
            return args[0] in DRAWERS and self.drawer[args[0]] > 0.5
        if pred == "turnon":
            return args[0] == "flat_stove_1" and self.stove_on
        a, b = args
        if a not in self.obj or self.held == a:
            return False
        pa = self.obj[a]
        if b in self.obj:
            return self._on_obj(a, b) if pred == "on" else False
        if b in SITES:
            c, half = SITES[b]
            c = np.asarray(c)
            if b in DRAWERS:
                if self.drawer[b] < 0.5:
                    return False
                c = c + DRAWER_AXIS * self.drawer[b] * DRAWER_TRAVEL
                return bool(np.all(np.abs(pa[:2] - c[:2]) <= half) and abs(pa[2] - c[2]) < 0.06)
            return bool(np.all(np.abs(pa[:2] - c[:2]) <= half) and c[2] - 0.02 <= pa[2] < c[2] + 0.1)
        return False

    def eval_atom(self, atom: str) -> bool | None:
        v = self._eval(atom_to_state(atom))
        return None if v is None else bool(v)

    # ------------------------------------------------------------------ poses and images

    def eef_pos(self) -> np.ndarray:
        return self.eef.copy()

    def eef_quat(self) -> np.ndarray:
        return HOME_QUAT.copy()

    def gripper_is_closed(self) -> bool:
        return self.grip_cmd < 0.5

    def object_pos(self, name: str) -> np.ndarray | None:
        if name in self.obj:
            return self.obj[name].copy()
        if name in FIXTURES:
            return np.array(FIXTURES[name], dtype=np.float64)
        if name in SITES:
            c = np.asarray(SITES[name][0], dtype=np.float64)
            if name in DRAWERS:
                c = c + DRAWER_AXIS * self.drawer[name] * DRAWER_TRAVEL
            return c
        return None

    def object_quat(self, name: str) -> np.ndarray | None:
        return np.array([1.0, 0.0, 0.0, 0.0]) if name in self.obj or name in FIXTURES else None

    BODY_NAMES = ["world", "robot0_right_hand", *INIT_OBJECTS, *FIXTURES]

    def body_poses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xpos = [np.zeros(3), self.eef, *(self.obj[k] for k in INIT_OBJECTS), *(np.asarray(v) for v in FIXTURES.values())]
        xpos = np.asarray(xpos, dtype=np.float32)
        xquat = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (len(xpos), 1))
        qpos = np.concatenate([self.eef, [self.grip_cmd], *(self.obj[k] for k in INIT_OBJECTS),
                               [self.drawer[d] for d in DRAWERS]]).astype(np.float32)
        return xpos, xquat, qpos

    def episode_info(self) -> EpisodeInfo:
        names = list(self.BODY_NAMES)
        return EpisodeInfo(
            scene_id=self.scene_id,
            model_hash="sha256:" + hashlib.sha256(("fake:" + ",".join(names)).encode()).hexdigest(),
            mj_ver="fake",
            body_names=names,
            dyn_body_ids=[1, 2, 3, 4, 5],
            model_patch={"body_pos": {k: list(v) for k, v in FIXTURES.items()}, "body_quat": {}},
        )

    def get_state(self) -> np.ndarray:
        held = -1 if self.held is None else list(INIT_OBJECTS).index(self.held)
        return np.concatenate([[self.t], self.eef, [self.grip_cmd, held], *(self.obj[k] for k in INIT_OBJECTS),
                               [self.drawer[d] for d in DRAWERS], [float(self.stove_on)]])

    def set_state(self, state: np.ndarray) -> None:
        s = np.asarray(state, dtype=np.float64)
        self.t = float(s[0])
        self.eef = s[1:4].copy()
        self.grip_cmd = float(s[4])
        held = int(s[5])
        self.held = None if held < 0 else list(INIT_OBJECTS)[held]
        i = 6
        for k in INIT_OBJECTS:
            self.obj[k] = s[i:i + 3].copy()
            i += 3
        for d in DRAWERS:
            self.drawer[d] = float(s[i])
            i += 1
        self.stove_on = bool(s[i])
        if self.held is not None:
            self.held_off = self.obj[self.held] - self.eef

    def render(self, camera: str, w: int, h: int) -> np.ndarray:
        """A top-down sketch: table, objects as discs, the hand as a cross. Deterministic."""
        img = np.full((h, w, 3), (196, 170, 130), dtype=np.uint8)
        if camera == "robot0_eye_in_hand":
            img[:] = (90, 90, 90)
        def px(p):
            # x in [-0.5, 0.3] -> rows, y in [-0.4, 0.4] -> cols
            r = int((p[0] + 0.5) / 0.8 * (h - 1))
            c = int((p[1] + 0.4) / 0.8 * (w - 1))
            return min(max(r, 0), h - 1), min(max(c, 0), w - 1)
        colors = {"akita_black_bowl_1": (20, 20, 20), "cream_cheese_1": (40, 60, 160),
                  "wine_bottle_1": (90, 20, 30), "plate_1": (240, 240, 240)}
        yy, xx = np.mgrid[0:h, 0:w]
        for name in ("plate_1", "akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1"):
            r, c = px(self.obj[name])
            rad = max(1, int(0.04 / 0.8 * w))
            img[(yy - r) ** 2 + (xx - c) ** 2 <= rad * rad] = colors[name]
        r, c = px(self.eef)
        img[r, :] = (255, 0, 0)
        img[:, c] = (255, 0, 0)
        return img

    def observation(self) -> dict[str, Any]:
        n = self.image_hw
        return {
            "video.image": self.render("agentview", n, n)[::-1, ::-1],
            "video.wrist_image": self.render("robot0_eye_in_hand", n, n)[::-1, ::-1],
            "state.x": [float(self.eef[0])],
            "state.y": [float(self.eef[1])],
            "state.z": [float(self.eef[2])],
            "state.roll": [0.0],
            "state.pitch": [0.0],
            "state.yaw": [0.0],
            "state.gripper": [0.04 * self.grip_cmd, -0.04 * self.grip_cmd],
            LANG_KEY: self.instruction,
        }


# ---------------------------------------------------------------------------- policy


ChunkFn = Callable[[dict[str, Any], int], np.ndarray]


class FakePolicy:
    """Policy stand-in: `fn(obs, call_index) -> [T, 7]` chunk (default: hold still, gripper open).

    `delay_s` emulates inference time. `gate` (a threading.Event), when given, makes every call
    block until it is set, which lets tests hold a chunk "in flight". `error` raises PolicyError.
    """

    def __init__(self, fn: ChunkFn | None = None, horizon: int = 16, delay_s: float = 0.0,
                 gate: threading.Event | None = None, error: Exception | None = None) -> None:
        self.fn = fn
        self.horizon = horizon
        self.delay_s = delay_s
        self.gate = gate
        self.error = error
        self.calls = 0
        self.observations: list[dict[str, Any]] = []
        self.returned: list[np.ndarray] = []

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        i = self.calls
        self.calls += 1
        self.observations.append(obs)
        if self.gate is not None:
            self.gate.wait()
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        if self.fn is None:
            chunk = np.zeros((self.horizon, 7), dtype=np.float32)
            chunk[:, 6] = 1.0
        else:
            chunk = np.asarray(self.fn(obs, i), dtype=np.float32)
        self.returned.append(chunk)
        return chunk

    def ping(self) -> bool:
        return True

    @classmethod
    def constant(cls, action, horizon: int = 16, **kw) -> "FakePolicy":
        a = np.asarray(action, dtype=np.float32).reshape(7)
        return cls(lambda _o, _i: np.tile(a, (horizon, 1)), horizon=horizon, **kw)


class FailingPolicy(FakePolicy):
    """Always raises PolicyError (server down)."""

    def __init__(self, msg: str = "policy server unreachable") -> None:
        super().__init__(error=PolicyError(msg))
