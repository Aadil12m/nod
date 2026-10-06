"""The one continuous LIBERO-Goal session (design 1.2, 1.3, 4.2.4 scene.py).

`LiberoSession` adapts a LIBERO `OffScreenRenderEnv` to the `SimEnv` interface the runner
drives. It is built from `sim/bddl/nod/bring_the_bowl_to_me.bddl`: the LIBERO-Goal scene
(regions, fixtures, objects and :init identical to the ten task files) plus one invisible table
region, `main_table_handover_region`, so every registry atom, including the custom skill's
goal, can be evaluated in the same session. `verify_single_scene()` checks that claim.

Goal swapping (1.3): `set_skill` assigns `parsed_problem["goal_state"]` from the target task's
BDDL and sets the instruction the policy sees. The episode is never reset by the loop.

Observations use the Isaac-GR00T `libero_sim` format (the same keys and transforms as
`gr00t/eval/sim/LIBERO/libero_env.py` at Isaac-GR00T 51d4c89): `video.image` and
`video.wrist_image` are the 256 px agentview and eye-in-hand renders rotated 180 degrees,
`state.*` is the end-effector position, axis-angle orientation and finger qpos. We replicate
`LiberoEnv._process_observation` and its action conversion instead of importing it, so the
runner needs neither gr00t nor gymnasium in the LIBERO venv.

LIBERO imports torch in `libero.libero.benchmark/__init__.py` only to load the fixed initial
states. We read the task list from `benchmark/libero_suite_task_map.py` directly and load the
`.pruned_init` files with a torch-free loader (they are zip-format pickles of plain numpy
arrays), falling back to `torch.load(weights_only=False)` when torch is installed.

Importing this module does not import LIBERO; only `make_session_env` / `LiberoSession` do.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import pickle
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .env_api import EpisodeInfo, LANG_KEY, to_robosuite

SCENE_ID = "libero_goal_v1"
SIM_DIR = Path(__file__).resolve().parents[1]
NOD_BDDL_DIR = SIM_DIR / "bddl" / "nod"
SESSION_BDDL = NOD_BDDL_DIR / "bring_the_bowl_to_me.bddl"
DEFAULT_INIT_TASK = "put_the_bowl_on_the_plate"
SETTLE_STEPS = 10  # as LeRobot's LIBERO wrapper and the Day-1 harness do
POLICY_CAMS = (("video.image", "agentview"), ("video.wrist_image", "robot0_eye_in_hand"))
POLICY_CAM_PX = 256

# The ten LIBERO-Goal tasks and the custom task: goal_ref -> (instruction, goal_state).
# Instructions come from the file name (LIBERO grab_language_from_filename), which is what
# the policy was trained on; goals are the BDDL :goal blocks at LIBERO 8f1084e. A LIBERO test
# (tests/test_libero_real.py) checks this table against the real files.
LIBERO_GOAL_GOALS: dict[str, list[list[str]]] = {
    "open_the_middle_drawer_of_the_cabinet": [["open", "wooden_cabinet_1_middle_region"]],
    "open_the_top_drawer_and_put_the_bowl_inside": [["in", "akita_black_bowl_1", "wooden_cabinet_1_top_region"]],
    "push_the_plate_to_the_front_of_the_stove": [["on", "plate_1", "main_table_stove_front_region"]],
    "put_the_bowl_on_the_plate": [["on", "akita_black_bowl_1", "plate_1"]],
    "put_the_bowl_on_the_stove": [["on", "akita_black_bowl_1", "flat_stove_1_cook_region"]],
    "put_the_bowl_on_top_of_the_cabinet": [["on", "akita_black_bowl_1", "wooden_cabinet_1_top_side"]],
    "put_the_cream_cheese_in_the_bowl": [["on", "cream_cheese_1", "akita_black_bowl_1"]],
    "put_the_wine_bottle_on_the_rack": [["on", "wine_bottle_1", "wine_rack_1_top_region"]],
    "put_the_wine_bottle_on_top_of_the_cabinet": [["on", "wine_bottle_1", "wooden_cabinet_1_top_side"]],
    "turn_on_the_stove": [["turnon", "flat_stove_1"]],
}
CUSTOM_GOALS: dict[str, list[list[str]]] = {
    "bring_the_bowl_to_me": [["on", "akita_black_bowl_1", "main_table_handover_region"]],
}


def instruction_from_name(task_name: str) -> str:
    """LIBERO's grab_language_from_filename for lowercase (non LIBERO-100) task files."""
    return " ".join(task_name.removesuffix(".bddl").split("_"))


GOAL_REFS: dict[str, tuple[str, list[list[str]]]] = {
    **{f"libero_goal/{n}": (instruction_from_name(n), g) for n, g in LIBERO_GOAL_GOALS.items()},
    **{f"nod/{n}": (instruction_from_name(n), g) for n, g in CUSTOM_GOALS.items()},
}


# ---------------------------------------------------------------------------- LIBERO files


def libero_path(key: str) -> str:
    from libero.libero import get_libero_path

    return get_libero_path(key)


def goal_task_names() -> list[str]:
    """The LIBERO-Goal task names, read from LIBERO's task map without importing torch."""
    p = Path(libero_path("benchmark_root")) / "benchmark" / "libero_suite_task_map.py"
    spec = importlib.util.spec_from_file_location("_libero_task_map", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return list(mod.libero_task_map["libero_goal"])


def bddl_path(goal_ref: str) -> Path:
    suite, _, name = goal_ref.partition("/")
    if suite == "nod":
        return NOD_BDDL_DIR / f"{name}.bddl"
    if suite == "libero_goal":
        return Path(libero_path("bddl_files")) / "libero_goal" / f"{name}.bddl"
    raise KeyError(f"unknown goal_ref suite in {goal_ref}")


def load_goals() -> dict[str, list]:
    """goal_ref -> parsed goal_state, parsed from the real BDDL files (all 10 plus custom)."""
    from libero.libero.envs.bddl_utils import robosuite_parse_problem

    refs = [f"libero_goal/{n}" for n in goal_task_names()] + [f"nod/{n}" for n in CUSTOM_GOALS]
    return {ref: robosuite_parse_problem(str(bddl_path(ref)))["goal_state"] for ref in refs}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickles the numpy arrays inside a torch zip checkpoint; refuses anything else."""

    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
               ("numpy._core.multiarray", "scalar"), ("collections", "OrderedDict"),
               ("_codecs", "encode")}  # protocol-2 pickles encode raw bytes with _codecs.encode

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from an init-state file")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("init-state file holds torch tensors; install torch to load it")


def load_init_states(task_name: str = DEFAULT_INIT_TASK) -> np.ndarray:
    """The fixed initial states of one LIBERO-Goal task, [N, nstate] float64."""
    path = Path(libero_path("init_states")) / "libero_goal" / f"{task_name}.pruned_init"
    try:
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("data.pkl"))
            return np.asarray(_NumpyOnlyUnpickler(io.BytesIO(z.read(name))).load())
    except (zipfile.BadZipFile, pickle.UnpicklingError, StopIteration):
        import torch  # LIBERO's own path; weights_only=False is needed under torch >= 2.6

        return np.asarray(torch.load(path, weights_only=False))


def strip_blocks(text: str, names=("language", "obj_of_interest", "goal")) -> str:
    """Remove the named top-level `(:name ...)` blocks from BDDL text (parenthesis aware)."""
    out, i = [], 0
    while True:
        m = re.compile(r"\(:(%s)\b" % "|".join(names)).search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        depth, j = 0, m.start()
        while j < len(text):
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            j += 1
        i = j
    return "".join(out)


def scene_fingerprint(bddl_text: str, drop_regions: tuple[str, ...] = ()) -> str:
    """sha256 of a BDDL with :language, :obj_of_interest and :goal removed (design 1.2).

    `drop_regions` removes named region blocks first (used to compare the Nod session BDDL,
    which adds handover_region, against the LIBERO files).
    """
    text = strip_blocks(re.sub(r";.*$", "", bddl_text, flags=re.MULTILINE))  # BDDL ; comments
    for region in drop_regions:
        m = re.search(r"\(\s*%s\s*\n" % re.escape(region), text)
        if m:
            depth, j = 0, m.start()
            while j < len(text):
                depth += text[j] == "("
                depth -= text[j] == ")"
                j += 1
                if depth == 0:
                    break
            text = text[:m.start()] + text[j:]
    norm = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(norm.encode()).hexdigest()


def verify_single_scene() -> dict[str, Any]:
    """Hash the ten LIBERO-Goal BDDLs (and the Nod session BDDL minus its added region).

    Returns {"task_hashes": {name: hash}, "identical": bool, "nod_matches": bool, ...}.
    """
    hashes = {n: scene_fingerprint(bddl_path(f"libero_goal/{n}").read_text()) for n in goal_task_names()}
    nod = scene_fingerprint(SESSION_BDDL.read_text(), drop_regions=("handover_region",))
    uniq = set(hashes.values())
    return {"task_hashes": hashes, "identical": len(uniq) == 1, "nod_hash": nod,
            "nod_matches": len(uniq) == 1 and nod in uniq, "n_tasks": len(hashes)}


# ---------------------------------------------------------------------------- the adapter


def _quat2axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """robosuite transform_utils.quat2axisangle, as copied by Isaac-GR00T LiberoEnv."""
    q = np.array(quat_xyzw, dtype=np.float64)
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(q[3])) / den


# Drawer handle bar centre relative to the drawer body origin (the body slides with the drawer).
# The cabinet has yaw pi, so drawers slide out along world +y. Measured from the collision geoms
# in the real scene (middle: bar at y -0.148, z 1.015, 8.8 cm long along x, 1.6 cm thick).
DRAWER_JOINTS = {"wooden_cabinet_1_top_region": "wooden_cabinet_1_top_level",
                 "wooden_cabinet_1_middle_region": "wooden_cabinet_1_middle_level",
                 "wooden_cabinet_1_bottom_region": "wooden_cabinet_1_bottom_level"}
DRAWER_BODIES = {"wooden_cabinet_1_top_region": "wooden_cabinet_1_cabinet_top",
                 "wooden_cabinet_1_middle_region": "wooden_cabinet_1_cabinet_middle",
                 "wooden_cabinet_1_bottom_region": "wooden_cabinet_1_cabinet_bottom"}
DRAWER_HANDLE_OFFSET = {"wooden_cabinet_1_top_region": (0.003, 0.102, 0.184),
                        "wooden_cabinet_1_middle_region": (0.003, 0.102, 0.110),
                        "wooden_cabinet_1_bottom_region": (0.003, 0.102, 0.040)}


class LiberoSession:
    """`SimEnv` over one LIBERO OffScreenRenderEnv kept alive for the whole session."""

    def __init__(self, scene_id: str = SCENE_ID, init_index: int = 0, *, bddl: str | Path = SESSION_BDDL,
                 init_task: str = DEFAULT_INIT_TASK, lazy_cameras: bool = True, camera_px: int = POLICY_CAM_PX) -> None:
        from libero.libero.envs import OffScreenRenderEnv

        self.scene_id = scene_id
        self.camera_px = camera_px
        self.lazy_cameras = lazy_cameras
        # lazy_cameras: robosuite renders no camera observables per step; the two policy cameras
        # are rendered from the current state only when an observation is requested. The images
        # are the same renders robosuite would have produced for that state.
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=camera_px, camera_widths=camera_px, ignore_done=True,
            use_camera_obs=not lazy_cameras, camera_names=["agentview", "robot0_eye_in_hand"],
        )
        self.goals = load_goals()
        self.init_states = load_init_states(init_task)
        self.init_task = init_task
        self.instruction = ""
        self.goal_ref: str | None = None
        self.step_count = 0
        self._obs: dict[str, Any] = {}
        self._grip_cmd = 1.0
        self._info: EpisodeInfo | None = None
        self.home_eef: np.ndarray | None = None
        self.home_quat: np.ndarray | None = None
        self._rest: dict[str, float] = {}
        self.reset(init_index)

    # -------------------------------------------------------------- handles

    @property
    def problem(self):
        """The LIBERO problem env (Libero_Tabletop_Manipulation): parsed_problem, _eval_predicate."""
        return self._env.env

    @property
    def sim(self):
        return self._env.env.sim

    # -------------------------------------------------------------- session

    def reset(self, init_index: int = 0) -> None:
        """reset, set_init_state(init_states[i]), then 10 settle steps (design 1.3)."""
        self._env.reset()
        state = self.init_states[init_index % len(self.init_states)]
        self._obs = self._env.set_init_state(state)
        self._grip_cmd = 1.0
        hold = np.zeros(7)
        hold[6] = 1.0
        for _ in range(SETTLE_STEPS):
            self._raw_step(hold)
        self.step_count = 0
        self.home_eef = self.eef_pos()
        self.home_quat = self.eef_quat()
        self._info = None
        self._rest = {}
        for name in self.problem.objects_dict:
            p = self.object_pos(name)
            self._rest[name] = float(p[2] - 0.895) if p is not None else 0.0
        if self.goal_ref:
            self.set_skill(self.instruction, self.goal_ref)

    def set_skill(self, instruction: str, goal_ref: str) -> None:
        from .skills import swap_skill

        if goal_ref not in self.goals:
            raise KeyError(f"unknown goal_ref {goal_ref}")
        swap_skill(self, instruction, self.goals[goal_ref])
        self.goal_ref = goal_ref

    def check_success(self) -> bool:
        return bool(self._env.check_success())

    @property
    def t_sim(self) -> float:
        return float(self.sim.data.time)

    # -------------------------------------------------------------- stepping

    def _raw_step(self, action_policy_units: np.ndarray) -> None:
        self._grip_cmd = float(action_policy_units[6])
        obs, _r, _d, _info = self._env.step(to_robosuite(action_policy_units))
        self._obs = obs

    def step(self, action: np.ndarray) -> bool:
        self._raw_step(np.asarray(action, dtype=np.float64).reshape(7))
        self.step_count += 1
        return self.check_success()

    # -------------------------------------------------------------- observation

    def _render_raw(self, camera: str, w: int, h: int) -> np.ndarray:
        return self.sim.render(width=w, height=h, camera_name=camera)  # OpenGL order: row 0 at bottom

    def observation(self) -> dict[str, Any]:
        o = self._obs
        xyz = o["robot0_eef_pos"]
        rpy = _quat2axisangle(o["robot0_eef_quat"])
        obs = {
            "state.x": [float(xyz[0])], "state.y": [float(xyz[1])], "state.z": [float(xyz[2])],
            "state.roll": [float(rpy[0])], "state.pitch": [float(rpy[1])], "state.yaw": [float(rpy[2])],
            "state.gripper": np.asarray(o["robot0_gripper_qpos"], dtype=np.float64),
            LANG_KEY: self.instruction,
        }
        for key, cam in POLICY_CAMS:
            if self.lazy_cameras:
                img = self._render_raw(cam, self.camera_px, self.camera_px)
            else:
                img = o[f"{cam}_image"]
            obs[key] = np.ascontiguousarray(img[::-1, ::-1])  # LiberoEnv: rotate 180 degrees
        return obs

    def render(self, camera: str, w: int, h: int) -> np.ndarray:
        return np.ascontiguousarray(self._render_raw(camera, w, h)[::-1])

    # -------------------------------------------------------------- poses and oracle

    def eef_pos(self) -> np.ndarray:
        return np.asarray(self._obs["robot0_eef_pos"], dtype=np.float64).copy()

    def eef_quat(self) -> np.ndarray:
        x, y, z, w = self._obs["robot0_eef_quat"]
        return np.array([w, x, y, z], dtype=np.float64)

    def gripper_is_closed(self) -> bool:
        return self._grip_cmd < 0.5

    def _body_id(self, name: str) -> int | None:
        pe = self.problem
        if name in pe.obj_body_id:
            return pe.obj_body_id[name]
        return None

    def object_pos(self, name: str) -> np.ndarray | None:
        bid = self._body_id(name)
        if bid is not None:
            return np.asarray(self.sim.data.body_xpos[bid], dtype=np.float64).copy()
        if name in self.problem.object_states_dict:
            try:
                return np.asarray(self.sim.data.get_site_xpos(name), dtype=np.float64).copy()
            except Exception:
                return None
        return None

    def object_quat(self, name: str) -> np.ndarray | None:
        bid = self._body_id(name)
        if bid is None:
            return None
        return np.asarray(self.sim.data.body_xquat[bid], dtype=np.float64).copy()

    def rest_height(self, obj: str) -> float | None:
        return self._rest.get(obj)

    def eval_atom(self, atom: str) -> bool | None:
        from .oracle import libero_eval_atom

        return libero_eval_atom(self.problem, atom)

    def drawer_handle(self, region: str) -> np.ndarray | None:
        body = DRAWER_BODIES.get(region)
        if body is None:
            return None
        bid = self.sim.model.body_name2id(body)
        return np.asarray(self.sim.data.body_xpos[bid], dtype=np.float64) + np.asarray(DRAWER_HANDLE_OFFSET[region])

    def drawer_closed_handle(self, region: str) -> np.ndarray | None:
        h = self.drawer_handle(region)
        if h is None:
            return None
        q = self.sim.data.qpos[self.sim.model.get_joint_qpos_addr(DRAWER_JOINTS[region])]
        return h - self.drawer_axis(region) * (-float(q))  # qpos is negative when open

    def drawer_axis(self, region: str) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0])

    def body_poses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self.sim.data
        return (np.asarray(d.xpos, dtype=np.float32).copy(), np.asarray(d.xquat, dtype=np.float32).copy(),
                np.asarray(d.qpos, dtype=np.float32).copy())

    def get_state(self) -> np.ndarray:
        return np.asarray(self.sim.get_state().flatten(), dtype=np.float64)

    def set_state(self, state: np.ndarray) -> None:
        self.sim.set_state_from_flattened(np.asarray(state, dtype=np.float64))
        self.sim.forward()

    def episode_info(self) -> EpisodeInfo:
        if self._info is None:
            self._info = episode_info(self.sim, self.scene_id)
        return self._info

    def close(self) -> None:
        self._env.close()


def episode_info(sim, scene_id: str) -> EpisodeInfo:
    """Header for the viewer: body names, which bodies move, and the fixture placement patch.

    The patch carries body_pos/body_quat of every body without a joint of its own (fixtures,
    robot base, table): LIBERO writes those into the model at reset and they are not in qpos.
    """
    import mujoco

    m = sim.model
    raw = m._model if hasattr(m, "_model") else m
    names = [mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}" for b in range(raw.nbody)]
    dyn = []
    moving = [False] * raw.nbody
    for b in range(1, raw.nbody):
        moving[b] = bool(raw.body_jntnum[b] > 0) or moving[raw.body_parentid[b]]
        if moving[b]:
            dyn.append(b)
    patch = {"body_pos": {}, "body_quat": {}}
    for b in range(1, raw.nbody):
        if raw.body_jntnum[b] == 0:
            patch["body_pos"][names[b]] = [round(float(x), 6) for x in raw.body_pos[b]]
            patch["body_quat"][names[b]] = [round(float(x), 6) for x in raw.body_quat[b]]
    h = hashlib.sha256()
    for arr in (raw.body_parentid, raw.jnt_type, raw.geom_type, raw.geom_size, raw.body_pos, raw.body_quat):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(",".join(names).encode())
    return EpisodeInfo(scene_id=scene_id, model_hash="sha256:" + h.hexdigest(), mj_ver=mujoco.__version__,
                       body_names=names, dyn_body_ids=dyn, model_patch=patch)


def make_session_env(scene_id: str = SCENE_ID, init_index: int = 0, **kw) -> LiberoSession:
    """Build the session env (design 4.2.4). Requires the LIBERO venv; see sim/README.md."""
    if scene_id != SCENE_ID:
        raise ValueError(f"unknown scene {scene_id}; only {SCENE_ID} exists")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    return LiberoSession(scene_id, init_index, **kw)
