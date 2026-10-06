"""Scripted waypoint skills on privileged object poses (design 1.6 tier T4, 4.2.4 scripted.py).

These generators drive any `SimEnv` (the real `LiberoSession` or the CPU `FakeEnv`) with the
same 7-D policy-unit actions a learned policy would produce, so the runner applies the same
execution-time limits (5.3) and the same interruptible e-stop path (5.5) to them. They are used
for:
  * return_home and the other scripted moves the gateway requests (`ctl.run_scripted`),
  * the T4 fallback actor (`SimRunner(actor="scripted")` or no policy server), and
  * demo generation for the custom skill (`training/gen_demos.py`).

Every generator yields one action per control step and *returns* a bool (StopIteration.value):
True when the motion reached its last waypoint. Whether the skill achieved its goal is decided
by the oracle (`env.check_success()` / registry effects), never by the script.

Controller: proportional servo on the end-effector position, a = clip(gain * err / 0.05, -1, 1),
plus an optional proportional hold on orientation (OSC rotation output_max is 0.5 rad per step).
Rotation is otherwise left alone, as in the Day-1 harness return-home.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from .env_api import GRIP_CLOSED, GRIP_OPEN, OSC_MAX_DELTA

Action = np.ndarray
Script = Iterator[Action]

# Session home pose: the end-effector position after reset and 10 settle steps in
# LIBERO-Goal (measured on robosuite 1.4.0 / MuJoCo 3.3.1, init state 0). Sessions override
# it with their own measured home (`env.home_eef`).
DEFAULT_HOME = np.array([-0.207, 0.002, 1.179])
OSC_MAX_ROT = 0.5  # rad per step at |action| = 1 (osc_pose.json output_max for rotation)


class ScriptError(RuntimeError):
    """No scripted controller exists for this skill, or the scene lacks what it needs."""


def act(xyz=(0.0, 0.0, 0.0), grip: float = GRIP_OPEN, rot=(0.0, 0.0, 0.0)) -> Action:
    a = np.zeros(7, dtype=np.float32)
    a[:3] = np.clip(np.asarray(xyz, dtype=np.float64), -1.0, 1.0)
    a[3:6] = np.clip(np.asarray(rot, dtype=np.float64), -1.0, 1.0)
    a[6] = grip
    return a


def _quat_err_axisangle(q_now_wxyz: np.ndarray, q_ref_wxyz: np.ndarray) -> np.ndarray:
    """World-frame rotation (axis * angle) that takes q_now to q_ref."""
    w1, x1, y1, z1 = q_ref_wxyz
    w2, x2, y2, z2 = q_now_wxyz[0], -q_now_wxyz[1], -q_now_wxyz[2], -q_now_wxyz[3]  # conj(q_now)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    if w < 0:  # shortest path
        w, x, y, z = -w, -x, -y, -z
    v = np.array([x, y, z])
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(s, w)
    return v / s * angle


@dataclass
class Servo:
    """Proportional end-effector servo. Holds `ref_quat` when given."""

    gain: float = 1.0
    tol: float = 0.008
    max_speed: float = 1.0  # cap on |action xyz| (1.0 = 0.05 m per step)
    ref_quat: np.ndarray | None = None
    rot_gain: float = 0.5

    def action(self, env, target: np.ndarray, grip: float) -> tuple[Action, float]:
        err = np.asarray(target, dtype=np.float64) - env.eef_pos()
        cmd = self.gain * err / OSC_MAX_DELTA
        n = np.linalg.norm(cmd)
        if n > self.max_speed:
            cmd = cmd / n * self.max_speed
        rot = (0.0, 0.0, 0.0)
        if self.ref_quat is not None:
            rot = self.rot_gain * _quat_err_axisangle(env.eef_quat(), self.ref_quat) / OSC_MAX_ROT
        return act(cmd, grip, rot), float(np.linalg.norm(err))


def move_to(env, target, grip: float, servo: Servo, max_steps: int = 80, tol: float | None = None) -> Iterator[Action]:
    """Servo to `target` (xyz) with the gripper command held. Returns True when within tol."""
    tol = servo.tol if tol is None else tol
    target = np.asarray(target, dtype=np.float64)
    for _ in range(max_steps):
        a, dist = servo.action(env, target, grip)
        if dist < tol:
            return True
        yield a
    return bool(np.linalg.norm(target - env.eef_pos()) < 2 * tol)


def hold(env, grip: float, steps: int) -> Iterator[Action]:
    for _ in range(steps):
        yield act(grip=grip)
    return True


def _home_of(env, home_xyz=None) -> np.ndarray:
    if home_xyz is not None:
        return np.asarray(home_xyz, dtype=np.float64)
    h = getattr(env, "home_eef", None)
    return np.asarray(h if h is not None else DEFAULT_HOME, dtype=np.float64)


def _ref_quat(env) -> np.ndarray | None:
    q = getattr(env, "home_quat", None)
    return None if q is None else np.asarray(q, dtype=np.float64)


# ---------------------------------------------------------------------------- motions


def return_home(env, obs=None, home_xyz=None, max_steps: int = 80) -> Script:
    """Open the gripper, lift to home height, move to the home xyz (design 1.4 mitigation 1).

    Same motion as `return_home` in docs/runbooks/chain_test.py, with orientation held toward
    the session's home orientation when the env reports one. `obs` is accepted for signature
    compatibility with the design (4.2.4) and unused: poses come from the env.
    """
    home = _home_of(env, home_xyz)
    servo = Servo(gain=1.0, tol=0.01, ref_quat=_ref_quat(env))
    yield from hold(env, GRIP_OPEN, 5)
    p = env.eef_pos()
    lift = np.array([p[0], p[1], max(home[2], p[2])])
    ok1 = yield from move_to(env, lift, GRIP_OPEN, servo, max_steps)
    ok2 = yield from move_to(env, home, GRIP_OPEN, servo, max_steps)
    return bool(ok2 or ok1 and np.linalg.norm(env.eef_pos() - home) < 0.03)


def handover_retreat(env, max_steps: int = 60) -> Script:
    """After a handover: release, rise 12 cm straight up, then return home (away from the user)."""
    servo = Servo(gain=1.0, tol=0.01, ref_quat=_ref_quat(env))
    yield from hold(env, GRIP_OPEN, 8)
    p = env.eef_pos()
    up = min(p[2] + 0.12, max(p[2], _home_of(env)[2]))  # never above home height
    yield from move_to(env, np.array([p[0], p[1], up]), GRIP_OPEN, servo, max_steps)
    return (yield from return_home(env, max_steps=max_steps))


@dataclass(frozen=True)
class GraspSpec:
    """Where to grasp an object, relative to its body origin (world-frame offsets, metres)."""

    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    approach: float = 0.10  # height above the grasp point to approach from
    lift: float = 0.12  # clearance above the destination when carrying
    place_drop: float = 0.02  # release this far above the computed place height
    close_steps: int = 12
    open_steps: int = 10


# Grasp points measured in the real LIBERO-Goal scene (see sim/README.md, "Scripted skills").
# The Panda's fingers close along the world y axis in the home orientation, so the bowl is
# taken by its rim on the +y side (the fingers straddle the rim wall there).
GRASPS: dict[str, GraspSpec] = {
    "akita_black_bowl_1": GraspSpec(offset=(0.0, 0.058, 0.045)),
    "cream_cheese_1": GraspSpec(offset=(0.0, 0.0, 0.0)),
    "wine_bottle_1": GraspSpec(offset=(0.0, 0.0, 0.12), approach=0.08, lift=0.05),
    "plate_1": GraspSpec(offset=(0.0, 0.0, 0.0)),
}


def pick_place(env, obj: str, target_xyz: np.ndarray, grasp: GraspSpec | None = None,
               max_steps: int = 60) -> Script:
    """Pick `obj` at its privileged pose and put it so its body origin lands at `target_xyz`.

    Waypoints: above grasp -> grasp -> close -> lift -> above target -> lower -> open -> retreat.
    """
    pos = env.object_pos(obj)
    if pos is None:
        raise ScriptError(f"object {obj} is not in this scene")
    g = grasp or GRASPS.get(obj, GraspSpec())
    off = np.asarray(g.offset, dtype=np.float64)
    servo = Servo(gain=1.0, tol=0.008, ref_quat=_ref_quat(env))
    fine = Servo(gain=0.8, tol=0.005, max_speed=0.5, ref_quat=_ref_quat(env))
    grasp_pt = np.asarray(pos, dtype=np.float64) + off
    target = np.asarray(target_xyz, dtype=np.float64)
    place_pt = target + off + np.array([0.0, 0.0, g.place_drop])
    carry_z = max(grasp_pt[2], place_pt[2]) + g.lift

    yield from hold(env, GRIP_OPEN, 3)
    yield from move_to(env, np.array([*env.eef_pos()[:2], max(env.eef_pos()[2], carry_z)]), GRIP_OPEN, servo, max_steps)
    if not (yield from move_to(env, grasp_pt + [0, 0, g.approach], GRIP_OPEN, servo, max_steps)):
        return False
    if not (yield from move_to(env, grasp_pt, GRIP_OPEN, fine, max_steps)):
        return False
    yield from hold(env, GRIP_CLOSED, g.close_steps)
    yield from move_to(env, np.array([*grasp_pt[:2], carry_z]), GRIP_CLOSED, servo, max_steps)
    if not (yield from move_to(env, np.array([*place_pt[:2], carry_z]), GRIP_CLOSED, servo, max_steps * 2)):
        return False
    yield from move_to(env, place_pt, GRIP_CLOSED, fine, max_steps)
    yield from hold(env, GRIP_OPEN, g.open_steps)
    # A short retreat: 6 cm keeps a bottle placed on the cabinet top inside the default eef box.
    yield from move_to(env, env.eef_pos() + np.array([0.0, 0.0, 0.06]), GRIP_OPEN, servo, max_steps)
    return True


def push(env, start_xyz, end_xyz, grip: float = GRIP_CLOSED, approach: float = 0.08, max_steps: int = 80) -> Script:
    """Lower the closed gripper at `start_xyz` and slide it to `end_xyz` (drawers, plate)."""
    servo = Servo(gain=1.0, tol=0.008, ref_quat=_ref_quat(env))
    slow = Servo(gain=0.8, tol=0.01, max_speed=0.4, ref_quat=_ref_quat(env))
    start = np.asarray(start_xyz, dtype=np.float64)
    end = np.asarray(end_xyz, dtype=np.float64)
    yield from hold(env, grip, 6)
    yield from move_to(env, np.array([*env.eef_pos()[:2], max(env.eef_pos()[2], start[2] + approach)]), grip, servo, max_steps)
    if not (yield from move_to(env, start + [0, 0, approach], grip, servo, max_steps)):
        return False
    if not (yield from move_to(env, start, grip, slow, max_steps)):
        return False
    ok = yield from move_to(env, end, grip, slow, max_steps * 2, tol=0.015)
    yield from move_to(env, env.eef_pos() + np.array([0.0, 0.0, approach]), grip, servo, max_steps)
    return ok


# ---------------------------------------------------------------------------- skill table


def _site(env, name: str) -> np.ndarray:
    p = env.object_pos(name)
    if p is None:
        raise ScriptError(f"{name} is not in this scene")
    return np.asarray(p, dtype=np.float64)


def _rest_offset(env, obj: str) -> float:
    """Height of an object's body origin above the surface it rests on, at the session start."""
    rest = getattr(env, "rest_height", None)
    if callable(rest):
        h = rest(obj)
        if h is not None:
            return float(h)
    return {"akita_black_bowl_1": 0.003, "cream_cheese_1": 0.014, "wine_bottle_1": 0.004, "plate_1": 0.007}.get(obj, 0.01)


def bowl_grasp(env) -> GraspSpec:
    """Rim grasp on the side of the bowl away from anything inside it (the cheese).

    The inner finger enters the bowl; with cream cheese in it, a finger on the cheese's side
    lands on the cheese and the hand stalls (seen in LIBERO from the chain-B state).
    """
    g = GRASPS["akita_black_bowl_1"]
    bowl = env.object_pos("akita_black_bowl_1")
    cheese = env.object_pos("cream_cheese_1")
    side = 1.0
    if bowl is not None and cheese is not None and np.linalg.norm(cheese[:2] - bowl[:2]) < 0.06:
        side = -1.0 if cheese[1] > bowl[1] else 1.0
    x, y, z = g.offset
    return GraspSpec(offset=(x, side * abs(y), z), approach=g.approach, lift=g.lift, place_drop=g.place_drop,
                     close_steps=g.close_steps, open_steps=g.open_steps)


def _bowl_on_plate(env) -> Script:
    plate = _site(env, "plate_1")
    return (yield from pick_place(env, "akita_black_bowl_1", plate + [0.0, 0.0, 0.012], bowl_grasp(env)))


def _cheese_in_bowl(env) -> Script:
    bowl = _site(env, "akita_black_bowl_1")
    # Release with the finger tips above the bowl rim: the opening fingers are wider than the
    # bowl's inner radius and would lift the bowl by its rim on the way out (seen in LIBERO).
    # Drop it 12 mm toward -y so the +y rim stays free for a later rim grasp (bowl_to_user).
    return (yield from pick_place(env, "cream_cheese_1", bowl + [0.0, -0.012, 0.055]))


def _bowl_to(region: str, dz: float = 0.0) -> Callable:
    def script(env) -> Script:
        dst = _site(env, region)
        return (yield from pick_place(env, "akita_black_bowl_1",
                                      dst + [0.0, 0.0, _rest_offset(env, "akita_black_bowl_1") + dz], bowl_grasp(env)))
    return script


def _bottle_to(region: str, dz: float) -> Callable:
    def script(env) -> Script:
        dst = _site(env, region)
        return (yield from pick_place(env, "wine_bottle_1", dst + [0.0, 0.0, dz]))
    return script


def _drawer_handle(env, region: str) -> np.ndarray:
    """Handle position of a drawer region. Sessions that know the geometry expose `drawer_handle`."""
    fn = getattr(env, "drawer_handle", None)
    if callable(fn):
        h = fn(region)
        if h is not None:
            return np.asarray(h, dtype=np.float64)
    raise ScriptError(f"no drawer handle geometry for {region}")


def _drawer_axis(env, region: str) -> np.ndarray:
    fn = getattr(env, "drawer_axis", None)
    if callable(fn):
        return np.asarray(fn(region), dtype=np.float64)
    return np.array([0.0, 1.0, 0.0])


def _open_middle_drawer(env) -> Script:
    """Hook the handle from above and pull the drawer out along its axis."""
    region = "wooden_cabinet_1_middle_region"
    h = _drawer_handle(env, region)
    ax = _drawer_axis(env, region)
    servo = Servo(gain=1.0, tol=0.008, ref_quat=_ref_quat(env))
    fine = Servo(gain=0.8, tol=0.006, max_speed=0.4, ref_quat=_ref_quat(env))
    # Stand the hand 2.7 cm out from the bar so the inner (open) finger comes down in front of
    # the drawer face instead of on top of it; closing pinches the bar between the fingers.
    grasp = h + ax * 0.027 + [0.0, 0.0, 0.007]
    yield from hold(env, GRIP_OPEN, 3)
    yield from move_to(env, grasp + [0, 0, 0.08], GRIP_OPEN, servo, 80)
    if not (yield from move_to(env, grasp, GRIP_OPEN, fine, 60)):
        return False
    yield from hold(env, GRIP_CLOSED, 12)
    ok = yield from move_to(env, grasp + ax * 0.15, GRIP_CLOSED, fine, 120, tol=0.015)
    yield from hold(env, GRIP_OPEN, 8)
    yield from move_to(env, env.eef_pos() + ax * 0.04 + [0, 0, 0.08], GRIP_OPEN, servo, 60)
    return ok


def close_middle_drawer(env) -> Script:
    """Undo of open_middle_drawer: push the drawer front back in with the closed gripper."""
    region = "wooden_cabinet_1_middle_region"
    h = _drawer_handle(env, region)
    ax = _drawer_axis(env, region)
    closed = getattr(env, "drawer_closed_handle", None)
    end = np.asarray(closed(region), dtype=np.float64) if callable(closed) else h - ax * 0.16
    start = h + ax * 0.03
    return (yield from push(env, start, end - ax * 0.01, grip=GRIP_CLOSED, approach=0.08))


def _plate_to_stove_front(env) -> Script:
    plate = _site(env, "plate_1")
    dst = _site(env, "main_table_stove_front_region")
    d = dst[:2] - plate[:2]
    n = np.linalg.norm(d)
    if n < 1e-6:
        return True
    u = np.array([*(d / n), 0.0])
    start = np.array([*plate[:2], plate[2] + 0.02]) - u * 0.11  # behind the plate rim
    end = np.array([*dst[:2], plate[2] + 0.02]) - u * 0.08
    return (yield from push(env, start, end))


SCRIPTED_SKILLS: dict[str, Callable] = {
    "bowl_on_plate": _bowl_on_plate,
    "cheese_in_bowl": _cheese_in_bowl,
    "bowl_on_cabinet": _bowl_to("wooden_cabinet_1_top_side", 0.005),
    "bowl_on_stove": _bowl_to("flat_stove_1_cook_region", 0.01),
    "bowl_to_user": _bowl_to("main_table_handover_region", 0.005),
    "bottle_on_cabinet": _bottle_to("wooden_cabinet_1_top_side", 0.01),
    "bottle_on_rack": _bottle_to("wine_rack_1_top_region", 0.0),
    "open_middle_drawer": _open_middle_drawer,
    "plate_to_stove_front": _plate_to_stove_front,
}
# stove_on and bowl_in_top_drawer have no scripted stand-in (knob turning and the
# open-then-place sequence were not attempted); they raise ScriptError.

# Which stand-ins reached their goal in the real LIBERO-Goal scene (robosuite 1.4.0, MuJoCo
# 3.3.1, CPU/OSMesa); see sim/README.md for the numbers. The others work only in FakeEnv: in
# LIBERO the drawer handle cannot be reached from above (the top drawer's handle is directly
# over it) and the plate push stalls before the plate. return_home is verified from the table
# and plate poses but NOT from the handover pose (x > 0.1 m): from there the OSC servo drifts
# up and rotates instead of retracting, so handover_retreat is unverified. The runner refuses
# unverified stand-ins unless it was built with allow_unverified_scripts=True.
VERIFIED_IN_LIBERO = {"bowl_on_plate", "cheese_in_bowl", "bowl_to_user", "bowl_on_stove", "bowl_on_cabinet",
                      "bottle_on_cabinet", "bottle_on_rack", "return_home"}

SCRIPTS: dict[str, Callable] = {
    "return_home": return_home,
    "handover_retreat": handover_retreat,
    "close_middle_drawer": close_middle_drawer,  # not verified in LIBERO (see VERIFIED_IN_LIBERO)
}


def scripted_skill(skill_id: str, allow_unverified: bool = True) -> Callable:
    """The scripted stand-in for a policy skill (tier T4). Raises ScriptError if there is none."""
    try:
        fn = SCRIPTED_SKILLS[skill_id]
    except KeyError:
        raise ScriptError(f"no scripted controller for skill {skill_id}") from None
    if not allow_unverified and skill_id not in VERIFIED_IN_LIBERO:
        raise ScriptError(f"the scripted controller for {skill_id} is not verified in LIBERO")
    return fn


def make_script(name: str, env) -> Script:
    """Generator for a `ctl.run_scripted` script. reset_scene is not a motion (the runner resets)."""
    if name == "reset_scene":
        raise ScriptError("reset_scene is handled by the runner (env.reset), not a waypoint script")
    try:
        fn = SCRIPTS[name]
    except KeyError:
        raise ScriptError(f"unknown script {name}") from None
    return fn(env)
