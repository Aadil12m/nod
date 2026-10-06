"""Oracle world state (design 3.5.5 `sim.world`, 4.2.4 oracle.py).

The oracle evaluates registry atoms ("on(akita_black_bowl_1,plate_1)") with LIBERO's own
predicate code (`problem_env._eval_predicate`), the same code `_check_success` uses, so the
validator's preconditions and the run's success check can never disagree.

It exists only because this is a simulation. On a real robot Cosmos would be the only check
(design 5.4).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from nod_protocol import Eef, ObjPose, WorldState, parse_atom


def atom_to_state(atom: str) -> list[str]:
    """Registry atom -> LIBERO goal-state list: "on(a,b)" -> ["on", "a", "b"]."""
    pred, args = parse_atom(atom)
    return [pred, *args]


def state_to_atom(state: list[str] | tuple[str, ...]) -> str:
    """LIBERO goal-state list -> registry atom: ["On", "a", "b"] -> "on(a,b)"."""
    pred, *args = state
    return f"{str(pred).lower()}({','.join(str(a).lower() for a in args)})"


def libero_eval_atom(problem_env: Any, atom: str) -> bool | None:
    """Evaluate one atom on a LIBERO problem env (`OffScreenRenderEnv.env`).

    Returns None when an argument is not an object, fixture or region of this scene,
    so the atom is reported UNKNOWN (absent) rather than guessed.
    """
    state = atom_to_state(atom)
    known = problem_env.object_states_dict
    if any(arg not in known for arg in state[1:]):
        return None
    return bool(problem_env._eval_predicate(state))


def world_state(
    env: Any,
    atoms: Iterable[str],
    objects: Iterable[str] = (),
    *,
    scene_id: str | None = None,
    step: int | None = None,
) -> WorldState:
    """Snapshot of every registry atom, the named objects' poses and the end effector.

    `env` is a `SimEnv` (LiberoSession or FakeEnv). Atoms the scene cannot evaluate are
    left out of `atoms`, which the validator treats as not satisfied (nod_protocol.holds).
    """
    values: dict[str, bool] = {}
    for atom in atoms:
        v = env.eval_atom(atom)
        if v is not None:
            values[atom] = bool(v)
    objs: dict[str, ObjPose] = {}
    for name in objects:
        pos = env.object_pos(name)
        if pos is None:
            continue
        quat = getattr(env, "object_quat", lambda _n: None)(name)
        objs[name] = ObjPose(pos=_vec(pos, 3), quat=_vec(quat, 4) if quat is not None else None)
    eef = Eef(
        pos=_vec(env.eef_pos(), 3),
        gripper="closed" if env.gripper_is_closed() else "open",
        quat=_vec(env.eef_quat(), 4),
    )
    return WorldState(
        scene_id=scene_id or env.scene_id,
        step=int(env.step_count if step is None else step),
        t_sim=round(float(env.t_sim), 3),
        atoms=values,
        objects=objs,
        eef=eef,
    )


def _vec(v: Any, n: int) -> tuple[float, ...]:
    a = np.asarray(v, dtype=np.float64).reshape(-1)[:n]
    return tuple(round(float(x), 4) for x in a)
