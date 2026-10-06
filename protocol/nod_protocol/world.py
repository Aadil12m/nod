"""World state: the oracle view the sim publishes and the validator reads (3.5.5, 4.2.1)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import NodModel

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (w, x, y, z), MuJoCo order


class ObjPose(NodModel):
    """Pose of one object in the sim world frame (metres). `quat` is optional."""

    pos: Vec3
    quat: Quat | None = None


class Eef(NodModel):
    """End-effector position and binarised gripper state."""

    pos: Vec3
    gripper: Literal["open", "closed"]
    quat: Quat | None = None


class WorldState(NodModel):
    """Oracle world state.

    `atoms` maps ground atoms in registry form, e.g. "on(akita_black_bowl_1,plate_1)",
    to their truth value, as evaluated by LIBERO's `_eval_predicate`. The sim reports
    every atom that appears (positive or negated) in any registry `pre` or `effects`.
    An atom missing from `atoms` is UNKNOWN; see `nod_protocol.atoms.holds`.
    """

    scene_id: str
    step: int
    t_sim: float
    atoms: dict[str, bool] = Field(default_factory=dict)
    objects: dict[str, ObjPose] = Field(default_factory=dict)
    eef: Eef
