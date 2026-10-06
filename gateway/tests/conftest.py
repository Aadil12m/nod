"""Shared fixtures: the real registry, worlds, session views, free ports."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from nod_protocol import Eef, ObjPose, WorldState

from nod_gateway.registry import Registry, load_registry
from nod_gateway.validator import SafetyPolicy, SessionView

REPO = Path(__file__).resolve().parents[2]
SKILLS = REPO / "registry" / "skills.yaml"
STATS = REPO / "registry" / "stats" / "latest.json"


@pytest.fixture(scope="session")
def reg() -> Registry:
    return load_registry(SKILLS, STATS)


def make_world(
    reg: Registry,
    true_atoms: tuple[str, ...] = (),
    *,
    drop_objects: tuple[str, ...] = (),
    scene_id: str = "libero_goal_v1",
    unknown: tuple[str, ...] = (),
) -> WorldState:
    atoms = {a: (a in true_atoms) for a in reg.atoms if a not in unknown}
    objects = {o: ObjPose(pos=(0.0, 0.0, 0.9)) for o in reg.objects if o not in drop_objects}
    return WorldState(
        scene_id=scene_id,
        step=0,
        t_sim=0.0,
        atoms=atoms,
        objects=objects,
        eef=Eef(pos=(-0.21, 0.02, 1.12), gripper="open"),
    )


@pytest.fixture()
def world(reg: Registry) -> WorldState:
    return make_world(reg)


@pytest.fixture()
def pol() -> SafetyPolicy:
    return SafetyPolicy()


def view(**kw) -> SessionView:
    base = dict(mode="AWAITING_CONFIRM", lease="user", face_present=True, hb_fresh=True)
    base.update(kw)
    return SessionView(**base)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def sim_addrs() -> dict[str, str]:
    return {k: f"tcp://127.0.0.1:{free_port()}" for k in ("ctl", "estop", "pub", "pose")}
