"""Shared fixtures: a SimRunner over FakeEnv/FakePolicy, with or without real ZMQ sockets."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import zmq
from nod_protocol import CtlEStop, RunScripted, RunSkill, SIM_OUT, dumps, loads, new_id, now_ms

from nod_sim.fakes import FakeEnv, FakePolicy
from nod_sim.runner import SimAddrs, SimRunner

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "registry" / "skills.yaml"

GOALS = {
    "bowl_on_plate": ("put the bowl on the plate", "libero_goal/put_the_bowl_on_the_plate"),
    "cheese_in_bowl": ("put the cream cheese in the bowl", "libero_goal/put_the_cream_cheese_in_the_bowl"),
    "bowl_to_user": ("bring the bowl to me", "nod/bring_the_bowl_to_me"),
    "open_middle_drawer": ("open the middle drawer of the cabinet", "libero_goal/open_the_middle_drawer_of_the_cabinet"),
}


def run_skill_cmd(skill_id="bowl_on_plate", epoch=0, max_steps=400, **kw) -> RunSkill:
    instr, ref = GOALS[skill_id]
    return RunSkill(req_id=new_id("q"), run_id=new_id("r"), epoch=epoch, skill_id=skill_id,
                    instruction=instr, goal_ref=ref, max_steps=max_steps, **kw)


def run_scripted_cmd(script="return_home", epoch=0, max_steps=160) -> RunScripted:
    return RunScripted(req_id=new_id("q"), run_id=new_id("r"), epoch=epoch, script=script, max_steps=max_steps)


class TimedEnv(FakeEnv):
    """FakeEnv that timestamps every step, so tests can tell which steps came after an e-stop."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.step_times: list[float] = []
        self.on_step = None

    def step(self, action):
        self.step_times.append(time.monotonic())
        out = super().step(action)
        if self.on_step is not None:
            self.on_step(self)
        return out


@pytest.fixture
def env():
    return TimedEnv()


def make_runner(env, policy=None, **kw) -> SimRunner:
    kw.setdefault("hz", None)
    kw.setdefault("hb_timeout_ms", None)
    kw.setdefault("registry", REGISTRY)
    return SimRunner(env, policy, None, **kw)


class Gateway:
    """Test double for the gateway side of the ZMQ link (DEALER ctl, PUSH estop, SUB pub+pose)."""

    def __init__(self, ctx: zmq.Context, addrs: SimAddrs) -> None:
        self.ctl = ctx.socket(zmq.DEALER)
        self.est = ctx.socket(zmq.PUSH)
        self.sub = ctx.socket(zmq.SUB)
        self.pose = ctx.socket(zmq.SUB)
        for s in (self.ctl, self.est, self.sub, self.pose):
            s.setsockopt(zmq.LINGER, 0)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.pose.setsockopt(zmq.SUBSCRIBE, b"")
        self.ctl.connect(addrs.ctl)
        self.est.connect(addrs.estop)
        self.sub.connect(addrs.pub)
        self.pose.connect(addrs.pose)
        self.seen: list = []

    def send(self, msg) -> None:
        self.ctl.send(dumps(msg))

    def estop(self, seq: int) -> int:
        t = now_ms()
        self.est.send(dumps(CtlEStop(estop_seq=seq, t_gw=t)))
        return t

    def drain(self, timeout: float = 0.0) -> list:
        """Everything that arrived on SUB and DEALER within `timeout` seconds."""
        out = []
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)
        poller.register(self.ctl, zmq.POLLIN)
        end = time.monotonic() + timeout
        while True:
            left = max(0.0, end - time.monotonic())
            events = dict(poller.poll(int(left * 1000)))
            if not events:
                break
            for sock in events:
                msg = loads(sock.recv(), SIM_OUT)
                out.append(msg)
        self.seen.extend(out)
        return out

    def wait_for(self, pred, timeout: float = 5.0):
        end = time.monotonic() + timeout
        for m in self.seen:
            if pred(m):
                return m
        while time.monotonic() < end:
            for m in self.drain(0.02):
                if pred(m):
                    return m
        raise AssertionError("timed out waiting for a sim message")

    def close(self) -> None:
        for s in (self.ctl, self.est, self.sub, self.pose):
            s.close(0)


@pytest.fixture
def ipc_addrs(tmp_path):
    base = f"ipc://{tmp_path}/{uuid.uuid4().hex[:8]}"
    return SimAddrs(ctl=base + "-ctl", estop=base + "-estop", pub=base + "-pub", pose=base + "-pose")


@pytest.fixture
def served(env, ipc_addrs):
    """A SimRunner serving over ipc sockets on a thread, plus a connected Gateway double."""
    ctx = zmq.Context()
    started = []

    def start(policy=None, **kw):
        kw.setdefault("hz", 20)
        kw.setdefault("hb_timeout_ms", None)
        kw.setdefault("registry", REGISTRY)
        r = SimRunner(env, policy, ipc_addrs, context=ctx, **kw)
        r.start()
        gw = Gateway(ctx, ipc_addrs)
        time.sleep(0.15)  # let SUB connect (slow joiner)
        started.append((r, gw))
        return r, gw

    yield start
    for r, gw in started:
        r.shutdown()
        gw.close()
    ctx.term()


def moving_policy(dx=0.2, grip=1.0, horizon=16, **kw) -> FakePolicy:
    a = np.zeros(7, dtype=np.float32)
    a[0] = dx
    a[6] = grip
    return FakePolicy.constant(a, horizon=horizon, **kw)


def in_thread(fn, *args) -> tuple[threading.Thread, list]:
    out: list = []
    t = threading.Thread(target=lambda: out.append(fn(*args)), daemon=True)
    t.start()
    return t, out
