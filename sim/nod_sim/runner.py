"""SimRunner: the interruptible 20 Hz control loop that serves the gateway over ZMQ.

Design sections: 1.3 (chaining, goal swap), 3.3 and 3.5.4-3.5.6 (sockets and messages),
5.3 (execution-time limits), 5.5 (e-stop), 5.6 (watchdog).

Sockets (all bound here; the gateway and viewer connect):
    ctl    ROUTER  :5560   requests from the gateway DEALER. Replies (`sim.ack`, `sim.frames`,
                           `sim.world` for `ctl.world`) go back to the requesting peer.
    estop  PULL    :5561   `ctl.estop` only. Polled (non-blocking) before every env.step and
                           every 5 ms inside every wait.
    pub    PUB     :5562   `sim.status`, `sim.estop_ack`, `sim.world`, `sim.episode_header`
    pose   PUB     :5563   `pose` for the viewer, every control step while running

Threads: everything runs on the thread that calls `serve_forever` (or `run_*`), including
all socket I/O. The only other thread is one policy worker that owns the policy client
(an Isaac-GR00T REQ socket is not thread-safe). The control thread never blocks on it:
waiting for a chunk is a loop of `stop_event.wait(0.005)` slices that also polls the
e-stop and ctl sockets, so a stop is never delayed by inference or by the pacing sleep.

E-stop (5.5): on `ctl.estop` the loop performs a controlled stop (4 zero-motion steps with
the gripper command held, run without pacing), halts physics stepping, increments `epoch`,
drops every action chunk from the old epoch (queued, in flight or late), marks the run
`aborted` and publishes `sim.estop_ack`. A run is never resumed after an e-stop. A repeated
`estop_seq` is acknowledged again without a second epoch bump.

Pause (5.6): `ctl.pause`, or no `ctl.hb` for `hb_timeout_ms` while a run is executing,
performs the same controlled stop and holds the run in phase `paused` (fault `hb_lost`
for the watchdog). `ctl.resume` continues it with a fresh chunk; `ctl.abort` ends it.

Limits (5.3), applied to every action before env.step:
    * every component finite and |a| <= limits.max_abs_action, else fault `nan_action`
    * predicted eef (current + 0.05 * action_xyz) inside limits.eef_box, else zero the
      translation for that step; 3 consecutive violations raise fault `eef_out_of_box`
    * run.step reaching max_steps raises fault `budget` (phase `failed`)
    * no chunk within policy_timeout_ms raises fault `policy_timeout`
Every fault ends with a controlled stop.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zmq
from nod_protocol import (
    CTL,
    Abort,
    Capture,
    CtlEStop,
    CtlHeartbeat,
    Limits,
    LoadScene,
    ModelPatch,
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
    dumps,
    load_skills,
    loads,
    now_ms,
    registry_atoms,
    registry_objects,
)

from .capture import capture_frames
from .env_api import CONTROL_HZ, GRIP_OPEN, OSC_MAX_DELTA, Policy, PolicyError, SimEnv, hold_action
from .oracle import world_state
from .recorder import EpisodeRecorder
from .scripted import ScriptError, make_script, scripted_skill

log = logging.getLogger("nod_sim.runner")

CONTROLLED_STOP_STEPS = 4  # 5.5: zero translation and rotation, gripper held
WAIT_SLICE_S = 0.005  # 5.5: interruptible waits poll every 5 ms
MAX_EEF_VIOLATIONS = 3  # 5.3
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "registry" / "skills.yaml"


@dataclass
class SimAddrs:
    ctl: str = "tcp://*:5560"
    estop: str = "tcp://*:5561"
    pub: str = "tcp://*:5562"
    pose: str = "tcp://*:5563"


@dataclass
class RunResult:
    """Outcome of one run, returned by `run_policy_skill` / `run_scripted` (also used by nod_eval)."""

    run_id: str
    kind: str  # "policy" | "scripted"
    skill_id: str
    actor: str  # "policy" (GR00T) | "scripted" (waypoint controller)
    phase: str  # "done" | "failed" | "aborted"
    success: bool
    fault: str | None
    steps: int
    stopped_at_step: int | None = None
    error: str | None = None
    policy_ms: list[float] = field(default_factory=list)
    step_ms: list[float] = field(default_factory=list)
    discarded_chunks: int = 0
    wall_s: float = 0.0
    already_true: bool = False
    record_dir: str | None = None


class _Fault(Exception):
    def __init__(self, fault: str, detail: str | None = None) -> None:
        super().__init__(detail or fault)
        self.fault = fault
        self.detail = detail


@dataclass
class _Run:
    """Mutable state of the run in flight."""

    run_id: str
    kind: str
    skill_id: str
    actor: str
    max_steps: int
    limits: Limits
    action_scale: float = 1.0
    n_action_steps: int = 8
    script: Iterator[np.ndarray] | None = None
    phase: str = "executing"
    step: int = 0
    fault: str | None = None
    error: str | None = None
    success: bool = False
    stopped_at_step: int | None = None
    grip_cmd: float = GRIP_OPEN
    violations: int = 0
    queue: deque = field(default_factory=deque)
    pause_requested: bool = False
    resume_requested: bool = False
    abort_requested: bool = False
    estopped: bool = False
    policy_ms: list[float] = field(default_factory=list)
    step_ms: list[float] = field(default_factory=list)
    discarded: int = 0
    script_result: bool | None = None
    t0: float = field(default_factory=time.monotonic)
    recorder: EpisodeRecorder | None = None


class SimRunner:
    """Serve one continuous sim session to the gateway (design 4.2.4).

    `env` is any `SimEnv` (LiberoSession in production, FakeEnv in tests). `policy` is
    any `Policy` (Gr00tPolicy in production) or None, in which case policy skills run on
    the scripted fallback actor (tier T4) when one exists for that skill.

    `addrs=None` runs without sockets: published messages are appended to `self.published`
    (used by nod_eval and tests). `hz=None` disables real-time pacing (nightly eval).
    """

    def __init__(
        self,
        env: SimEnv,
        policy: Policy | None,
        addrs: SimAddrs | None = None,
        hz: float | None = CONTROL_HZ,
        *,
        registry: str | Path | None = DEFAULT_REGISTRY,
        scene_id: str = "libero_goal_v1",
        init_index: int = 0,
        hb_timeout_ms: int | None = 500,
        policy_timeout_ms: int = 2000,
        record_dir: str | Path | None = None,
        status_every: int = 2,
        world_every: int = 20,
        publish_poses: bool = True,
        actor: str = "policy",
        context: zmq.Context | None = None,
    ) -> None:
        if actor not in ("policy", "scripted"):
            raise ValueError("actor must be 'policy' or 'scripted'")
        self.env = env
        self.policy = policy
        self.addrs = addrs
        self.hz = hz
        self.scene_id = scene_id
        self.init_index = init_index
        self.hb_timeout_s = None if hb_timeout_ms is None else hb_timeout_ms / 1000.0
        self.policy_timeout_s = policy_timeout_ms / 1000.0
        self.record_dir = Path(record_dir) if record_dir else None
        self.status_every = max(1, status_every)
        self.world_every = max(1, world_every)
        self.publish_poses = publish_poses
        self.actor = actor

        specs = load_skills(registry) if registry and Path(registry).exists() else []
        self.specs = {s.skill_id: s for s in specs}
        self.atoms = registry_atoms(specs)
        self.objects = registry_objects(specs)

        self.epoch = 0
        self.sim_step = 0  # env steps since start (all runs, stops and idle)
        self.run: _Run | None = None
        self.last_hb = time.monotonic()
        self.published: list[Any] = []  # filled when there are no sockets
        self.handled_estops: dict[int, SimEStopAck] = {}
        self.last_result: RunResult | None = None

        self.stop_event = threading.Event()  # the e-stop Event every wait sleeps on (5.5)
        self._ext_estops: deque[tuple[CtlEStop, int]] = deque()  # from request_estop()
        self._pending_estops: list[tuple[CtlEStop, int]] = []
        self._next_cmd: RunSkill | RunScripted | None = None
        self._shutdown = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy")
        self._inflight: cf.Future | None = None
        self._ctx = context
        self._own_ctx = False
        self._ctl = self._estop = self._pub = self._pose = None
        self._last_idle_pub = 0.0

    # ------------------------------------------------------------------ sockets

    def open(self) -> None:
        """Bind the four sockets (no-op when addrs is None)."""
        if self.addrs is None or self._ctl is not None:
            return
        if self._ctx is None:
            self._ctx, self._own_ctx = zmq.Context(), True
        ctx = self._ctx
        self._ctl = ctx.socket(zmq.ROUTER)
        self._estop = ctx.socket(zmq.PULL)
        self._pub = ctx.socket(zmq.PUB)
        self._pose = ctx.socket(zmq.PUB)
        for sock, addr in (
            (self._ctl, self.addrs.ctl),
            (self._estop, self.addrs.estop),
            (self._pub, self.addrs.pub),
            (self._pose, self.addrs.pose),
        ):
            sock.setsockopt(zmq.LINGER, 0)
            sock.bind(addr)
        # Viewers never queue old poses: keep only the newest message per subscriber.
        self._pose.setsockopt(zmq.SNDHWM, 2)

    def close(self) -> None:
        for sock in (self._ctl, self._estop, self._pub, self._pose):
            if sock is not None:
                sock.close(0)
        self._ctl = self._estop = self._pub = self._pose = None
        if self._own_ctx and self._ctx is not None:
            self._ctx.term()
            self._ctx = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _publish(self, msg) -> None:
        if self._pub is not None:
            self._pub.send(dumps(msg))
        else:
            self.published.append(msg)

    def _reply(self, envelope: list[bytes] | None, msg) -> None:
        if envelope is not None and self._ctl is not None:
            self._ctl.send_multipart([*envelope, dumps(msg)])
        else:
            self.published.append(msg)

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> "SimRunner":
        """Run serve_forever on a background thread (tests, embedding)."""
        self._thread = threading.Thread(target=self.serve_forever, name="sim-runner", daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("SimRunner did not start")
        return self

    def shutdown(self, timeout: float = 5.0) -> None:
        self._shutdown.set()
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def serve_forever(self) -> None:
        """Serve ctl requests until shutdown(). Runs start here, one at a time."""
        self.open()
        self._publish_header()
        self._publish_world()
        self._ready.set()
        try:
            while not self._shutdown.is_set():
                self._poll_estop()
                self._handle_estops()
                self._service_ctl()
                if self._next_cmd is not None:
                    cmd, self._next_cmd = self._next_cmd, None
                    if isinstance(cmd, RunSkill):
                        self.run_policy_skill(cmd)
                    else:
                        self.run_scripted(cmd)
                    continue
                now = time.monotonic()
                if now - self._last_idle_pub >= 0.5:  # idle: world and pose at 2 Hz
                    self._last_idle_pub = now
                    self._publish_world()
                    self._publish_pose()
                self._sleep_until(time.monotonic() + 0.02, idle=True)
        finally:
            self.close()

    # ------------------------------------------------------------------ e-stop

    def request_estop(self, estop_seq: int, t_gw: int | None = None) -> None:
        """Thread-safe e-stop entry for in-process callers (same handling as the socket)."""
        self._ext_estops.append((CtlEStop(estop_seq=estop_seq, t_gw=t_gw or now_ms()), now_ms()))
        self.stop_event.set()

    def _poll_estop(self) -> bool:
        """Drain the dedicated e-stop socket without blocking. True if a stop is pending."""
        while self._ext_estops:
            self._pending_estops.append(self._ext_estops.popleft())
        if self._estop is not None:
            while True:
                try:
                    data = self._estop.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                t_rx = now_ms()
                try:
                    msg = loads(data, CTL)
                except Exception as exc:  # never let a malformed frame block the stop path
                    log.warning("bad message on e-stop socket: %s", exc)
                    continue
                if isinstance(msg, CtlEStop):
                    self._pending_estops.append((msg, t_rx))
        if self._pending_estops:
            self.stop_event.set()
        return bool(self._pending_estops)

    def _handle_estops(self) -> None:
        """Act on every pending e-stop: controlled stop, epoch bump, ack. Runs on the control thread."""
        if not self._pending_estops:
            if self._shutdown.is_set():
                return
            self.stop_event.clear()
            return
        pending, self._pending_estops = self._pending_estops, []
        for msg, t_rx in pending:
            prev = self.handled_estops.get(msg.estop_seq)
            if prev is not None:  # duplicate: acknowledge again, no second stop or epoch bump
                self._publish(prev)
                continue
            stopped_at = None
            r = self.run
            if r is not None and r.phase not in ("done", "failed", "aborted"):
                stopped_at = self.controlled_stop() if r.phase == "executing" else r.step
                r.phase, r.estopped, r.stopped_at_step = "aborted", True, stopped_at
            self.epoch += 1
            self._drop_chunks(r)
            ack = SimEStopAck(
                estop_seq=msg.estop_seq, epoch=self.epoch, stopped_at_step=stopped_at, t_rx=t_rx, t_halt=now_ms()
            )
            self.handled_estops[msg.estop_seq] = ack
            self._publish(ack)
            if r is not None and r.recorder is not None:
                r.recorder.event("estop", estop_seq=msg.estop_seq, epoch=self.epoch, stopped_at_step=stopped_at,
                                 sim_halt_ms=ack.t_halt - ack.t_rx)
        if not self._shutdown.is_set():
            self.stop_event.clear()

    def _drop_chunks(self, r: _Run | None) -> None:
        """Discard every action chunk from the old epoch: queued actions and any request in flight."""
        if r is not None and r.queue:
            r.queue.clear()
            r.discarded += 1
        if self._inflight is not None:
            if r is not None:
                r.discarded += 1
            self._inflight = None  # its result is never read

    def controlled_stop(self) -> int:
        """Category 1 analogue (5.5): 4 zero-motion steps holding the gripper, then stop stepping.

        Runs without pacing so the halt completes as fast as physics allows. Returns the
        run step at which the stop began (stopped_at_step).
        """
        r = self.run
        grip = r.grip_cmd if r is not None else GRIP_OPEN
        stopped_at = r.step if r is not None else self.sim_step
        a = hold_action(grip)
        for _ in range(CONTROLLED_STOP_STEPS):
            self.env.step(a)
            self.sim_step += 1
            if r is not None and r.recorder is not None:
                r.recorder.step(self.sim_step, a, self.env, stop=True)
            self._publish_pose()
        if r is not None:
            r.queue.clear()
        return stopped_at

    # ------------------------------------------------------------------ ctl

    def _service_ctl(self) -> None:
        if self._ctl is None:
            return
        while True:
            try:
                frames = self._ctl.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return
            envelope, data = frames[:-1], frames[-1]
            try:
                msg = loads(data, CTL)
            except Exception as exc:
                log.warning("bad ctl message: %s", exc)
                continue
            try:
                self.handle_ctl(msg, envelope)
            except Exception as exc:  # a bad request must not kill the control loop
                log.exception("ctl handler failed for %s", msg.type)
                req_id = getattr(msg, "req_id", None)
                if req_id:
                    self._reply(envelope, SimAck(req_id=req_id, ok=False, error=f"{type(exc).__name__}: {exc}"))

    def handle_ctl(self, msg, envelope: list[bytes] | None = None) -> None:
        """Dispatch one ctl message. Public so tests and nod_eval can drive the runner without sockets."""
        r = self.run
        if isinstance(msg, CtlHeartbeat):
            if msg.phone_ok:
                self.last_hb = time.monotonic()
            return
        if isinstance(msg, CtlEStop):  # tolerate an e-stop on the ctl socket; same handling
            self._pending_estops.append((msg, now_ms()))
            self.stop_event.set()
            return
        if isinstance(msg, LoadScene):
            if r is not None:
                return self._ack(envelope, msg.req_id, False, "busy: a run is in flight")
            if msg.scene_id != self.scene_id:
                return self._ack(envelope, msg.req_id, False, f"unknown scene {msg.scene_id}")
            self.epoch = max(self.epoch, msg.epoch)
            self.init_index = msg.init_index
            self.env.reset(msg.init_index)
            self._ack(envelope, msg.req_id, True)
            self._publish_header()
            self._publish_world()
            return
        if isinstance(msg, (RunSkill, RunScripted)):
            if msg.epoch < self.epoch:
                return self._ack(envelope, msg.req_id, False, f"stale epoch {msg.epoch} < {self.epoch}")
            if r is not None or self._next_cmd is not None:
                return self._ack(envelope, msg.req_id, False, "busy: a run is in flight")
            problem = self._check_run(msg)
            if problem:
                return self._ack(envelope, msg.req_id, False, problem)
            self.epoch = max(self.epoch, msg.epoch)
            self._next_cmd = msg
            return self._ack(envelope, msg.req_id, True)
        if isinstance(msg, (Pause, Resume, Abort)):
            if msg.epoch < self.epoch:
                return self._ack(envelope, msg.req_id, False, f"stale epoch {msg.epoch} < {self.epoch}")
            if isinstance(msg, Abort):
                if r is not None:
                    r.abort_requested = True
                    self.stop_event.set()
                elif self._next_cmd is not None:
                    self._next_cmd = None
                return self._ack(envelope, msg.req_id, True)
            if r is None:
                return self._ack(envelope, msg.req_id, False, "no run")
            if isinstance(msg, Pause):
                r.pause_requested = True
                self.stop_event.set()
            else:
                r.resume_requested = True
            return self._ack(envelope, msg.req_id, True)
        if isinstance(msg, Capture):
            frames = capture_frames(self.env, msg.cameras, msg.w, msg.h, step=self.sim_step)
            return self._reply(envelope, SimFrames(req_id=msg.req_id, step=self.sim_step, frames=frames))
        if isinstance(msg, WorldRequest):
            return self._reply(envelope, self._world_msg())
        log.warning("unhandled ctl message %s", msg.type)

    def _ack(self, envelope, req_id: str, ok: bool, error: str | None = None) -> None:
        self._reply(envelope, SimAck(req_id=req_id, ok=ok, error=error))

    def _check_run(self, msg: RunSkill | RunScripted) -> str | None:
        if isinstance(msg, RunScripted):
            return None  # every ScriptName has a script (validated by the protocol model)
        goals = getattr(self.env, "goals", None)
        if goals is not None and msg.goal_ref not in goals:
            return f"unknown goal_ref {msg.goal_ref}"
        if self.policy is None or self.actor == "scripted":
            try:
                scripted_skill(msg.skill_id)
            except ScriptError as exc:
                return str(exc)
        return None

    # ------------------------------------------------------------------ publishing

    def _world_msg(self) -> SimWorld:
        w = world_state(self.env, self.atoms, self.objects, scene_id=self.scene_id, step=self.sim_step)
        return SimWorld.from_world(w)

    def _publish_world(self) -> None:
        self._publish(self._world_msg())

    def _publish_header(self) -> None:
        info = self.env.episode_info()
        self._publish(
            SimEpisodeHeader(
                scene_id=info.scene_id,
                model_hash=info.model_hash,
                mj_ver=info.mj_ver,
                body_names=info.body_names,
                dyn_body_ids=info.dyn_body_ids,
                model_patch=ModelPatch.model_validate(info.model_patch),
            )
        )

    def _publish_pose(self) -> None:
        if not self.publish_poses or self._pose is None:
            return  # without sockets poses are not collected (they would dominate `published`)
        xpos, xquat, qpos = self.env.body_poses()
        msg = Pose(epoch=self.epoch, step=self.sim_step, t_sim=float(self.env.t_sim), t_wall=time.time(),
                   xpos=xpos, xquat=xquat, qpos=qpos)
        try:
            self._pose.send(dumps(msg), zmq.NOBLOCK)
        except zmq.Again:  # slow viewer: newest pose wins, drop this one
            pass

    def _status(self, r: _Run) -> SimStatus:
        return SimStatus(
            run_id=r.run_id,
            epoch=self.epoch,
            phase=r.phase,  # type: ignore[arg-type]
            step=r.step,
            success=r.success,
            fault=r.fault,  # type: ignore[arg-type]
            policy_ms=round(r.policy_ms[-1], 1) if r.policy_ms else None,
            step_ms=round(r.step_ms[-1], 1) if r.step_ms else None,
        )

    def _publish_status(self, r: _Run) -> None:
        self._publish(self._status(r))

    # ------------------------------------------------------------------ waits

    def _hb_lost(self) -> bool:
        return self.hb_timeout_s is not None and time.monotonic() - self.last_hb > self.hb_timeout_s

    def _interrupted(self) -> bool:
        """Poll both sockets; True if the run must stop stepping (e-stop, abort, pause, hb loss)."""
        if self._poll_estop():
            return True
        self._service_ctl()
        r = self.run
        if self._pending_estops:
            return True
        if r is not None and (r.abort_requested or r.pause_requested or (r.phase == "executing" and self._hb_lost())):
            return True
        if self.stop_event.is_set() and not self._ext_estops:
            self.stop_event.clear()  # a wake-up with nothing left to do: keep waits sleeping
        return False

    def _sleep_until(self, t_end: float, idle: bool = False) -> None:
        """Pacing sleep in 5 ms slices on the e-stop Event; returns early on any interrupt."""
        while True:
            if idle:
                if self._poll_estop():
                    return
                self._service_ctl()
                if self._next_cmd is not None or self._shutdown.is_set():
                    return
            elif self._interrupted():
                return
            remaining = t_end - time.monotonic()
            if remaining <= 0:
                return
            self.stop_event.wait(min(WAIT_SLICE_S, remaining))

    # ------------------------------------------------------------------ runs

    def run_policy_skill(self, cmd: RunSkill) -> RunResult:
        """Run one learned skill (or its scripted stand-in when actor is scripted), interruptibly."""
        use_script = self.policy is None or self.actor == "scripted"
        r = _Run(
            run_id=cmd.run_id,
            kind="policy",
            skill_id=cmd.skill_id,
            actor="scripted" if use_script else "policy",
            max_steps=cmd.max_steps,
            limits=cmd.limits,
            action_scale=cmd.action_scale,
            n_action_steps=cmd.n_action_steps,
        )
        self.env.set_skill(cmd.instruction, cmd.goal_ref)
        if use_script:
            try:
                r.script = scripted_skill(cmd.skill_id)(self.env)
            except ScriptError as exc:
                r.phase, r.fault, r.error = "failed", None, str(exc)
                return self._finish(r)
        return self._execute(r, meta={"instruction": cmd.instruction, "goal_ref": cmd.goal_ref,
                                      "n_action_steps": cmd.n_action_steps, "req_id": cmd.req_id})

    def run_scripted(self, cmd: RunScripted) -> RunResult:
        """Run one scripted waypoint motion (return_home, reset_scene, close_middle_drawer, handover_retreat)."""
        r = _Run(run_id=cmd.run_id, kind="scripted", skill_id=cmd.script, actor="scripted",
                 max_steps=cmd.max_steps, limits=cmd.limits)
        if cmd.script == "reset_scene":
            # Not an arm motion: restores the fixed initial layout (LIBERO reset + init state),
            # shown to the user as the scripted "tidy the table" step (tier T3, 1.6).
            self.run = r
            self._publish_status(r)
            self.env.reset(self.init_index)
            self._publish_header()
            r.phase, r.success = "done", True
            return self._finish(r)
        r.script = make_script(cmd.script, self.env)
        return self._execute(r, meta={"script": cmd.script, "req_id": cmd.req_id})

    def _execute(self, r: _Run, meta: dict | None = None) -> RunResult:
        self.run = r
        if self.record_dir is not None:
            r.recorder = EpisodeRecorder(self.record_dir / r.run_id, run_id=r.run_id, skill_id=r.skill_id,
                                         kind=r.kind, actor=r.actor, env=self.env, epoch=self.epoch, meta=meta or {})
        # The R5 validator normally prevents this; a goal that already holds ends at once.
        if r.kind == "policy" and self.env.check_success():
            r.phase, r.success = "done", True
            return self._finish(r, already_true=True)
        self._publish_status(r)
        next_t = time.monotonic()
        try:
            while True:
                self._poll_estop()
                self._handle_estops()
                if r.phase == "aborted":  # e-stop handled above
                    return self._finish(r)
                self._service_ctl()
                if self._pending_estops:
                    continue
                if r.abort_requested:
                    if r.phase == "executing":
                        r.stopped_at_step = self.controlled_stop()
                    r.phase = "aborted"
                    return self._finish(r)
                if r.phase == "executing" and (r.pause_requested or self._hb_lost()):
                    self._enter_pause(r, None if r.pause_requested else "hb_lost")
                    continue
                if r.phase == "paused":
                    r.pause_requested = False
                    if r.resume_requested:
                        r.resume_requested = False
                        r.phase, r.fault = "executing", None
                        self.last_hb = time.monotonic()  # grace period for the heartbeat
                        self._publish_status(r)
                        if r.recorder:
                            r.recorder.event("resume", step=r.step)
                        next_t = time.monotonic()
                    else:
                        self._sleep_until(time.monotonic() + 0.02)
                    continue
                r.resume_requested = False
                if r.step >= r.max_steps:
                    raise _Fault("budget")
                action = self._next_action(r)
                if action is None:  # interrupted, stale chunk dropped, or the script ended
                    if r.phase in ("done", "failed"):
                        return self._finish(r)
                    continue
                success = self._apply(r, action)
                if success:
                    r.phase, r.success = "done", True
                    return self._finish(r)
                if self.hz:
                    next_t += 1.0 / self.hz
                    if time.monotonic() - next_t > 1.0:  # far behind (debugger, slow render): resync
                        next_t = time.monotonic()
                    self._sleep_until(next_t)
        except _Fault as f:
            r.fault, r.error = f.fault, f.detail
            r.stopped_at_step = self.controlled_stop()
            r.phase = "failed"
            if r.recorder:
                r.recorder.event("fault", fault=f.fault, detail=f.detail, step=r.step)
            return self._finish(r)

    def _enter_pause(self, r: _Run, fault: str | None) -> None:
        self._inflight = None  # a chunk computed before the pause is never executed
        r.stopped_at_step = self.controlled_stop()
        r.phase, r.fault, r.pause_requested = "paused", fault, False
        if r.queue:  # the observation behind the rest of this chunk is stale now
            r.queue.clear()
        self._publish_status(r)
        if r.recorder:
            r.recorder.event("pause", fault=fault, step=r.step)

    def _next_action(self, r: _Run) -> np.ndarray | None:
        if r.script is not None:
            try:
                return next(r.script)
            except StopIteration as stop:
                r.script_result = bool(stop.value) if stop.value is not None else True
                r.phase, r.success = "done", r.script_result
                if not r.script_result:
                    r.phase, r.error = "failed", "scripted motion did not reach its target"
                    return None
                return None
        if not r.queue:
            chunk = self._request_chunk(r)
            if chunk is None:
                return None
            r.queue.extend(chunk[: max(1, r.n_action_steps)])
        return r.queue.popleft()

    def _request_chunk(self, r: _Run) -> np.ndarray | None:
        """Ask the policy worker for a chunk and wait for it interruptibly (5 ms slices)."""
        assert self.policy is not None
        epoch = self.epoch
        obs = self.env.observation()
        t0 = time.perf_counter()
        fut = self._executor.submit(self.policy.get_action, obs)
        self._inflight = fut
        deadline = time.monotonic() + self.policy_timeout_s
        while not fut.done():
            if self._interrupted():
                return None  # the e-stop/pause handler drops the in-flight chunk
            if time.monotonic() > deadline:
                self._inflight = None
                self._replace_executor()
                raise _Fault("policy_timeout", f"no chunk within {self.policy_timeout_s * 1000:.0f} ms")
            self.stop_event.wait(WAIT_SLICE_S)
        if self._inflight is not fut or epoch != self.epoch:
            r.discarded += 1  # arrived after an e-stop moved the epoch on: never executed
            return None
        self._inflight = None
        try:
            chunk = fut.result()
        except PolicyError as exc:
            raise _Fault("policy_timeout", f"{type(exc).__name__}: {exc}") from exc
        except Exception as exc:  # any other client failure also leaves us without a chunk
            raise _Fault("policy_timeout", f"{type(exc).__name__}: {exc}") from exc
        ms = (time.perf_counter() - t0) * 1000.0
        r.policy_ms.append(ms)
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 7 or chunk.shape[0] < 1:
            raise _Fault("nan_action", f"chunk shape {list(chunk.shape)} is not [T, 7]")
        if r.recorder:
            r.recorder.chunk(r.step, ms, epoch)
        return chunk

    def _replace_executor(self) -> None:
        # A hung worker keeps its thread; later requests go to a fresh one.
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy")

    def _apply(self, r: _Run, action: np.ndarray) -> bool:
        """Check limits (5.3), step the env once, publish. Returns goal success."""
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        lim = r.limits
        if a.shape != (7,) or not np.all(np.isfinite(a)) or np.any(np.abs(a) > lim.max_abs_action + 1e-6):
            raise _Fault("nan_action", f"action {a.tolist()} outside [-{lim.max_abs_action}, {lim.max_abs_action}]")
        a = a.copy()
        a[:6] *= r.action_scale
        lo, hi = np.asarray(lim.eef_box[0]), np.asarray(lim.eef_box[1])
        predicted = self.env.eef_pos() + OSC_MAX_DELTA * a[:3]
        if np.any(predicted < lo) or np.any(predicted > hi):
            r.violations += 1
            if r.violations >= MAX_EEF_VIOLATIONS:
                raise _Fault("eef_out_of_box", f"predicted eef {np.round(predicted, 3).tolist()} outside the box")
            a[:3] = 0.0
        else:
            r.violations = 0
        r.grip_cmd = float(a[6])
        t0 = time.perf_counter()
        success = bool(self.env.step(a))
        r.step_ms.append((time.perf_counter() - t0) * 1000.0)
        r.step += 1
        self.sim_step += 1
        if r.recorder:
            r.recorder.step(self.sim_step, a, self.env)
        self._publish_pose()
        if r.step % self.status_every == 0:
            self._publish_status(r)
        if r.step % self.world_every == 0:
            self._publish_world()
        # A learned policy ends on the goal predicate. Scripted motions (including scripted
        # stand-ins for policy skills) end when the script returns; _finish checks the goal.
        return success if r.script is None else False

    def _finish(self, r: _Run, already_true: bool = False) -> RunResult:
        if r.phase == "executing":
            r.phase = "failed"
        if r.phase == "done" and r.kind == "policy":
            r.success = bool(self.env.check_success())
            if not r.success:  # a scripted stand-in finished without reaching the goal
                r.phase = "failed"
        self._publish_status(r)
        self._publish_world()
        wall = time.monotonic() - r.t0
        result = RunResult(
            run_id=r.run_id, kind=r.kind, skill_id=r.skill_id, actor=r.actor, phase=r.phase, success=r.success,
            fault=r.fault, steps=r.step, stopped_at_step=r.stopped_at_step, error=r.error, policy_ms=r.policy_ms,
            step_ms=r.step_ms, discarded_chunks=r.discarded, wall_s=round(wall, 3), already_true=already_true,
        )
        if r.recorder is not None:
            result.record_dir = str(r.recorder.finish(result, world=self._world_msg()))
        self.run = None
        self.last_result = result
        log.info("run %s %s -> %s (fault=%s, steps=%d)", r.run_id, r.skill_id, r.phase, r.fault, r.step)
        return result


# ---------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    """`python -m nod_sim.runner --policy tcp://policy:5555` (compose service `sim`)."""
    ap = argparse.ArgumentParser(description="Nod sim runner (LIBERO-Goal session served over ZMQ)")
    ap.add_argument("--policy", default=os.environ.get("NOD_POLICY", "tcp://127.0.0.1:5555"),
                    help="Isaac-GR00T policy server, tcp://host:port; 'none' runs the scripted actor (T4)")
    ap.add_argument("--actor", choices=["policy", "scripted"], default=os.environ.get("NOD_ACTOR", "policy"))
    ap.add_argument("--ctl", default="tcp://*:5560")
    ap.add_argument("--estop", default="tcp://*:5561")
    ap.add_argument("--pub", default="tcp://*:5562")
    ap.add_argument("--pose", default="tcp://*:5563")
    ap.add_argument("--scene", default="libero_goal_v1")
    ap.add_argument("--init-index", type=int, default=0)
    ap.add_argument("--registry", default=os.environ.get("NOD_REGISTRY", str(DEFAULT_REGISTRY)))
    ap.add_argument("--record-dir", default=os.environ.get("NOD_RECORD_DIR"))
    ap.add_argument("--hz", type=float, default=CONTROL_HZ)
    ap.add_argument("--hb-timeout-ms", type=int, default=500)
    ap.add_argument("--policy-timeout-ms", type=int, default=2000)
    ap.add_argument("--lazy-cameras", action="store_true",
                    help="render policy cameras only when a chunk is requested (CPU); identical images")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from .scene import make_session_env

    env = make_session_env(args.scene, args.init_index, lazy_cameras=args.lazy_cameras)
    policy = None
    if args.policy != "none" and args.actor == "policy":
        from .policy import Gr00tPolicy

        policy = Gr00tPolicy.from_url(args.policy, timeout_ms=args.policy_timeout_ms)
        if not policy.ping():
            log.warning("policy server %s is not answering yet; runs will fault policy_timeout until it is", args.policy)
    runner = SimRunner(env, policy, SimAddrs(args.ctl, args.estop, args.pub, args.pose), hz=args.hz,
                       registry=args.registry, scene_id=args.scene, init_index=args.init_index,
                       hb_timeout_ms=args.hb_timeout_ms, policy_timeout_ms=args.policy_timeout_ms,
                       record_dir=args.record_dir, actor=args.actor if policy is not None else "scripted")
    log.info("sim runner serving %s (actor=%s)", args.scene, runner.actor)
    try:
        runner.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
