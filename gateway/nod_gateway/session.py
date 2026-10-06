"""Session state machine (design 3.4, 5.4, 5.6, 5.7).

    IDLE --menu|utterance--> LISTENING --utter.end--> PLANNING --proposal--> AWAITING_CONFIRM
    AWAITING_CONFIRM --confirm ok--> COUNTDOWN(1.0 s) --> EXECUTING --> POSTCHECK --> IDLE (or next plan step)
    AWAITING_CONFIRM --cancel|expire--> IDLE
    EXECUTING --hb lost|face lost--> PAUSED --confirm resume--> EXECUTING ; --cancel--> IDLE
    any --estop--> ESTOPPED --estop.reset--> IDLE   (never resumes the run)

One run is in flight at a time. A plan runs as consecutive runs; by default each step is
confirmed (`plan_mode=per_step`); `plan_mode=whole` starts the next step after a cancellable
3 s countdown. Between learned skills a scripted `return_home` runs (or `handover_retreat`
after a near-user skill), then the post-check.

Long work (ASR, planner, Cosmos, the run itself) runs in tasks tracked in `self.tasks`;
`latch_estop` cancels all of them synchronously. Messages leave through `Outbox`, which
never blocks.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nod_protocol import (
    MENU_NO_SPEECH,
    CosmosCheck,
    CosmosResult,
    CosmosVerdict,
    EStopReset,
    FailureInfo,
    FeasibleSkill,
    FixedOption,
    Frame,
    FrameRef,
    HistoryItem,
    InfeasibleSkill,
    Notice,
    Option,
    OracleAtom,
    PlannerOption,
    PlannerOutput,
    PlanRequest,
    Precheck,
    Profile,
    Proposal,
    RunScripted,
    RunSkill,
    RunUpdate,
    SimStatus,
    SkillSpec,
    State,
    Transcript,
    UiCancel,
    UiConfirm,
    UiFocus,
    WorldState,
    holds,
    new_id,
    now_ms,
)

from .planner.base import PlannerError, PlannerResult
from .registry import Registry, deterministic_menu
from .safety import HbState
from .simlink import SimError
from .validator import (
    PROPOSAL_MODES,
    SafetyPolicy,
    SessionView,
    feasible_skills,
    oracle_success,
    post_decision,
    pre_decision,
    skill_rules,
    spacing_wait_ms,
    validate_confirm,
    validate_proposal,
)

log = logging.getLogger(__name__)

MAX_AUDIO_BYTES = 30 * 16000 * 2  # 30 s of 16 kHz PCM16


@dataclass
class Utterance:
    utt_id: int
    text: str | None = None
    pcm: bytes | None = None
    sr: int = 16000


@dataclass
class PlanCtx:
    proposal_id: str
    option_id: str
    skill_ids: list[str]
    step: int = 0


@dataclass
class LiveProposal:
    proposal: Proposal
    sent_ms: int
    purpose: str = "choose"  # choose | resume | plan_next
    focus_option: str | None = None
    focus_since_ms: int | None = None
    pending_double: tuple[str, str] | None = None
    plan: PlanCtx | None = None
    precheck: dict[str, CosmosResult | None] = field(default_factory=dict)
    precheck_events: dict[str, asyncio.Event] = field(default_factory=dict)


@dataclass
class RunCtx:
    run_id: str
    spec: SkillSpec
    plan: PlanCtx
    started_ms: int
    sim_run_id: str  # the run id of the sim motion in flight (the skill, or its return-home)
    queue: asyncio.Queue[SimStatus] = field(default_factory=asyncio.Queue)
    sim_step: int = 0
    phase: str = "countdown"
    fault: str | None = None
    last_forward_ms: int = 0
    precheck: CosmosResult | None = None


class Session:
    def __init__(
        self,
        cfg,
        reg: Registry,
        sim,
        planner,
        asr,
        verifier,
        home,
        out,
        hb: HbState,
        *,
        recorder,
        presenter,
        metrics,
        health,
        pol: SafetyPolicy | None = None,
        clock=now_ms,
    ) -> None:
        self.cfg = cfg
        self.reg = reg
        self.sim = sim
        self.planner = planner
        self.asr = asr
        self.verifier = verifier
        self.home = home
        self.out = out
        self.hb = hb
        self.recorder = recorder
        self.presenter = presenter
        self.metrics = metrics
        self.health = health
        self.pol = pol or SafetyPolicy.from_config(cfg)
        self.clock = clock
        self.sess_id: str = out.sess
        self.mode: str = "IDLE"
        self.epoch = 0
        self.world: WorldState | None = None
        self.live: LiveProposal | None = None
        self.run: RunCtx | None = None
        self.plan: PlanCtx | None = None
        self.tasks: set[asyncio.Task] = set()
        self.run_task: asyncio.Task | None = None
        self.used_nonces: deque[str] = deque(maxlen=1000)
        self.near_user_used: set[str] = set()
        self.history: deque[HistoryItem] = deque(maxlen=5)
        self.last_motion_end_ms: int | None = None
        self.in_chain = False
        self.caregiver_enabled = bool(cfg.caregiver_enabled)
        self.prefetched: tuple[frozenset, PlannerResult] | None = None
        self.pause_reason: str | None = None
        self.audio: dict[int, bytearray] = {}
        self.audio_sr: dict[int, int] = {}
        self.client_estop_seq = 0

    # ================================================================ views and output

    def snapshot(self) -> SessionView:
        now = self.clock()
        lp = self.live
        return SessionView(
            mode=self.mode,  # type: ignore[arg-type]
            lease="user",
            face_present=self.hb.face_present,
            hb_fresh=self.hb.fresh(now, self.cfg.hb_timeout_ms),
            caregiver_enabled=self.caregiver_enabled,
            in_chain=self.in_chain,
            proposal_id=lp.proposal.proposal_id if lp else None,
            proposal_sent_ms=lp.sent_ms if lp else None,
            proposal_purpose=lp.purpose if lp else "choose",  # type: ignore[arg-type]
            focus_option=lp.focus_option if lp else None,
            focus_since_ms=lp.focus_since_ms if lp else None,
            used_nonces=frozenset(self.used_nonces),
            near_user_used=frozenset(self.near_user_used),
            pending_double=lp.pending_double if lp else None,
            last_motion_end_ms=self.last_motion_end_ms,
        )

    def state_msg(self) -> State:
        return State(
            mode=self.mode,
            run_id=self.run.run_id if self.run else None,  # type: ignore[arg-type]
            scene_id=self.cfg.scene_id,
            estop_seq=self.client_estop_seq,
        )

    def _set_mode(self, mode: str) -> None:
        if mode != self.mode:
            log.info("session %s -> %s", self.mode, mode)
        self.mode = mode
        self.out.to_phone(self.state_msg())

    def _notice(self, text: str, level: str = "info", speak: bool = True) -> None:
        self.out.to_phone(Notice(level=level, text=text, speak=speak))  # type: ignore[arg-type]

    def _spawn(self, coro, name: str) -> asyncio.Task:
        t = asyncio.create_task(coro, name=name)
        self.tasks.add(t)
        t.add_done_callback(self._task_done)
        return t

    def _task_done(self, t: asyncio.Task) -> None:
        self.tasks.discard(t)
        if not t.cancelled() and t.exception() is not None:
            log.error("session task %s failed", t.get_name(), exc_info=t.exception())

    def _cancel_tasks(self, keep: asyncio.Task | None = None) -> None:
        for t in list(self.tasks):
            if t is not keep and not t.done():
                t.cancel()
        self.tasks = {t for t in self.tasks if t is keep}

    # ================================================================ world

    def on_world(self, world: WorldState) -> None:
        self.world = world

    async def fresh_world(self) -> WorldState:
        try:
            self.world = await self.sim.world(timeout=1.0)
            self.health.ok("sim")
        except SimError as exc:
            self.health.error("sim", str(exc))
            if self.world is None:
                raise
        assert self.world is not None
        return self.world

    # ================================================================ phone events

    async def on_menu(self, src: str) -> None:
        """Brow raise (or keyboard/button): show the prefetched set or the deterministic menu."""
        if self.mode != "IDLE":
            if self.mode == "AWAITING_CONFIRM":
                return  # already showing options
            self._notice("Not now", level="warn", speak=False)
            return
        self._set_mode("LISTENING")
        world = self.world
        if world is None:
            try:
                world = await self.fresh_world()
            except SimError:
                self._notice("The robot is not connected", level="error")
                self._set_mode("IDLE")
                return
        pre = self.prefetched
        if pre is not None and pre[0] == frozenset(world.atoms.items()):
            res = pre[1]
            self._offer(res.output, source=res.source, model=res.model, transcript=None, trace=res)
        else:
            self._offer_menu(world, transcript=None)

    async def on_focus(self, f: UiFocus) -> None:
        lp = self.live
        if lp is None or f.proposal_id != lp.proposal.proposal_id:
            return
        ids = {o.option_id for o in lp.proposal.options} | {o.option_id for o in lp.proposal.fixed}
        if f.option_id in ids and f.option_id != lp.focus_option:
            lp.focus_option, lp.focus_since_ms = f.option_id, self.clock()

    def on_utter_start(self, utt_id: int, sr: int) -> None:
        if self.mode not in ("IDLE", "LISTENING", "AWAITING_CONFIRM"):
            return
        self._drop_proposal()
        self.audio = {utt_id: bytearray()}
        self.audio_sr = {utt_id: sr}
        self._set_mode("LISTENING")

    def on_audio(self, utt_id: int, pcm: bytes) -> None:
        buf = self.audio.get(utt_id)
        if buf is not None and len(buf) + len(pcm) <= MAX_AUDIO_BYTES:
            buf.extend(pcm)

    async def on_utter_end(self, utt_id: int) -> None:
        buf = self.audio.pop(utt_id, None)
        sr = self.audio_sr.pop(utt_id, 16000)
        if buf is None:
            return
        await self.on_utterance(Utterance(utt_id=utt_id, pcm=bytes(buf), sr=sr))

    async def on_utterance(self, utt: Utterance) -> None:
        if self.mode not in PROPOSAL_MODES:
            self._notice("I can't take a request right now", level="warn", speak=False)
            return
        self._drop_proposal()
        for t in list(self.tasks):  # a newer utterance replaces any planning in progress
            if t.get_name().startswith("plan") and not t.done():
                t.cancel()
        self._set_mode("PLANNING")
        self._spawn(self._plan_flow(utt), f"plan-{utt.utt_id}")

    async def on_confirm(self, c: UiConfirm, lease: str = "user") -> None:
        lp = self.live
        if (
            lp is not None
            and c.proposal_id == lp.proposal.proposal_id
            and c.option_id in {o.option_id for o in lp.proposal.fixed}
        ):
            await self.on_cancel(UiCancel(proposal_id=c.proposal_id, source="none_of_these"))
            return
        if lp is None:
            self._notice("Try again", level="warn", speak=False)
            self.presenter.trace("validator", c.proposal_id, {"confirm": "reject", "reason": "no live proposal"})
            return
        view = dataclasses.replace(self.snapshot(), lease=lease)  # type: ignore[arg-type]
        d = validate_confirm(c, lp.proposal, view, self.clock(), self.pol)
        self.presenter.trace(
            "validator",
            lp.proposal.proposal_id,
            {"confirm": d.outcome, "rule": d.rule, "reason": d.reason, "option_id": c.option_id},
        )
        if d.outcome == "reject":
            self._notice("Try again", level="warn", speak=False)
            return
        self.used_nonces.append(c.nonce)
        assert d.option is not None
        if d.outcome == "second":
            self._ask_second(lp, d.option)
            return
        self._accept(lp, d.option)

    async def on_cancel(self, c: UiCancel) -> None:
        mode = self.mode
        if mode == "AWAITING_CONFIRM":
            lp = self.live
            if lp and lp.plan is not None and lp.purpose == "plan_next":
                self.plan = None
            self._drop_proposal()
            self._set_mode("IDLE")
            self._after_idle()
        elif mode in ("LISTENING", "PLANNING"):
            self._cancel_tasks()
            self._drop_proposal()
            self._set_mode("IDLE")
        elif mode == "COUNTDOWN":
            self._abort_before_motion("cancelled")
        elif mode in ("PAUSED", "EXECUTING"):
            self._drop_proposal()
            self._spawn(self._abort_sim(), "abort")
        # IDLE, POSTCHECK, ESTOPPED: nothing to cancel (STOP when idle means back on the phone).

    # ================================================================ planning

    def _profile(self) -> Profile:
        return Profile(
            preferred_name=self.cfg.preferred_name,
            caregiver_name=self.cfg.caregiver_name,
            caregiver_enabled=self.caregiver_enabled,
        )

    def _plan_request(
        self, world: WorldState, text: str, mode: str = "fast", failure: FailureInfo | None = None
    ) -> PlanRequest:
        feas = feasible_skills(world, self.reg, self.pol, in_chain=self.in_chain)
        feasible = []
        for f in feas:
            if f.ok:
                spec = self.reg[f.skill_id]
                feasible.append(
                    FeasibleSkill(
                        skill_id=f.skill_id,
                        kind=spec.kind,
                        label=spec.label,
                        success_alone=f.success_alone,
                        success_in_chain=f.success_in_chain,
                        est_s=f.est_s,
                        risk=spec.risk,
                    )
                )
        infeasible = [
            InfeasibleSkill(skill_id=f.skill_id, rule=f.rule or "", reason=f.reason) for f in feas if not f.ok
        ]
        return PlanRequest(
            mode=mode,
            input=text,
            profile=self._profile(),  # type: ignore[arg-type]
            time=datetime.now().astimezone().isoformat(timespec="seconds"),
            world=world,
            feasible=feasible,
            infeasible=infeasible,
            plan_ids=self.reg.plan_ids(),
            history=list(self.history),
            failure=failure,
        )

    async def _call_planner(self, req: PlanRequest) -> PlannerResult:
        timeout = (self.cfg.fast_timeout_s if req.mode == "fast" else self.cfg.plan_timeout_s) + 0.5
        propose_ex = getattr(self.planner, "propose_ex", None)
        if propose_ex is not None:
            res = await asyncio.wait_for(propose_ex(req), timeout)
        else:  # any object with the 4.2.2 Planner interface
            out = await asyncio.wait_for(self.planner.propose(req), timeout)
            res = PlannerResult(
                output=out,
                model=getattr(self.planner, "model", "planner"),
                source=getattr(self.planner, "source", "nemotron"),
            )
        self.metrics.add("llm_ms", res.latency_ms)
        if res.source == "nemotron":
            self.health.ok("token_factory", mechanism=res.mechanism)
        return res

    async def _plan_flow(self, utt: Utterance) -> None:
        text = utt.text
        if text is None:
            try:
                tr = await self.asr.transcribe(utt.pcm or b"", utt.sr)
                self.health.ok("asr")
            except Exception as exc:
                self.health.error("asr", str(exc))
                self.presenter.trace("asr", None, {"utt_id": utt.utt_id, "error": str(exc)})
                self._notice("Sorry, I didn't catch that", level="warn")
                self._set_mode("IDLE")
                return
            text = tr.text
            self.metrics.add("asr_ms", tr.asr_ms)
            self.out.to_phone(Transcript(utt_id=utt.utt_id, text=text, asr_ms=tr.asr_ms))
            self.presenter.trace("asr", None, {"utt_id": utt.utt_id, "text": text, "asr_ms": tr.asr_ms})
        else:
            self.out.to_phone(Transcript(utt_id=utt.utt_id, text=text, asr_ms=None))
        if not text.strip():
            self._notice("Sorry, I didn't catch that", level="warn")
            self._set_mode("IDLE")
            return
        try:
            world = await self.fresh_world()
        except SimError:
            self._notice("The robot is not connected", level="error")
            self._set_mode("IDLE")
            return
        req = self._plan_request(world, text)
        try:
            res = await self._call_planner(req)
        except (PlannerError, TimeoutError) as exc:
            err = str(exc) or "timeout"
            self.health.error("token_factory", err)
            self.presenter.trace("planner", None, {"input": text, "error": err})
            self._notice("Planner offline, here is the menu", level="warn", speak=False)
            self._offer_menu(world, transcript=text)
            return
        self._offer(res.output, source=res.source, model=res.model, transcript=text, trace=res, req=req)

    def _offer_menu(self, world: WorldState, transcript: str | None) -> None:
        feas = feasible_skills(world, self.reg, self.pol, in_chain=self.in_chain)
        self._offer(deterministic_menu(feas, self.reg), source="menu", model=None, transcript=transcript)

    def _drop_proposal(self) -> None:
        if self.live is not None:
            for t in list(self.tasks):
                if t.get_name().startswith(("precheck", "expire")) and not t.done():
                    t.cancel()
        self.live = None

    def _offer(
        self,
        out: PlannerOutput,
        *,
        source: str,
        model: str | None,
        transcript: str | None,
        trace: PlannerResult | None = None,
        req: PlanRequest | None = None,
        purpose: str = "choose",
        plan: PlanCtx | None = None,
        focus: str | None = None,
    ) -> bool:
        """Validate a planner output and show it. Returns False if nothing could be offered."""
        world = self.world
        if world is None:
            self._notice("The robot is not connected", level="error")
            self._set_mode("IDLE")
            return False
        now = self.clock()
        view = self.snapshot()
        res = validate_proposal(out, world, self.reg, view, self.pol, now)
        pid = new_id("p")
        self.recorder.start(pid, "proposal", source=source, model=model, transcript=transcript)
        if trace is not None:
            self.recorder.file(
                pid,
                "planner_trace.json",
                {
                    "request": req.model_dump(mode="json") if req is not None else trace.request,
                    "response": trace.output.model_dump(mode="json"),
                    "raw": trace.raw,
                    "model": trace.model,
                    "mechanism": trace.mechanism,
                    "latency_ms": trace.latency_ms,
                    "source": trace.source,
                },
            )
            self.presenter.trace(
                "planner",
                pid,
                {
                    "input": transcript,
                    "model": trace.model,
                    "mechanism": trace.mechanism,
                    "latency_ms": round(trace.latency_ms, 1),
                    "output": trace.output.model_dump(mode="json"),
                },
            )
        self.presenter.trace("validator", pid, res.trace())
        if not res.ok:
            self._notice("I can't do that right now", level="warn")
            if self.mode in ("LISTENING", "PLANNING"):
                self._set_mode("IDLE")
            return False
        if not res.options:
            if source != "menu" and purpose == "choose":
                self._notice("I couldn't find a safe option for that. Here is the menu.", level="warn")
                feas = feasible_skills(world, self.reg, self.pol, in_chain=self.in_chain)
                return self._offer(deterministic_menu(feas, self.reg), source="menu", model=None, transcript=transcript)
            self._notice("Nothing to offer right now", level="warn")
            self._set_mode("IDLE")
            return False
        if res.clarify_question:
            self._notice(res.clarify_question, speak=True)
        options = []
        for o in res.options:
            if o.kind != "home" and self.verifier is not None:
                o = o.model_copy(update={"precheck": Precheck(status="pending")})
            options.append(o)
        focus_id = focus if focus in {o.option_id for o in options} else options[0].option_id
        p = Proposal(
            proposal_id=pid,
            expires_at=now + self.pol.proposal_ttl_ms,
            arm_delay_ms=self.pol.arm_delay_ms,
            source=source,
            model=model,
            transcript=transcript,
            options=options,  # type: ignore[arg-type]
            fixed=[FixedOption(option_id="none", label="None of these")],
            focus=focus_id,
            nonce=new_id("n"),
        )
        self._drop_proposal()
        lp = LiveProposal(
            proposal=p, sent_ms=now, purpose=purpose, focus_option=focus_id, focus_since_ms=now, plan=plan
        )
        self.live = lp
        self._set_mode("AWAITING_CONFIRM")
        self.out.to_phone(p)
        self.recorder.update(pid, options=[o.model_dump(mode="json") for o in options])
        for o in options:
            if o.precheck is not None:
                lp.precheck_events[o.option_id] = asyncio.Event()
                self._spawn(self._precheck(lp, o), f"precheck-{o.option_id}")
        self._spawn(self._expire(lp), "expire")
        return True

    async def _expire(self, lp: LiveProposal) -> None:
        await asyncio.sleep(max(0.0, (lp.proposal.expires_at - self.clock()) / 1000))
        if self.live is lp and self.mode in ("AWAITING_CONFIRM", "PAUSED"):
            if lp.purpose == "resume":
                return  # a paused run stays paused until resume or cancel
            self.live = None
            if lp.purpose == "plan_next":
                self.plan = None
            self._notice("The options timed out", speak=False)
            self._set_mode("IDLE")

    def _replace_option(self, lp: LiveProposal, option_id: str, **update: Any) -> Option | None:
        opts = list(lp.proposal.options)
        for i, o in enumerate(opts):
            if o.option_id == option_id:
                opts[i] = o.model_copy(update=update)
                lp.proposal = lp.proposal.model_copy(update={"options": opts})
                return opts[i]
        return None

    async def _capture(self) -> list[Frame]:
        try:
            return await self.sim.capture(list(self.cfg.cosmos_cameras), 640, 480)
        except SimError as exc:
            log.warning("capture failed: %s", exc)
            return []

    async def _cosmos(
        self,
        phase: str,
        spec: SkillSpec,
        question: str,
        *,
        run_id: str | None = None,
        proposal_id: str | None = None,
        option_id: str | None = None,
    ) -> CosmosResult | None:
        if self.verifier is None:
            return None
        frames = await self._capture()
        req = CosmosCheck(
            check_id=new_id("k"),
            run_id=run_id,
            proposal_id=proposal_id,
            option_id=option_id,
            phase=phase,
            skill_id=spec.skill_id,
            question=question,  # type: ignore[arg-type]
            frames=[FrameRef(camera=f.camera, w=f.w, h=f.h, step=f.step or 0) for f in frames],
            model=getattr(self.verifier, "model", self.cfg.cosmos_model),
        )
        self.out.to_presenter(req)
        try:
            res = await asyncio.wait_for(self.verifier.check(req, frames), self.cfg.cosmos_timeout_s)
            self.health.ok("cosmos")
        except Exception as exc:  # Cosmos is advisory: its failure never blocks or moves the arm
            self.health.error("cosmos", f"{type(exc).__name__}: {exc}")
            self.presenter.trace(
                "cosmos",
                run_id or proposal_id,
                {"check_id": req.check_id, "phase": phase, "error": str(exc) or type(exc).__name__},
            )
            return None
        self.metrics.add("cosmos_ms", res.latency_ms)
        return res

    async def _precheck(self, lp: LiveProposal, opt: Option) -> None:
        ev = lp.precheck_events[opt.option_id]
        try:
            spec = self.reg[opt.skill_ids[0]]
            q = spec.cosmos.pre if spec.cosmos else None
            res = None
            if q:
                res = await self._cosmos("pre", spec, q, proposal_id=lp.proposal.proposal_id, option_id=opt.option_id)
            lp.precheck[opt.option_id] = res
            if self.live is not lp:
                return
            world = self.world
            holds_ok = world is not None and all(holds(p, world.atoms) for p in spec.pre)
            decision, unsure = pre_decision(holds_ok, res)
            if res is None:
                status, reason = ("ok", "no camera question") if not q else ("error", "camera check unavailable")
            elif decision == "block":
                status, reason = "block", "preconditions no longer hold"
            else:
                status, reason = ("unsure" if unsure else "ok"), res.reason
            update: dict[str, Any] = {"precheck": Precheck(status=status, reason=reason or None)}  # type: ignore[arg-type]
            cur = next((o for o in lp.proposal.options if o.option_id == opt.option_id), opt)
            if unsure:
                update["badges"] = list(dict.fromkeys([*cur.badges, "unsure_from_camera"]))
                if cur.confirm == "single":
                    update["confirm"] = "double"
            if decision == "block":
                update["locked"] = True
            self._replace_option(lp, opt.option_id, **update)
            if res is not None:
                final = res.model_copy(update={"decision": decision})
                self.out.to_presenter(final)
                self.presenter.trace(
                    "cosmos",
                    lp.proposal.proposal_id,
                    {
                        "check_id": res.check_id,
                        "phase": "pre",
                        "option_id": opt.option_id,
                        "answer": res.answer,
                        "reason": res.reason,
                        "decision": decision,
                        "unsure_from_camera": unsure,
                    },
                )
            if self.live is lp and self.mode in ("AWAITING_CONFIRM",):
                self.out.to_phone(lp.proposal)  # same proposal_id and nonce: the phone updates in place
        finally:
            ev.set()

    # ================================================================ confirm outcomes

    def _ask_second(self, lp: LiveProposal, opt: Option) -> None:
        """First half of a double confirm: fresh nonce, the arm delay and focus dwell start again."""
        now = self.clock()
        lp.pending_double = (lp.proposal.proposal_id, opt.option_id)
        lp.proposal = lp.proposal.model_copy(update={"nonce": new_id("n"), "focus": opt.option_id})
        lp.sent_ms, lp.focus_option, lp.focus_since_ms = now, opt.option_id, now
        self.out.to_phone(lp.proposal)
        self._notice("Blink once more to confirm", speak=True)

    def _accept(self, lp: LiveProposal, opt: Option) -> None:
        self.live = None
        for t in list(self.tasks):
            if t.get_name() == "expire" and not t.done():
                t.cancel()
        pid = lp.proposal.proposal_id
        self.recorder.update(pid, confirmed=opt.option_id)
        if lp.purpose == "resume":
            self._spawn(self._resume(), "resume")
            return
        if opt.kind == "home":
            self._spawn(self._do_home(pid, opt), "home")
            return
        if lp.purpose == "plan_next" and lp.plan is not None:
            plan = lp.plan
        else:
            plan = PlanCtx(proposal_id=pid, option_id=opt.option_id, skill_ids=list(opt.skill_ids))
        self.plan = plan
        self._set_mode("COUNTDOWN")  # leave AWAITING_CONFIRM at once; the run task takes it from here
        self.run_task = self._spawn(self._run_step(plan, user_confirmed=True, card=(lp, opt)), "run")

    # ================================================================ runs

    def _send_run(self, rc: RunCtx, phase: str, **kw: Any) -> None:
        rc.phase = phase
        spec = rc.spec
        max_steps = spec.policy.max_steps if spec.policy else (spec.scripted.max_steps if spec.scripted else 0)
        self.out.to_phone(
            RunUpdate(
                run_id=rc.run_id,
                proposal_id=rc.plan.proposal_id,
                option_id=rc.plan.option_id,
                plan_step=rc.plan.step,
                plan_len=len(rc.plan.skill_ids),
                skill_id=spec.skill_id,
                phase=phase,  # type: ignore[arg-type]
                sim_step=rc.sim_step,
                max_steps=max_steps,
                elapsed_ms=self.clock() - rc.started_ms,
                fault=rc.fault,
                **kw,
            )
        )  # type: ignore[arg-type]

    def _finish_run(self, rc: RunCtx, outcome: str) -> None:
        self.history.append(HistoryItem(skill_id=rc.spec.skill_id, outcome=outcome, at=self.clock()))  # type: ignore[arg-type]
        self.recorder.update(rc.run_id, outcome=outcome, ended_ms=self.clock(), sim_steps=rc.sim_step, fault=rc.fault)
        self.presenter.trace(
            "run",
            rc.run_id,
            {"skill_id": rc.spec.skill_id, "outcome": outcome, "fault": rc.fault, "sim_step": rc.sim_step},
        )
        if self.run is rc:
            self.run = None

    def _abort_before_motion(self, why: str) -> None:
        """Countdown cancelled (cancel, stop or a protective trigger): nothing moved."""
        rc = self.run
        if self.run_task is not None and not self.run_task.done():
            self.run_task.cancel()
        if rc is not None:
            self._send_run(rc, "aborted")
            self._finish_run(rc, "aborted")
        self.plan = None
        self._notice("Cancelled" if why == "cancelled" else "Stopped before moving", speak=True)
        self._set_mode("IDLE")

    async def _run_step(
        self, plan: PlanCtx, user_confirmed: bool, card: tuple[LiveProposal, Option] | None = None
    ) -> None:
        """One motion: spacing, countdown, pre-check, start re-check, run, return home, post-check.

        `card` is the confirmed proposal and option, whose speculative pre-check (5.4) may still
        be running."""
        spec = self.reg[plan.skill_ids[plan.step]]
        run_id = new_id("r")
        rc = RunCtx(run_id=run_id, spec=spec, plan=plan, started_ms=self.clock(), sim_run_id=run_id)
        self.run = rc
        self.recorder.start(
            run_id,
            "run",
            skill_id=spec.skill_id,
            proposal_id=plan.proposal_id,
            option_id=plan.option_id,
            plan_step=plan.step,
            plan_len=len(plan.skill_ids),
        )

        # R11 spacing between motions (user-confirmed steps are exempt).
        wait = spacing_wait_ms(self.last_motion_end_ms, self.clock(), user_confirmed, self.pol)
        if wait:
            self.presenter.trace("validator", run_id, {"rule": "R11_spacing", "wait_ms": wait})
            await asyncio.sleep(wait / 1000)

        # COUNTDOWN 1.0 s, spoken; cancel or stop ends it.
        self._set_mode("COUNTDOWN")
        self._send_run(rc, "countdown")
        self._notice(spec.label, speak=True)
        await asyncio.sleep(self.cfg.countdown_s)

        # Pre-check: usually finished while the card was shown; otherwise wait for it here.
        pre_res: CosmosResult | None = None
        precheck_status: Precheck | None = None
        if card is not None:
            lp, opt = card
            ev = lp.precheck_events.get(opt.option_id)
            if ev is not None and not ev.is_set():
                self._send_run(rc, "precheck")
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(ev.wait(), self.cfg.cosmos_timeout_s + 1.0)
            pre_res = lp.precheck.get(opt.option_id)
            precheck_status = opt.precheck
        rc.precheck = pre_res

        # Re-check the skill against the latest world (the table may have changed since the card).
        try:
            world = await self.fresh_world()
        except SimError:
            await self._fail_start(rc, "The robot is not connected")
            return
        f = skill_rules(spec, world, self.pol, world.atoms, self.in_chain, offer_required=False)
        if not f.ok:
            self.presenter.trace("validator", run_id, {"rule": f.rule, "reason": f.reason, "at": "run start"})
            await self._fail_start(rc, "That is no longer possible: " + f.reason)
            return
        if pre_res is not None:
            _, unsure = pre_decision(True, pre_res)
            already_double = precheck_status is not None and precheck_status.status in ("unsure", "block")
            if unsure and not already_double and user_confirmed:
                # Cosmos was unsure after the user confirmed a single-confirm card: ask again (5.4).
                self._send_run(rc, "aborted")
                self._finish_run(rc, "aborted")
                self._set_mode("IDLE")
                opt = PlannerOption(
                    kind="skill",
                    skill_id=spec.skill_id,
                    plan=None,
                    home_action=None,
                    why="Unsure from camera, confirm twice",
                )
                self._offer_unsure(plan, opt)
                return

        # EXECUTING.
        self._set_mode("EXECUTING")
        self.pause_reason = None
        if spec.near_user:
            self.near_user_used.add(spec.skill_id)
        cmd = self._command(spec, run_id)
        try:
            await self.sim.run(cmd)
            self.health.ok("sim")
        except SimError as exc:
            self.health.error("sim", str(exc))
            await self._fail_start(rc, "The robot did not start")
            return
        self._send_run(rc, "executing")
        term = await self._wait_terminal(rc)
        if term is None or term.phase == "aborted":
            self._end_aborted(rc, "Stopped" if term is not None else "Lost contact with the robot")
            return
        rc.fault = term.fault
        motion_ok = term.phase == "done"

        # Return home between skills (scripted): handover_retreat after a near-user skill.
        if spec.kind == "policy":
            script = "handover_retreat" if spec.near_user else "return_home"
            home_spec = self.reg.get(script)
            if home_spec is not None and home_spec.scripted is not None:
                self._send_run(rc, "returning_home")
                rc.sim_run_id = new_id("r")
                try:
                    await self.sim.run(
                        RunScripted(
                            req_id=new_id("q"),
                            run_id=rc.sim_run_id,
                            epoch=self.epoch,
                            script=home_spec.scripted.script,
                            max_steps=home_spec.scripted.max_steps,
                        )
                    )
                    hterm = await self._wait_terminal(rc, forward=False)
                    if hterm is None or hterm.phase == "aborted":
                        self._end_aborted(rc, "Stopped")
                        return
                except SimError as exc:
                    self.presenter.trace("run", run_id, {"return_home_error": str(exc)})
        self.last_motion_end_ms = self.clock()
        self.in_chain = spec.skill_id != "reset_scene"

        # POSTCHECK: oracle plus Cosmos (5.4).
        self._set_mode("POSTCHECK")
        self._send_run(rc, "postcheck")
        try:
            world = await self.fresh_world()
            oracle = oracle_success(spec, world)
        except SimError:
            oracle = None
        if not motion_ok and oracle is None:
            oracle = False
        cosmos = None
        if spec.cosmos and spec.cosmos.post:
            cosmos = await self._cosmos("post", spec, spec.cosmos.post, run_id=run_id)
        decision = post_decision(oracle, cosmos)
        verdict = None
        if cosmos is not None:
            agree = None if oracle is None else (oracle == (cosmos.answer == "yes"))
            final = cosmos.model_copy(
                update={
                    "decision": decision,
                    "agree": agree,
                    "oracle": OracleAtom(atom=spec.goal_atom, value=bool(oracle))
                    if spec.goal_atom and oracle is not None
                    else None,
                }
            )
            self.out.to_presenter(final)
            self.presenter.trace(
                "cosmos",
                run_id,
                {
                    "check_id": cosmos.check_id,
                    "phase": "post",
                    "answer": cosmos.answer,
                    "reason": cosmos.reason,
                    "oracle": oracle,
                    "agree": agree,
                    "decision": decision,
                },
            )
            verdict = CosmosVerdict(
                check_id=cosmos.check_id, phase="post", answer=cosmos.answer, reason=cosmos.reason, decision=decision
            )
        self._send_run(rc, decision, cosmos=verdict, oracle=oracle)
        self._finish_run(rc, decision)
        if spec.skill_id == "reset_scene" and decision == "done":
            self.in_chain = False
        self._after_run(plan, spec, decision, rc, cosmos)

    def _command(self, spec: SkillSpec, run_id: str) -> RunSkill | RunScripted:
        if spec.kind == "policy" and spec.policy is not None and spec.goal_ref:
            return RunSkill(
                req_id=new_id("q"),
                run_id=run_id,
                epoch=self.epoch,
                skill_id=spec.skill_id,
                instruction=spec.policy.instruction,
                goal_ref=spec.goal_ref,
                n_action_steps=spec.policy.n_action_steps,
                max_steps=spec.policy.max_steps,
                action_scale=self.pol.near_user_scale if spec.near_user else 1.0,
            )  # R14
        if spec.kind == "scripted" and spec.scripted is not None:
            return RunScripted(
                req_id=new_id("q"),
                run_id=run_id,
                epoch=self.epoch,
                script=spec.scripted.script,
                max_steps=spec.scripted.max_steps,
            )
        raise ValueError(f"{spec.skill_id} is not a motion")

    async def _fail_start(self, rc: RunCtx, text: str) -> None:
        self._send_run(rc, "failed")
        self._finish_run(rc, "failed")
        self.plan = None
        self._notice(text, level="warn")
        self._set_mode("IDLE")
        self._after_idle()

    def _end_aborted(self, rc: RunCtx, text: str) -> None:
        self._send_run(rc, "aborted")
        self._finish_run(rc, "aborted")
        self.last_motion_end_ms = self.clock()
        self.plan = None
        self._notice(text, level="warn")
        self._set_mode("IDLE")
        self._after_idle()

    async def _wait_terminal(self, rc: RunCtx, forward: bool = True) -> SimStatus | None:
        """Consume sim.status for the motion in flight until done/failed/aborted.

        Handles pauses from the sim (e.g. its own hb watchdog). Returns None if the sim goes
        silent for `sim_status_timeout_s` (it is then sent ctl.abort).
        """
        while True:
            try:
                s = await asyncio.wait_for(rc.queue.get(), self.cfg.sim_status_timeout_s)
            except TimeoutError:
                self.health.error("sim", "no status for the run in flight")
                with contextlib.suppress(SimError):
                    await self.sim.abort(self.epoch)
                return None
            if s.run_id != rc.sim_run_id:
                continue
            rc.sim_step = s.step if forward else rc.sim_step
            if s.policy_ms is not None:
                self.metrics.add("policy_ms", s.policy_ms)
                self.health.ok("policy")
            if s.fault == "policy_timeout":
                self.health.error("policy", "policy_timeout")
            if s.phase in ("done", "failed", "aborted"):
                if self.live is not None and self.live.purpose == "resume":
                    self._drop_proposal()  # the motion ended while paused (e.g. a pause raced the finish)
                return s
            if s.phase == "paused":
                if self.mode != "PAUSED":
                    self._enter_paused(s.fault or "paused", send_pause=False)
                continue
            if s.phase == "executing" and forward:
                now = self.clock()
                if now - rc.last_forward_ms >= 200 and self.mode == "EXECUTING":
                    rc.last_forward_ms = now
                    self._send_run(rc, "executing")

    def _after_run(
        self, plan: PlanCtx, spec: SkillSpec, decision: str, rc: RunCtx, cosmos: CosmosResult | None
    ) -> None:
        if decision == "done" and plan.step + 1 < len(plan.skill_ids):
            plan.step += 1
            self._set_mode("IDLE")
            if self.cfg.plan_mode == "whole":
                self.run_task = self._spawn(self._whole_plan_next(plan), "run")
            else:
                self._offer_plan_next(plan)
            return
        self.plan = None
        self._set_mode("IDLE")
        if decision == "done":
            self._notice("Done", speak=True)
            self._after_idle()
            return
        # failed or uncertain: recovery options (Nemotron Super recovery prompt, or a fallback).
        self._notice("That didn't work" if decision == "failed" else "I'm not sure it worked", level="warn")
        failure = FailureInfo(
            skill_id=spec.skill_id,
            steps=rc.sim_step,
            cosmos_reason=cosmos.reason if cosmos else None,
            oracle=None if decision == "uncertain" else False,
        )
        self._spawn(self._recovery(failure), "plan-recovery")

    def _offer_plan_next(self, plan: PlanCtx) -> None:
        sid = plan.skill_ids[plan.step]
        opt = PlannerOption(
            kind="skill",
            skill_id=sid,
            plan=None,
            home_action=None,
            why=f"Step {plan.step + 1} of {len(plan.skill_ids)}",
        )
        out = PlannerOutput(intent="manipulate", clarify_question=None, options=[opt])
        if not self._offer(out, source="menu", model=None, transcript=None, purpose="plan_next", plan=plan):
            self.plan = None

    def _offer_unsure(self, plan: PlanCtx, opt: PlannerOption) -> None:
        out = PlannerOutput(intent="manipulate", clarify_question=None, options=[opt])
        ok = self._offer(
            out,
            source="menu",
            model=None,
            transcript=None,
            purpose="plan_next" if len(plan.skill_ids) > 1 else "choose",
            plan=plan if len(plan.skill_ids) > 1 else None,
        )
        if ok and self.live is not None:
            # Force a double confirm and the camera badge on this re-offer (5.4).
            o = self.live.proposal.options[0]
            self._replace_option(
                self.live,
                o.option_id,
                confirm="double",
                badges=list(dict.fromkeys([*o.badges, "unsure_from_camera"])),
                precheck=Precheck(status="unsure", reason="camera unsure"),
            )
            self.out.to_phone(self.live.proposal)

    async def _whole_plan_next(self, plan: PlanCtx) -> None:
        sid = plan.skill_ids[plan.step]
        self._notice(f"Next: {self.reg[sid].label}. Say stop to cancel.", speak=True)
        self._set_mode("COUNTDOWN")
        await asyncio.sleep(self.cfg.plan_countdown_s)
        await self._run_step(plan, user_confirmed=False)

    async def _recovery(self, failure: FailureInfo) -> None:
        try:
            world = await self.fresh_world()
        except SimError:
            return
        req = self._plan_request(world, f"recover from {failure.skill_id}", mode="recovery", failure=failure)
        try:
            res = await self._call_planner(req)
            out, source, model, trace = res.output, res.source, res.model, res
        except (PlannerError, TimeoutError) as exc:
            self.health.error("token_factory", str(exc) or "timeout")
            out = PlannerOutput(
                intent="manipulate",
                clarify_question=None,
                options=[
                    PlannerOption(
                        kind="skill", skill_id=failure.skill_id, plan=None, home_action=None, why="Try again?"
                    ),
                    PlannerOption(
                        kind="home",
                        skill_id=None,
                        plan=None,
                        home_action="call_caregiver",
                        why="Or I can call for help",
                    ),
                ],
            )
            source, model, trace = "menu", None, None
        if self.mode == "IDLE":
            self._offer(out, source=source, model=model, transcript=None, trace=trace, req=req)

    async def _abort_sim(self) -> None:
        try:
            await self.sim.abort(self.epoch)
        except SimError as exc:
            self.health.error("sim", str(exc))
            self._notice("Could not reach the robot to cancel. Use STOP.", level="error")

    async def _do_home(self, pid: str, opt: Option) -> None:
        action = opt.home_action or ""
        spec = self.reg.home_skill(action)
        run_id = new_id("r")
        plan = PlanCtx(proposal_id=pid, option_id=opt.option_id, skill_ids=[spec.skill_id if spec else action])
        self._set_mode("IDLE")
        self.recorder.start(run_id, "home", action=action, proposal_id=pid)
        try:
            res = await self.home.call(action)
        except Exception as exc:
            res = None
            self.health.error("hass", str(exc))
        ok = bool(res and res.ok)
        if spec is not None:
            rc = RunCtx(run_id=run_id, spec=spec, plan=plan, started_ms=self.clock(), sim_run_id=run_id)
            self._send_run(rc, "done" if ok else "failed")
        if ok:
            self.health.ok("hass")
        elif res is not None:
            self.health.error("hass", res.detail or "failed")
        self.presenter.trace("home", run_id, {"action": action, "ok": ok, "detail": res.detail if res else None})
        self.recorder.update(run_id, outcome="done" if ok else "failed")
        self._notice(
            (res.detail if res and res.detail else action) if ok else "That didn't work", level="info" if ok else "warn"
        )
        self._after_idle()

    # ================================================================ pause / resume

    def protective_pause(self, reason: str) -> None:
        """Watchdog trigger (5.6): heartbeat loss or face lost while moving. Synchronous."""
        if self.mode == "COUNTDOWN":
            self._abort_before_motion(reason)
            return
        if self.mode != "EXECUTING":
            return
        self._enter_paused(reason, send_pause=True)

    def _enter_paused(self, reason: str, send_pause: bool) -> None:
        self.pause_reason = reason
        self._set_mode("PAUSED")
        if self.run is not None:
            self._send_run(self.run, "paused")
        if send_pause:
            self._spawn(self._pause_sim(), "pause")
        self.presenter.trace("run", self.run.run_id if self.run else None, {"paused": reason})
        self._offer_resume(reason)

    async def _pause_sim(self) -> None:
        try:
            await self.sim.pause(self.epoch)
        except SimError as exc:  # the sim's own hb watchdog still stops it within 500 ms
            self.health.error("sim", str(exc))

    def _offer_resume(self, reason: str) -> None:
        rc = self.run
        if rc is None:
            return
        words = {"face_lost": "I can't see you", "hb_lost": "Lost contact with your phone"}.get(reason, "Paused")
        now = self.clock()
        opt = Option(
            option_id="resume",
            kind="skill",
            skill_ids=[rc.spec.skill_id],
            label=rc.spec.label,
            why=f"{words}. Resume?",
            success=None,
            est_s=None,
            risk=rc.spec.risk,
            confirm="single",
            checks=[],
            precheck=None,
        )
        p = Proposal(
            proposal_id=new_id("p"),
            expires_at=now + 24 * 3600 * 1000,
            arm_delay_ms=self.pol.arm_delay_ms,
            source="menu",
            model=None,
            transcript=None,
            options=[opt],
            fixed=[FixedOption(option_id="none", label="Cancel")],
            focus="resume",
            nonce=new_id("n"),
        )
        self._drop_proposal()
        self.live = LiveProposal(proposal=p, sent_ms=now, purpose="resume", focus_option="resume", focus_since_ms=now)
        self.out.to_phone(p)
        self._notice(f"Paused. {words}.", level="warn", speak=True)

    async def _resume(self) -> None:
        if self.mode != "PAUSED":
            return
        try:
            await self.sim.resume(self.epoch)
        except SimError as exc:
            self.health.error("sim", str(exc))
            self._notice("Could not resume", level="error")
            self._offer_resume(self.pause_reason or "paused")
            return
        self.hb.face_absent_since_ms = None if self.hb.face_present else self.hb.face_absent_since_ms
        self.pause_reason = None
        self._set_mode("EXECUTING")
        if self.run is not None:
            self._send_run(self.run, "executing")

    # ================================================================ e-stop (called by SafetyChannel)

    def latch_estop(self, reason: str) -> None:
        """Synchronous. ESTOPPED; cancel every task; invalidate the proposal; bump the epoch."""
        self.epoch += 1
        self._cancel_tasks()
        self.live = None
        self.audio.clear()
        self.prefetched = None
        rc = self.run
        if rc is not None:
            self._send_run(rc, "aborted")
            self._finish_run(rc, "aborted")
            self.last_motion_end_ms = self.clock()
        self.run = None
        self.plan = None
        self.run_task = None
        self._set_mode("ESTOPPED")

    def on_estop_seq(self, seq: int) -> None:
        self.client_estop_seq = max(self.client_estop_seq, seq)

    def on_sim_epoch(self, epoch: int) -> None:
        self.epoch = max(self.epoch, epoch)

    def reset_estop(self, msg: EStopReset) -> bool:
        """`estop.reset`: back to IDLE. The interrupted run stays aborted, never resumed."""
        if self.mode != "ESTOPPED":
            self.out.to_phone(self.state_msg())
            return False
        self.plan = None
        self.run = None
        self.live = None
        self._set_mode("IDLE")
        self._notice("Stop cleared", speak=True)
        self._after_idle()
        return True

    # ================================================================ sim status

    async def on_sim_status(self, s: SimStatus) -> None:
        rc = self.run
        if rc is None or s.epoch < self.epoch:
            return  # stale (e.g. from before an e-stop)
        if s.run_id == rc.sim_run_id:
            rc.queue.put_nowait(s)

    # ================================================================ idle work

    def _after_idle(self) -> None:
        """Background prefetch of suggestions for the next brow raise (7.1 fast path)."""
        if self.mode != "IDLE" or self.world is None:
            return
        self._spawn(self._prefetch(), "prefetch")

    async def _prefetch(self) -> None:
        try:
            world = await self.fresh_world()
            req = self._plan_request(world, MENU_NO_SPEECH)
            res = await self._call_planner(req)
        except (PlannerError, SimError, TimeoutError):
            return
        self.prefetched = (frozenset(world.atoms.items()), res)

    # ================================================================ dev / caregiver

    def set_caregiver(self, enabled: bool) -> None:
        self.caregiver_enabled = enabled
        self.presenter.trace("validator", None, {"caregiver_enabled": enabled})

    def start_reset_scene(self) -> bool:
        if self.mode != "IDLE" or "reset_scene" not in self.reg:
            return False
        plan = PlanCtx(proposal_id=new_id("p"), option_id="reset", skill_ids=["reset_scene"])
        self.plan = plan
        self.run_task = self._spawn(self._run_step(plan, user_confirmed=True), "run")
        return True
