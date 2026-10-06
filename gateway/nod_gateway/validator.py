"""The deterministic validator (design 5.2). PURE: no I/O, no clock reads, no randomness.

The LLM proposes, this module decides, the user confirms. Every function takes all the
state it needs as arguments (`now_ms` is passed in), so each rule is unit-testable.

Rules (5.2):
  R1 schema            planner output parses into PlannerOutput; inconsistent options dropped
  R2 known skill       exists, enabled, offerable, matches the scene
  R3 objects           every object the skill references is in WorldState.objects
  R4 preconditions     every `pre` literal holds in the oracle world
  R5 not a no-op       not all effects already true (skills with no effects are exempt)
  R6 reliability       n >= 20 and wilson_lo >= 0.60 normal; 0.30-0.60 badge + double confirm;
                       hidden below 0.30 or n < 10 (policy skills only; scripted and home exempt)
  R7 risk              risk high needs caregiver_enabled plus a double confirm, else locked
  R8 mode              new motion only from IDLE or AWAITING_CONFIRM (and the planning modes
                       that lead there); never ESTOPPED, PAUSED, EXECUTING, COUNTDOWN, POSTCHECK
  R9 confirm binding   ids, nonce (single use), arm delay, 300 ms focus dwell, expiry, lease, face
  R10 plan check       symbolic rollout; estimate is the product of in-chain rates
  R11 spacing          2 s between motions unless the user confirmed that step
  R12 count            1-3 options, no duplicates
  R13 labels           registry label; LLM text only as `why`, <= 80 chars, plain text
  R14 near user        speed scale 0.6, double confirm on first use per session
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Literal

from nod_protocol import (
    Check,
    CosmosResult,
    Option,
    PlannerOption,
    PlannerOutput,
    Proposal,
    SessionMode,
    SkillSpec,
    StatBlock,
    Success,
    UiConfirm,
    WorldState,
    apply_effects,
    holds,
    plan_rollout,
    unmet,
)

from .registry import Registry

R1 = "R1_schema"
R2 = "R2_known_skill"
R3 = "R3_objects"
R4 = "R4_preconditions"
R5 = "R5_not_noop"
R6 = "R6_reliability"
R7 = "R7_risk"
R8 = "R8_mode"
R9 = "R9_confirm_binding"
R10 = "R10_plan_check"
R11 = "R11_spacing"
R12 = "R12_count"
R13 = "R13_labels"
R14 = "R14_near_user"

# Modes from which a new proposal may be made / motion may start (R8).
PROPOSAL_MODES: frozenset[str] = frozenset({"IDLE", "LISTENING", "PLANNING", "AWAITING_CONFIRM"})
NO_MOTION_MODES: frozenset[str] = frozenset({"ESTOPPED", "PAUSED", "EXECUTING", "COUNTDOWN", "POSTCHECK"})

Lease = Literal["user", "caregiver", "none"]
Purpose = Literal["choose", "resume", "plan_next"]


@dataclass(frozen=True)
class SafetyPolicy:
    """Thresholds for the rules. Defaults are the design values; see `from_config`."""

    scene_id: str = "libero_goal_v1"
    min_n_ok: int = 20
    ok_wilson: float = 0.60
    low_wilson: float = 0.30
    hide_n: int = 10
    arm_delay_ms: int = 300
    focus_dwell_ms: int = 300
    proposal_ttl_ms: int = 30_000
    min_confirm_hold_ms: int = 600
    spacing_ms: int = 2000
    near_user_scale: float = 0.6
    why_max: int = 80
    max_options: int = 3

    @classmethod
    def from_config(cls, cfg) -> SafetyPolicy:
        return cls(
            scene_id=cfg.scene_id,
            arm_delay_ms=cfg.arm_delay_ms,
            focus_dwell_ms=cfg.focus_dwell_ms,
            proposal_ttl_ms=cfg.proposal_ttl_s * 1000,
            min_confirm_hold_ms=cfg.min_confirm_hold_ms,
            spacing_ms=cfg.spacing_ms,
            near_user_scale=cfg.near_user_scale,
        )


DEFAULT_POLICY = SafetyPolicy()


@dataclass(frozen=True)
class SessionView:
    """Immutable snapshot of the session handed to the validator (`Session.snapshot()`).

    `lease` is the lease of the connection the message under validation came from.
    Gateway-clock times are in ms.
    """

    mode: SessionMode = "IDLE"
    lease: Lease = "user"
    face_present: bool = False
    hb_fresh: bool = False
    caregiver_enabled: bool = False
    in_chain: bool = False  # a motion ran since the last reset: use in-chain stats
    proposal_id: str | None = None
    proposal_sent_ms: int | None = None
    proposal_purpose: Purpose = "choose"
    focus_option: str | None = None
    focus_since_ms: int | None = None
    used_nonces: frozenset[str] = frozenset()
    near_user_used: frozenset[str] = frozenset()
    pending_double: tuple[str, str] | None = None  # (proposal_id, option_id) awaiting its 2nd confirm
    last_motion_end_ms: int | None = None

    def with_lease(self, lease: Lease) -> SessionView:
        return dataclasses.replace(self, lease=lease)


# ---------------------------------------------------------------- results


@dataclass(frozen=True)
class Feasibility:
    """Skill-level outcome of R2-R6 against the current world (session-independent)."""

    skill_id: str
    kind: str
    ok: bool
    rule: str | None = None  # first failing rule
    reason: str = ""
    risk: str = "low"
    success: Success | None = None  # on the basis in use (alone or in_chain)
    success_alone: Success | None = None
    success_in_chain: Success | None = None
    badges: tuple[str, ...] = ()
    confirm: Literal["single", "double"] = "single"
    checks: tuple[Check, ...] = ()
    est_s: float | None = None


@dataclass(frozen=True)
class PlanCheck:
    ok: bool
    failed_step: int | None = None
    failed_skill: str | None = None
    unmet: tuple[str, ...] = ()
    rule: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class Dropped:
    index: int
    rule: str
    reason: str
    skill_ids: tuple[str, ...] = ()


@dataclass
class ValidationResult:
    ok: bool  # False: the whole output is rejected (R1 or R8); the caller shows the menu or nothing
    options: list[Option] = field(default_factory=list)
    dropped: list[Dropped] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    rule: str | None = None
    reason: str = ""
    intent: str | None = None
    clarify_question: str | None = None

    def trace(self) -> dict:
        return {
            "ok": self.ok,
            "rule": self.rule,
            "reason": self.reason,
            "intent": self.intent,
            "options": [o.model_dump(mode="json") for o in self.options],
            "dropped": [dataclasses.asdict(d) for d in self.dropped],
            "checks": [c.model_dump(mode="json") for c in self.checks],
        }


@dataclass(frozen=True)
class ConfirmDecision:
    outcome: Literal["accept", "second", "reject"]
    rule: str | None = None
    reason: str = ""
    option: Option | None = None

    @property
    def ok(self) -> bool:
        return self.outcome != "reject"


# ---------------------------------------------------------------- helpers

_CTRL = re.compile(r"[\x00-\x1f\x7f<>`*_#\[\]{}|\\]")
_WS = re.compile(r"\s+")


def sanitize_why(text: str | None, max_len: int = 80) -> str:
    """R13: plain text, single line, no markup characters, at most `max_len` characters."""
    if not text:
        return ""
    t = _WS.sub(" ", _CTRL.sub(" ", text)).strip()
    if len(t) > max_len:
        t = t[: max_len - 1].rstrip() + "…"
    return t


def _success_from(block: StatBlock | None, basis: Literal["alone", "in_chain"]) -> Success | None:
    if block is None:
        return None
    p = block.k / block.n if block.n else 0.0
    return Success(p_est=round(p, 4), basis=basis, n=block.n, wilson_lo=block.wilson_lo, k=block.k)


def _stats_for(spec: SkillSpec, in_chain: bool) -> tuple[Success | None, Success | None, Success | None]:
    alone = _success_from(spec.stats.alone, "alone")
    chain = _success_from(spec.stats.in_chain, "in_chain")
    # In-chain numbers are used once the table is no longer in its initial state; fall back to
    # the alone numbers when no in-chain eval exists yet (e.g. a new custom skill).
    use = (chain or alone) if in_chain else alone
    return use, alone, chain


def _est_s(spec: SkillSpec, basis: str) -> float | None:
    block = spec.stats.in_chain if basis == "in_chain" else spec.stats.alone
    if block is not None and block.median_s:
        return block.median_s
    return spec.est_s


def reliability(success: Success | None, pol: SafetyPolicy) -> Literal["ok", "low", "hidden"]:
    """R6 band for a measured success block."""
    if success is None or success.n < pol.hide_n or success.wilson_lo < pol.low_wilson:
        return "hidden"
    if success.n >= pol.min_n_ok and success.wilson_lo >= pol.ok_wilson:
        return "ok"
    return "low"


def skill_rules(
    spec: SkillSpec,
    world: WorldState,
    pol: SafetyPolicy,
    atoms: dict[str, bool],
    in_chain: bool,
    check_pre: bool = True,
    offer_required: bool = True,
) -> Feasibility:
    """Evaluate R2-R6 for one skill against `atoms` (the world, or a rolled-out plan state).

    `offer_required=False` skips the "offerable" part of R2: used when re-checking an
    accepted step at run start (a plan step or a caregiver reset is not a card).
    """
    checks: list[Check] = []
    badges: list[str] = []
    confirm: Literal["single", "double"] = spec.confirm
    success, s_alone, s_chain = _stats_for(spec, in_chain)

    def fail(rule: str, reason: str) -> Feasibility:
        checks.append(Check(rule=rule, ok=False, detail=reason))
        return Feasibility(
            spec.skill_id,
            spec.kind,
            False,
            rule,
            reason,
            spec.risk,
            success,
            s_alone,
            s_chain,
            tuple(badges),
            confirm,
            tuple(checks),
            _est_s(spec, success.basis if success else "alone"),
        )

    # R2 known skill: enabled, offerable, right scene.
    if not spec.enabled:
        return fail(R2, "skill is disabled")
    if offer_required and not spec.offer:
        return fail(R2, "internal motion, never offered")
    if spec.kind != "home" and spec.scene_id != world.scene_id:
        return fail(R2, f"skill is for scene {spec.scene_id}, world is {world.scene_id}")
    checks.append(Check(rule=R2, ok=True))

    # R3 objects present.
    missing_obj = [o for o in spec.objects if o not in world.objects]
    if missing_obj:
        return fail(R3, f"objects not in the world: {', '.join(missing_obj)}")
    checks.append(Check(rule=R3, ok=True))

    # R4 preconditions (unknown atoms never satisfy a literal).
    if check_pre:
        missing = unmet(spec.pre, atoms)
        if missing:
            return fail(R4, "precondition false: " + "; ".join(missing))
        checks.append(Check(rule=R4, ok=True))

    # R5 not a no-op (skills with no effects, such as return_home, are exempt).
    if spec.effects and all(holds(e, atoms) for e in spec.effects):
        return fail(R5, "all effects already true")
    checks.append(Check(rule=R5, ok=True))

    # R6 reliability: learned policy skills only. Scripted motions are deterministic; home
    # actions do not move the arm.
    if spec.kind == "policy":
        band = reliability(success, pol)
        if band == "hidden":
            detail = "no stats" if success is None else f"n={success.n}, wilson_lo={success.wilson_lo:.2f}"
            return fail(R6, f"hidden: {detail}")
        if band == "low":
            badges.append("less_reliable")
            confirm = "double"
            checks.append(Check(rule=R6, ok=True, detail="less reliable: double confirm"))
        else:
            checks.append(Check(rule=R6, ok=True))
    elif spec.kind == "scripted":
        badges.append("scripted")

    if spec.near_user:
        badges.append("near_user")
    return Feasibility(
        spec.skill_id,
        spec.kind,
        True,
        None,
        "",
        spec.risk,
        success,
        s_alone,
        s_chain,
        tuple(badges),
        confirm,
        tuple(checks),
        _est_s(spec, success.basis if success else "alone"),
    )


# ---------------------------------------------------------------- public API


def feasible_skills(
    world: WorldState, reg: Registry, pol: SafetyPolicy = DEFAULT_POLICY, *, in_chain: bool = False
) -> list[Feasibility]:
    """R2-R6 for every offerable skill, in registry order. Feeds the planner enum (7.4),
    the `infeasible[]` context (7.3) and the deterministic menu."""
    out = []
    for spec in reg:
        if not spec.offer:
            continue
        out.append(skill_rules(spec, world, pol, world.atoms, in_chain))
    return out


def check_plan(skill_ids: list[str], world: WorldState, reg: Registry) -> PlanCheck:
    """R10: STRIPS-style rollout; each step's `pre` must hold in world + earlier effects."""
    if not skill_ids:
        return PlanCheck(ok=False, rule=R10, reason="empty plan")
    r = plan_rollout(list(skill_ids), dict(world.atoms), list(reg))
    if r.ok:
        return PlanCheck(ok=True)
    rule = R2 if r.unmet == ["unknown skill"] else (R4 if r.failed_step == 0 else R10)
    return PlanCheck(
        ok=False,
        failed_step=r.failed_step,
        failed_skill=r.failed_skill,
        unmet=tuple(r.unmet),
        rule=rule,
        reason=f"step {r.failed_step} ({r.failed_skill}): " + "; ".join(r.unmet),
    )


def _session_rules(
    spec_list: list[SkillSpec], sess: SessionView, badges: list[str], checks: list[Check], confirm: str
) -> tuple[str, bool]:
    """R7 and R14 for a card whose motions are `spec_list`. Returns (confirm, locked)."""
    locked = False
    if any(s.risk == "high" for s in spec_list):
        confirm = "double" if confirm != "per_step" else confirm
        if not sess.caregiver_enabled:
            locked = True
            badges.append("needs_caregiver")
            checks.append(Check(rule=R7, ok=False, detail="needs caregiver"))
        else:
            checks.append(Check(rule=R7, ok=True, detail="caregiver enabled: double confirm"))
    near = [s for s in spec_list if s.near_user]
    if near:
        first = [s.skill_id for s in near if s.skill_id not in sess.near_user_used]
        if first:
            badges.append("first_use")
            if confirm != "per_step":
                confirm = "double"
        checks.append(
            Check(rule=R14, ok=True, detail="speed scale 0.6" + ("; first use: double confirm" if first else ""))
        )
    return confirm, locked


def _dedupe(badges: list[str]) -> list[str]:
    return list(dict.fromkeys(badges))


def _plan_option(
    opt: PlannerOption, world: WorldState, reg: Registry, sess: SessionView, pol: SafetyPolicy
) -> Option | Dropped:
    ids = list(opt.plan or [])
    allowed = set(reg.plan_ids())
    for sid in ids:
        if sid not in reg:
            return Dropped(-1, R2, f"unknown skill {sid!r}", tuple(ids))
        if sid not in allowed:
            return Dropped(-1, R2, f"{sid!r} is not allowed in a plan", tuple(ids))
    pc = check_plan(ids, world, reg)
    if not pc.ok:
        return Dropped(-1, pc.rule or R10, pc.reason, tuple(ids))
    # Per-step R2, R3, R5, R6 against the rolled-out state (R4 is covered by the rollout).
    atoms = dict(world.atoms)
    specs: list[SkillSpec] = []
    badges: list[str] = []
    steps: list[Success] = []
    est = 0.0
    for sid in ids:
        spec = reg[sid]
        f = skill_rules(spec, world, pol, atoms, in_chain=True, check_pre=False)
        if not f.ok:
            return Dropped(-1, f.rule or R10, f"step {sid}: {f.reason}", tuple(ids))
        badges.extend(b for b in f.badges if b in ("less_reliable", "near_user"))
        if f.success is not None:
            steps.append(f.success)
        est += f.est_s or 0.0
        specs.append(spec)
        atoms = apply_effects(atoms, spec.effects)
    chain = reg.chain_for(ids)
    weakest = min(steps, key=lambda s: s.wilson_lo) if steps else None
    if chain is not None and chain.n:
        success = Success(
            p_est=round(chain.k / chain.n, 4), basis="in_chain", n=chain.n, wilson_lo=chain.wilson_lo, k=chain.k
        )
    elif weakest is not None:
        p = 1.0
        for s in steps:
            p *= s.p_est
        success = Success(p_est=round(p, 4), basis="in_chain", n=weakest.n, wilson_lo=weakest.wilson_lo, k=weakest.k)
    else:
        success = None
    # R6 on the whole plan: the measured chain (or the weakest step) sets the band.
    band = reliability(success, pol) if success is not None else "ok"
    if band == "hidden" and success is not None:
        return Dropped(-1, R6, f"plan hidden: n={success.n}, wilson_lo={success.wilson_lo:.2f}", tuple(ids))
    if band == "low":
        badges.append("less_reliable")
    checks = [Check(rule=R2, ok=True), Check(rule=R4, ok=True), Check(rule=R10, ok=True)]
    confirm, locked = _session_rules(specs, sess, badges, checks, "per_step")
    return Option(
        option_id="",
        kind="plan",
        skill_ids=ids,
        label=_plan_label(ids, reg),
        why=sanitize_why(opt.why, pol.why_max),
        success=success,
        est_s=round(est, 1) if est else None,
        risk=_max_risk(specs),
        confirm=confirm,
        checks=checks,
        badges=_dedupe(badges),
        locked=locked,
    )


def _plan_label(ids: list[str], reg: Registry) -> str:
    """R13: a plan's label is built only from registry labels."""
    return f"{reg[ids[0]].label}, then {len(ids) - 1} more step{'s' if len(ids) > 2 else ''}"


def _max_risk(specs: list[SkillSpec]) -> str:
    order = {"none": 0, "low": 1, "high": 2}
    return max((s.risk for s in specs), key=lambda r: order[r], default="none")


def _skill_option(
    opt: PlannerOption, world: WorldState, reg: Registry, sess: SessionView, pol: SafetyPolicy
) -> Option | Dropped:
    spec = reg.get(opt.skill_id)
    if spec is None:
        return Dropped(-1, R2, f"unknown skill {opt.skill_id!r}", (str(opt.skill_id),))
    if spec.kind == "home":
        return Dropped(-1, R2, f"{spec.skill_id!r} is a home action, not a skill", (spec.skill_id,))
    f = skill_rules(spec, world, pol, world.atoms, sess.in_chain)
    if not f.ok:
        return Dropped(-1, f.rule or R2, f.reason, (spec.skill_id,))
    badges = list(f.badges)
    checks = list(f.checks)
    confirm, locked = _session_rules([spec], sess, badges, checks, f.confirm)
    return Option(
        option_id="",
        kind="skill",
        skill_ids=[spec.skill_id],
        label=spec.label,
        why=sanitize_why(opt.why, pol.why_max),
        success=f.success,
        est_s=f.est_s,
        risk=spec.risk,
        confirm=confirm,
        checks=checks,
        badges=_dedupe(badges),
        locked=locked,
    )


def _home_option(opt: PlannerOption, reg: Registry, pol: SafetyPolicy) -> Option | Dropped:
    spec = reg.home_skill(opt.home_action or "")
    if spec is None or not spec.enabled:
        return Dropped(-1, R2, f"home action {opt.home_action!r} not available", (str(opt.home_action),))
    return Option(
        option_id="",
        kind="home",
        skill_ids=[],
        home_action=spec.home.action if spec.home else None,
        label=spec.label,
        why=sanitize_why(opt.why, pol.why_max),
        success=None,
        est_s=spec.est_s,
        risk=spec.risk,
        confirm=spec.confirm,
        checks=[Check(rule=R2, ok=True)],
        precheck=None,
    )


def _option_key(o: Option) -> tuple:
    return (o.kind, tuple(o.skill_ids), o.home_action)


def validate_proposal(
    out: PlannerOutput,
    world: WorldState,
    reg: Registry,
    sess: SessionView,
    pol: SafetyPolicy = DEFAULT_POLICY,
    now_ms: int = 0,
) -> ValidationResult:
    """Check a planner output (already parsed: R1) against every proposal rule.

    Returns the options to show (ids o1..o3, canonical labels, badges, confirm policy),
    plus what was dropped and why. `now_ms` is accepted for interface symmetry; no
    proposal rule depends on time (R11 is applied when a motion starts, `spacing_wait_ms`).
    """
    del now_ms
    res = ValidationResult(ok=True, intent=out.intent, clarify_question=out.clarify_question)
    res.checks.append(Check(rule=R1, ok=True))
    if sess.mode not in PROPOSAL_MODES:
        res.ok = False
        res.rule, res.reason = R8, f"no new motion in mode {sess.mode}"
        res.checks.append(Check(rule=R8, ok=False, detail=res.reason))
        return res
    res.checks.append(Check(rule=R8, ok=True))

    kept: list[Option] = []
    seen: set[tuple] = set()
    for i, opt in enumerate(out.options):
        if not opt.is_consistent():
            res.dropped.append(Dropped(i, R1, f"fields do not match kind {opt.kind!r}"))
            continue
        if opt.kind == "home":
            r = _home_option(opt, reg, pol)
        elif opt.kind == "skill":
            r = _skill_option(opt, world, reg, sess, pol)
        else:
            r = _plan_option(opt, world, reg, sess, pol)
        if isinstance(r, Dropped):
            res.dropped.append(dataclasses.replace(r, index=i))
            continue
        key = _option_key(r)
        if key in seen:
            res.dropped.append(Dropped(i, R12, "duplicate option", tuple(r.skill_ids)))
            continue
        if len(kept) >= pol.max_options:
            res.dropped.append(Dropped(i, R12, "more than 3 options", tuple(r.skill_ids)))
            continue
        seen.add(key)
        kept.append(r)

    res.options = [o.model_copy(update={"option_id": f"o{n}"}) for n, o in enumerate(kept, start=1)]
    res.checks.append(Check(rule=R12, ok=bool(res.options), detail=f"{len(res.options)} option(s)"))
    return res


def validate_confirm(
    c: UiConfirm, p: Proposal, sess: SessionView, now_ms: int, pol: SafetyPolicy = DEFAULT_POLICY
) -> ConfirmDecision:
    """R7, R8 and R9 for one `ui.confirm`. Returns accept, second (first half of a double
    confirm; the caller issues a fresh nonce) or reject.

    Times: `now_ms`, `sess.proposal_sent_ms`, `sess.focus_since_ms` and `p.expires_at` are
    gateway clock; `c.ts`, `c.shown_at` and `c.focused_at` are phone clock and are only
    compared with each other, so no clock offset is needed.
    """

    def reject(rule: str, reason: str, option: Option | None = None) -> ConfirmDecision:
        return ConfirmDecision("reject", rule, reason, option)

    # R8: a confirm starts motion, so only from AWAITING_CONFIRM; a resume card only while PAUSED.
    want_mode = "PAUSED" if sess.proposal_purpose == "resume" else "AWAITING_CONFIRM"
    if sess.mode != want_mode:
        return reject(R8, f"confirm not allowed in mode {sess.mode}")

    # R9 binding to the live proposal.
    if sess.proposal_id is None or sess.proposal_id != p.proposal_id:
        return reject(R9, "no live proposal")
    if c.proposal_id != p.proposal_id:
        return reject(R9, "proposal_id does not match the live proposal")
    option = next((o for o in p.options if o.option_id == c.option_id), None)
    if option is None:
        return reject(R9, f"unknown option {c.option_id!r}")
    if c.nonce != p.nonce:
        return reject(R9, "nonce does not match", option)
    if c.nonce in sess.used_nonces:
        return reject(R9, "nonce already used", option)
    if sess.lease != "user":
        return reject(R9, f"confirm from a session without the user lease ({sess.lease})", option)
    if not sess.hb_fresh:
        return reject(R9, "phone heartbeat is stale", option)
    if not sess.face_present:
        return reject(R9, "last heartbeat did not see a face", option)
    if now_ms >= p.expires_at:
        return reject(R9, "proposal expired", option)

    # Arm delay: gateway clock (since the proposal was sent) and phone clock (since shown).
    if sess.proposal_sent_ms is None or now_ms - sess.proposal_sent_ms < p.arm_delay_ms:
        return reject(R9, "confirm before the arm delay", option)
    if c.ts - c.shown_at < p.arm_delay_ms:
        return reject(R9, "confirm before the arm delay (phone clock)", option)

    # Focus dwell: the gateway's own focus record, and the phone's.
    if sess.focus_option != c.option_id or sess.focus_since_ms is None:
        return reject(R9, "option is not the focused option", option)
    if now_ms - sess.focus_since_ms < pol.focus_dwell_ms:
        return reject(R9, "focus has not rested on the option long enough", option)
    if c.focused_at < c.shown_at or c.ts - c.focused_at < pol.focus_dwell_ms:
        return reject(R9, "focus dwell too short (phone clock)", option)

    # Gesture: a double blink never starts motion (6.1); a long blink must be long.
    if c.gesture == "double_blink":
        return reject(R9, "double blink never starts motion", option)
    if c.gesture == "long_blink" and c.held_ms < pol.min_confirm_hold_ms:
        return reject(R9, f"long blink held {c.held_ms} ms < {pol.min_confirm_hold_ms} ms", option)

    # R7: locked cards ("Needs caregiver") cannot be confirmed.
    if option.locked:
        return reject(R7, "needs caregiver", option)

    if option.confirm == "double" and sess.pending_double != (p.proposal_id, option.option_id):
        return ConfirmDecision("second", None, "first of two confirms", option)
    return ConfirmDecision("accept", None, "", option)


def spacing_wait_ms(
    last_motion_end_ms: int | None, now_ms: int, user_confirmed: bool, pol: SafetyPolicy = DEFAULT_POLICY
) -> int:
    """R11: how long to wait before starting a motion (0 if the user confirmed this step)."""
    if user_confirmed or last_motion_end_ms is None:
        return 0
    return max(0, last_motion_end_ms + pol.spacing_ms - now_ms)


def pre_decision(preconditions_hold: bool, cosmos: CosmosResult | None) -> tuple[Literal["proceed", "block"], bool]:
    """5.4 pre-check table. Returns (decision, unsure_from_camera).

    Oracle fail: block. Oracle holds and Cosmos yes: proceed. Oracle holds and Cosmos
    no/unsure (or Cosmos unavailable): proceed with an "Unsure from camera" badge and a
    double confirm. Cosmos unavailable (None) does not add the badge: the camera check is
    advisory and its absence is shown on /healthz instead.
    """
    if not preconditions_hold:
        return "block", False
    if cosmos is None:
        return "proceed", False
    return "proceed", cosmos.answer != "yes"


def post_decision(oracle: bool | None, cosmos: CosmosResult | None) -> Literal["done", "failed", "uncertain"]:
    """5.4 post-check table.

    | oracle | cosmos      | decision  |
    | true   | yes         | done      |
    | false  | no          | failed    |
    | true   | no / unsure | uncertain |
    | false  | yes         | uncertain |
    Cases the table leaves open: false + unsure -> failed (the oracle is ground truth in
    sim and Cosmos does not contradict it); no Cosmos -> the oracle alone; no oracle (a
    real robot) -> Cosmos alone (yes done, no failed, unsure uncertain); neither -> uncertain.
    """
    answer = cosmos.answer if cosmos is not None else None
    if oracle is None:
        return {"yes": "done", "no": "failed"}.get(answer or "", "uncertain")  # type: ignore[return-value]
    if answer is None:
        return "done" if oracle else "failed"
    if oracle:
        return "done" if answer == "yes" else "uncertain"
    return "uncertain" if answer == "yes" else "failed"


def oracle_success(spec: SkillSpec, world: WorldState) -> bool | None:
    """Oracle success after a run: the goal atom (first positive effect) holds; for a skill with
    only delete effects, all effects hold; None for a skill with no effects."""
    if spec.goal_atom:
        return holds(spec.goal_atom, world.atoms)
    if spec.effects:
        return all(holds(e, world.atoms) for e in spec.effects)
    return None
