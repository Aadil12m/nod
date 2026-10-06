"""Table-driven tests for every validator rule in design 5.2, plus property tests (4.6)."""

from __future__ import annotations

import dataclasses

import pytest
from conftest import make_world, view
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from nod_protocol import (
    CosmosResult,
    PlannerOption,
    PlannerOutput,
    Proposal,
    SkillSpec,
    SkillStats,
    StatBlock,
    Success,
    UiConfirm,
    holds,
    parse_literal,
)

from nod_gateway.registry import Registry, deterministic_menu
from nod_gateway.validator import (
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
    R8,
    R9,
    R10,
    R12,
    R14,
    SafetyPolicy,
    check_plan,
    feasible_skills,
    oracle_success,
    post_decision,
    pre_decision,
    reliability,
    sanitize_why,
    skill_rules,
    spacing_wait_ms,
    validate_confirm,
    validate_proposal,
)

BOWL_ON_PLATE = "on(akita_black_bowl_1,plate_1)"
CHEESE_IN_BOWL = "on(cream_cheese_1,akita_black_bowl_1)"
CHAIN_A = ["bottle_on_rack", "open_middle_drawer", "cheese_in_bowl", "plate_to_stove_front"]
CHAIN_B = ["bowl_on_plate", "cheese_in_bowl", "bowl_to_user"]


def skill(sid: str, why: str = "") -> PlannerOption:
    return PlannerOption(kind="skill", skill_id=sid, plan=None, home_action=None, why=why)


def plan(*ids: str, why: str = "") -> PlannerOption:
    return PlannerOption(kind="plan", skill_id=None, plan=list(ids), home_action=None, why=why)


def home(action: str, why: str = "") -> PlannerOption:
    return PlannerOption(kind="home", skill_id=None, plan=None, home_action=action, why=why)


def out(*opts: PlannerOption, intent: str = "manipulate") -> PlannerOutput:
    return PlannerOutput(intent=intent, clarify_question=None, options=list(opts))


def modify(reg: Registry, sid: str, **update) -> Registry:
    specs = [s.model_copy(update=update) if s.skill_id == sid else s for s in reg]
    return Registry.from_specs(specs, reg.stats, reg.scene_id)


def with_alone(reg: Registry, sid: str, k: int | None, n: int | None) -> Registry:
    alone = StatBlock.from_counts(k, n) if n is not None else None
    return modify(reg, sid, stats=SkillStats(alone=alone, in_chain=None))


def dropped_rule(res, sid: str) -> str | None:
    for d in res.dropped:
        if sid in d.skill_ids:
            return d.rule
    return None


# ---------------------------------------------------------------- R1 schema


def test_r1_inconsistent_option_is_dropped(reg, world):
    bad = PlannerOption(
        kind="skill", skill_id="bowl_on_plate", plan=["bowl_on_plate", "cheese_in_bowl"], home_action=None, why=""
    )
    res = validate_proposal(out(bad, skill("cheese_in_bowl")), world, reg, view())
    assert res.ok
    assert [d.rule for d in res.dropped] == [R1]
    assert [o.skill_ids for o in res.options] == [["cheese_in_bowl"]]
    assert res.checks[0].rule == R1 and res.checks[0].ok


# ---------------------------------------------------------------- R2 known skill


@pytest.mark.parametrize("case", ["unknown", "disabled", "scene", "not_offered", "home_as_skill"])
def test_r2_known_skill(reg, case):
    world = make_world(reg)
    sid = "bowl_on_plate"
    r = reg
    if case == "unknown":
        sid = "microwave_snack"
    elif case == "disabled":
        r = modify(reg, sid, enabled=False)
    elif case == "scene":
        world = make_world(reg, scene_id="other_scene")
    elif case == "not_offered":
        sid = "return_home"
    elif case == "home_as_skill":
        sid = "lights_on"
    res = validate_proposal(out(skill(sid)), world, r, view())
    assert res.options == []
    assert dropped_rule(res, sid) == R2


def test_r2_home_action_disabled(reg, world):
    r = modify(reg, "lights_on", enabled=False)
    res = validate_proposal(out(home("lights_on"), home("call_caregiver")), world, r, view())
    assert [o.home_action for o in res.options] == ["call_caregiver"]
    assert res.dropped[0].rule == R2


@pytest.mark.parametrize(
    "ids",
    [
        ("bowl_on_plate", "microwave_snack"),
        ("open_middle_drawer", "close_middle_drawer"),
        ("bowl_on_plate", "lights_on"),
    ],
)
def test_r2_plan_ids_must_be_policy_skills(reg, world, ids):
    res = validate_proposal(out(plan(*ids)), world, reg, view())
    assert res.options == [] and res.dropped[0].rule == R2


# ---------------------------------------------------------------- R3 objects


def test_r3_missing_object(reg):
    world = make_world(reg, drop_objects=("akita_black_bowl_1",))
    res = validate_proposal(out(skill("bowl_on_plate"), skill("bottle_on_rack")), world, reg, view())
    assert dropped_rule(res, "bowl_on_plate") == R3
    assert [o.skill_ids for o in res.options] == [["bottle_on_rack"]]


def test_r3_plan_step_missing_object(reg):
    world = make_world(reg, drop_objects=("plate_1",))
    res = validate_proposal(out(plan(*CHAIN_A)), world, reg, view())
    assert res.options == [] and res.dropped[0].rule == R3


# ---------------------------------------------------------------- R4 preconditions


@pytest.mark.parametrize(
    "sid,true_atoms",
    [
        ("bowl_on_plate", (CHEESE_IN_BOWL,)),  # don't move a bowl holding cheese (4.2.3)
        ("bowl_on_plate", (BOWL_ON_PLATE,)),
        ("stove_on", ("on(akita_black_bowl_1,flat_stove_1_cook_region)",)),
        ("open_middle_drawer", ("open(wooden_cabinet_1_top_region)",)),
        ("close_middle_drawer", ()),  # the drawer is closed
    ],
)
def test_r4_false_precondition(reg, sid, true_atoms):
    world = make_world(reg, true_atoms)
    res = validate_proposal(out(skill(sid)), world, reg, view())
    assert res.options == []
    assert dropped_rule(res, sid) == R4
    assert "precondition false" in res.dropped[0].reason


def test_r4_unknown_atom_never_satisfies(reg):
    world = make_world(reg, unknown=(BOWL_ON_PLATE,))
    res = validate_proposal(out(skill("bowl_on_plate")), world, reg, view())
    assert dropped_rule(res, "bowl_on_plate") == R4


def test_r4_holds_in_initial_world(reg, world):
    res = validate_proposal(out(skill("bowl_on_plate")), world, reg, view())
    assert len(res.options) == 1
    assert {c.rule: c.ok for c in res.options[0].checks}[R4] is True


# ---------------------------------------------------------------- R5 not a no-op


def _synthetic(reg: Registry, **kw) -> Registry:
    base = dict(
        skill_id="nudge",
        kind="policy",
        label="Nudge",
        scene_id="libero_goal_v1",
        policy={"checkpoint": "x", "instruction": "nudge it"},
        goal_ref="nod/nudge",
        stats=SkillStats(alone=StatBlock.from_counts(20, 20)),
    )
    base.update(kw)
    return Registry.from_specs([*reg, SkillSpec.model_validate(base)], reg.stats, reg.scene_id)


def test_r5_all_effects_already_true(reg):
    r = _synthetic(reg, pre=[], effects=[BOWL_ON_PLATE])
    world = make_world(reg, (BOWL_ON_PLATE,))
    res = validate_proposal(out(skill("nudge")), world, r, view())
    assert dropped_rule(res, "nudge") == R5


def test_r5_not_noop_when_some_effect_false(reg):
    r = _synthetic(reg, pre=[], effects=[BOWL_ON_PLATE, CHEESE_IN_BOWL])
    world = make_world(reg, (BOWL_ON_PLATE,))
    res = validate_proposal(out(skill("nudge")), world, r, view())
    assert len(res.options) == 1


def test_r5_empty_effects_exempt(reg, world, pol):
    f = skill_rules(reg["return_home"], world, pol, world.atoms, False, offer_required=False)
    assert f.ok and "scripted" in f.badges


# ---------------------------------------------------------------- R6 reliability


@pytest.mark.parametrize(
    "k,n,band",
    [
        (19, 20, "ok"),  # wilson_lo 0.76
        (20, 20, "ok"),
        (16, 20, "low"),  # 0.584: less reliable, double confirm
        (13, 20, "low"),  # 0.43
        (8, 20, "hidden"),  # 0.22 < 0.30
        (9, 9, "hidden"),  # n < 10
        (15, 15, "low"),  # high rate but 10 <= n < 20: not "offered normally"
        (None, None, "hidden"),  # no stats at all
    ],
)
def test_r6_bands(reg, world, k, n, band):
    r = with_alone(reg, "bottle_on_rack", k, n)
    res = validate_proposal(out(skill("bottle_on_rack")), world, r, view())
    if band == "hidden":
        assert res.options == [] and dropped_rule(res, "bottle_on_rack") == R6
        return
    (o,) = res.options
    assert o.success is not None and o.success.n == n and o.success.k == k
    if band == "ok":
        assert o.confirm == "single" and "less_reliable" not in o.badges
    else:
        assert o.confirm == "double" and "less_reliable" in o.badges


def test_r6_uses_in_chain_stats_once_in_chain(reg, world):
    # bottle_on_cabinet: alone 18/20 (0.70, ok), in-chain 15/20 (0.53, low).
    alone = validate_proposal(out(skill("bottle_on_cabinet")), world, reg, view(in_chain=False)).options[0]
    chain = validate_proposal(out(skill("bottle_on_cabinet")), world, reg, view(in_chain=True)).options[0]
    assert alone.success.basis == "alone" and alone.confirm == "single"
    assert chain.success.basis == "in_chain" and chain.confirm == "double" and "less_reliable" in chain.badges


def test_r6_exempts_scripted_and_home(reg, pol):
    world = make_world(reg, ("open(wooden_cabinet_1_middle_region)",))
    res = validate_proposal(out(skill("close_middle_drawer"), home("lights_on")), world, reg, view())
    assert [o.label for o in res.options] == ["Close the middle drawer", "Turn the lights on"]
    assert "scripted" in res.options[0].badges
    assert res.options[1].success is None


@pytest.mark.parametrize(
    "wl,n,expected",
    [
        (0.6, 20, "ok"),
        (0.5999, 20, "low"),
        (0.3, 20, "low"),
        (0.2999, 20, "hidden"),
        (0.9, 10, "low"),
        (0.9, 9, "hidden"),
    ],
)
def test_reliability_thresholds(pol, wl, n, expected):
    assert reliability(Success(p_est=0.9, basis="alone", n=n, wilson_lo=wl), pol) == expected


# ---------------------------------------------------------------- R7 risk


def test_r7_high_risk_locked_without_caregiver(reg, world):
    res = validate_proposal(out(skill("stove_on")), world, reg, view(caregiver_enabled=False))
    (o,) = res.options
    assert o.locked and "needs_caregiver" in o.badges and o.confirm == "double"
    assert any(c.rule == R7 and not c.ok for c in o.checks)


def test_r7_high_risk_with_caregiver_needs_double(reg, world):
    (o,) = validate_proposal(out(skill("stove_on")), world, reg, view(caregiver_enabled=True)).options
    assert not o.locked and o.confirm == "double"


def test_r7_plan_with_high_risk_step_locked(reg, world):
    (o,) = validate_proposal(out(plan("bowl_on_plate", "stove_on")), world, reg, view()).options
    assert o.locked and o.risk == "high" and o.confirm == "per_step"


def test_r7_locked_option_cannot_be_confirmed(reg, world):
    p, v, c, now = _confirmable(reg, world, "stove_on")
    d = validate_confirm(c, p, v, now)
    assert d.outcome == "reject" and d.rule == R7


# ---------------------------------------------------------------- R8 mode


@pytest.mark.parametrize("mode", ["ESTOPPED", "PAUSED", "EXECUTING", "COUNTDOWN", "POSTCHECK"])
def test_r8_no_new_motion(reg, world, mode):
    res = validate_proposal(out(skill("bowl_on_plate")), world, reg, view(mode=mode))
    assert not res.ok and res.rule == R8 and res.options == []


@pytest.mark.parametrize("mode", ["IDLE", "LISTENING", "PLANNING", "AWAITING_CONFIRM"])
def test_r8_allowed_modes(reg, world, mode):
    assert validate_proposal(out(skill("bowl_on_plate")), world, reg, view(mode=mode)).ok


@pytest.mark.parametrize("mode", ["IDLE", "ESTOPPED", "PAUSED", "EXECUTING", "COUNTDOWN", "POSTCHECK"])
def test_r8_confirm_only_when_awaiting(reg, world, mode):
    p, v, c, now = _confirmable(reg, world)
    d = validate_confirm(c, p, dataclasses.replace(v, mode=mode), now)
    assert d.outcome == "reject" and d.rule == R8


def test_r8_resume_card_only_while_paused(reg, world):
    p, v, c, now = _confirmable(reg, world)
    v = dataclasses.replace(v, proposal_purpose="resume")
    assert validate_confirm(c, p, dataclasses.replace(v, mode="AWAITING_CONFIRM"), now).rule == R8
    assert validate_confirm(c, p, dataclasses.replace(v, mode="PAUSED"), now).outcome == "accept"


# ---------------------------------------------------------------- R9 confirm binding

T0 = 1_791_205_200_000  # gateway clock when the proposal was sent
P0 = 1_000_000  # phone clock when the card was shown


def _confirmable(reg, world, sid: str = "bowl_on_plate"):
    res = validate_proposal(out(skill(sid)), world, reg, view())
    p = Proposal(
        proposal_id="p_1",
        expires_at=T0 + 30_000,
        arm_delay_ms=300,
        source="fake",
        options=res.options,
        nonce="n_1",
        focus="o1",
    )
    v = view(proposal_id="p_1", proposal_sent_ms=T0, focus_option="o1", focus_since_ms=T0 + 100)
    c = UiConfirm(
        proposal_id="p_1",
        option_id="o1",
        nonce="n_1",
        gesture="long_blink",
        held_ms=750,
        shown_at=P0,
        focused_at=P0 + 100,
        ts=P0 + 1500,
    )
    return p, v, c, T0 + 1500


def test_r9_valid_confirm_accepted(reg, world):
    p, v, c, now = _confirmable(reg, world)
    d = validate_confirm(c, p, v, now)
    assert d.outcome == "accept" and d.ok and d.option.option_id == "o1"


R9_CASES = {
    "no_live_proposal": lambda p, v, c, n: (p, dataclasses.replace(v, proposal_id=None), c, n),
    "stale_proposal": lambda p, v, c, n: (p, dataclasses.replace(v, proposal_id="p_0"), c, n),
    "wrong_proposal_id": lambda p, v, c, n: (p, v, c.model_copy(update={"proposal_id": "p_x"}), n),
    "unknown_option": lambda p, v, c, n: (p, v, c.model_copy(update={"option_id": "o9"}), n),
    "wrong_nonce": lambda p, v, c, n: (p, v, c.model_copy(update={"nonce": "n_x"}), n),
    "nonce_reused": lambda p, v, c, n: (p, dataclasses.replace(v, used_nonces=frozenset({"n_1"})), c, n),
    "caregiver_lease": lambda p, v, c, n: (p, v.with_lease("caregiver"), c, n),
    "no_lease": lambda p, v, c, n: (p, v.with_lease("none"), c, n),
    "hb_stale": lambda p, v, c, n: (p, dataclasses.replace(v, hb_fresh=False), c, n),
    "no_face": lambda p, v, c, n: (p, dataclasses.replace(v, face_present=False), c, n),
    "expired": lambda p, v, c, n: (p, v, c, T0 + 30_000),
    "arm_delay_gateway": lambda p, v, c, n: (p, dataclasses.replace(v, focus_since_ms=T0 - 1000), c, T0 + 299),
    "arm_delay_phone": lambda p, v, c, n: (p, v, c.model_copy(update={"ts": P0 + 299, "focused_at": P0 - 1}), n),
    "not_focused": lambda p, v, c, n: (p, dataclasses.replace(v, focus_option="o2"), c, n),
    "focus_dwell_gateway": lambda p, v, c, n: (p, dataclasses.replace(v, focus_since_ms=n - 299), c, n),
    "focus_dwell_phone": lambda p, v, c, n: (p, v, c.model_copy(update={"focused_at": P0 + 1201}), n),
    "focused_before_shown": lambda p, v, c, n: (p, v, c.model_copy(update={"focused_at": P0 - 5}), n),
    "double_blink": lambda p, v, c, n: (p, v, c.model_copy(update={"gesture": "double_blink"}), n),
    "short_long_blink": lambda p, v, c, n: (p, v, c.model_copy(update={"held_ms": 599}), n),
}


@pytest.mark.parametrize("case", sorted(R9_CASES))
def test_r9_rejections(reg, world, case):
    p, v, c, now = R9_CASES[case](*_confirmable(reg, world))
    d = validate_confirm(c, p, v, now)
    assert d.outcome == "reject", case
    assert d.rule == R9, (case, d.rule, d.reason)


def test_r9_boundaries_accept(reg, world):
    p, v, c, _ = _confirmable(reg, world)
    # exactly the arm delay and exactly the dwell, on both clocks
    v = dataclasses.replace(v, focus_since_ms=T0)
    c = c.model_copy(update={"focused_at": P0, "ts": P0 + 300, "held_ms": 600})
    assert validate_confirm(c, p, v, T0 + 300).outcome == "accept"
    assert validate_confirm(c, p, v, T0 + 29_999).outcome == "accept"


def test_r9_other_confirm_gestures_allowed(reg, world):
    # 6.3: users who cannot long-blink may confirm with an alternative gesture
    p, v, c, now = _confirmable(reg, world)
    assert validate_confirm(c.model_copy(update={"gesture": "head_hold", "held_ms": 1500}), p, v, now).ok


def test_double_confirm_needs_two_confirms(reg, world):
    p, v, c, now = _confirmable(reg, world, "bowl_in_top_drawer")  # less reliable: double
    assert p.options[0].confirm == "double"
    first = validate_confirm(c, p, v, now)
    assert first.outcome == "second"
    second = validate_confirm(c, p, dataclasses.replace(v, pending_double=("p_1", "o1")), now)
    assert second.outcome == "accept"
    # a pending double for another option does not count
    other = validate_confirm(c, p, dataclasses.replace(v, pending_double=("p_1", "o2")), now)
    assert other.outcome == "second"


# ---------------------------------------------------------------- R10 plan check


@pytest.mark.parametrize("chain", [CHAIN_A, CHAIN_B])
def test_r10_measured_chains_pass(reg, world, chain):
    assert check_plan(chain, world, reg).ok
    (o,) = validate_proposal(out(plan(*chain)), world, reg, view()).options
    ch = reg.chain_for(chain)
    assert o.kind == "plan" and o.skill_ids == chain and o.confirm == "per_step"
    assert o.success.basis == "in_chain" and o.success.n == ch.n and o.success.k == ch.k
    assert o.success.p_est == pytest.approx(ch.k / ch.n)


def test_r10_estimate_is_product_of_in_chain_rates(reg, world):
    ids = ["bottle_on_rack", "open_middle_drawer"]  # not a measured chain
    (o,) = validate_proposal(out(plan(*ids)), world, reg, view()).options
    expected = 1.0
    for sid in ids:
        st_ = reg[sid].stats.in_chain
        expected *= st_.k / st_.n
    assert o.success.p_est == pytest.approx(expected, abs=1e-3)
    weakest = min((reg[s].stats.in_chain for s in ids), key=lambda b: b.wilson_lo)
    assert o.success.wilson_lo == weakest.wilson_lo


@pytest.mark.parametrize(
    "ids,step,rule",
    [
        (["bowl_on_stove", "stove_on"], 1, R10),  # lighting a stove with the bowl on it
        (["bowl_on_plate", "bowl_on_cabinet"], 1, R10),  # the bowl is no longer at its training start
        (["cheese_in_bowl", "bowl_on_plate"], 1, R10),  # never move a bowl holding cheese
        (["open_middle_drawer", "bowl_in_top_drawer"], 1, R10),  # both drawers open
        (["plate_to_stove_front", "bowl_on_plate"], 1, R10),
        (["bowl_on_plate", "bowl_on_plate"], 1, R10),
    ],
)
def test_r10_infeasible_plans(reg, world, ids, step, rule):
    pc = check_plan(ids, world, reg)
    assert not pc.ok and pc.failed_step == step and pc.rule == rule
    res = validate_proposal(out(plan(*ids)), world, reg, view())
    assert res.options == [] and res.dropped[0].rule == rule


def test_r10_first_step_failure_is_r4(reg):
    world = make_world(reg, (CHEESE_IN_BOWL,))
    pc = check_plan(["bowl_on_plate", "bowl_to_user"], world, reg)
    assert not pc.ok and pc.failed_step == 0 and pc.rule == R4


def test_r10_unknown_step_is_r2(reg, world):
    pc = check_plan(["bowl_on_plate", "nope"], world, reg)
    assert not pc.ok and pc.rule == R2 and pc.failed_step == 1


def test_plan_hidden_when_measured_chain_unreliable(reg, world):
    # make chain A's measured result very poor: the plan is hidden, singles are still offered
    stats = reg.stats.model_copy(deep=True)
    stats.chains["A"] = stats.chains["A"].model_copy(update={"k": 2, "n": 20, "wilson_lo": 0.03})
    r = Registry.from_specs(list(reg), stats, reg.scene_id)
    res = validate_proposal(out(plan(*CHAIN_A), skill("bottle_on_rack")), world, r, view())
    assert [o.kind for o in res.options] == ["skill"] and res.dropped[0].rule == R6


# ---------------------------------------------------------------- R11 spacing


@pytest.mark.parametrize(
    "last_end,now,confirmed,wait",
    [
        (None, 5000, False, 0),
        (1000, 2000, False, 1000),
        (1000, 3000, False, 0),
        (1000, 3500, False, 0),
        (1000, 1000, False, 2000),
        (1000, 1001, True, 0),  # the user confirmed this step
    ],
)
def test_r11_spacing(last_end, now, confirmed, wait):
    assert spacing_wait_ms(last_end, now, confirmed) == wait


# ---------------------------------------------------------------- R12 count


def test_r12_duplicates_dropped(reg, world):
    res = validate_proposal(out(skill("bowl_on_plate"), skill("bowl_on_plate"), home("lights_on")), world, reg, view())
    assert [o.option_id for o in res.options] == ["o1", "o2"]
    assert res.dropped[0].rule == R12


def test_r12_ids_and_max(reg, world):
    res = validate_proposal(
        out(skill("microwave"), skill("bowl_on_plate"), skill("bottle_on_rack")), world, reg, view()
    )
    assert [o.option_id for o in res.options] == ["o1", "o2"]
    pol = SafetyPolicy(max_options=1)
    res = validate_proposal(out(skill("bowl_on_plate"), skill("bottle_on_rack")), world, reg, view(), pol)
    assert len(res.options) == 1 and res.dropped[0].rule == R12


def test_r12_empty_output(reg, world):
    res = validate_proposal(out(intent="none"), world, reg, view())
    assert res.ok and res.options == []
    assert {c.rule: c.ok for c in res.checks}[R12] is False


# ---------------------------------------------------------------- R13 labels


def test_r13_label_from_registry_why_sanitized(reg, world):
    why = "<b>Tasty</b>\n**now**  `rm -rf`"
    (o,) = validate_proposal(out(skill("bowl_on_plate", why=why)), world, reg, view()).options
    assert o.label == reg["bowl_on_plate"].label
    assert "<" not in o.why and "*" not in o.why and "\n" not in o.why and "`" not in o.why
    assert "Tasty" in o.why


def test_r13_why_max_80():
    long = "word " * 40
    assert len(sanitize_why(long)) <= 80
    assert sanitize_why(None) == ""


def test_r13_plan_label_built_from_registry(reg, world):
    (o,) = validate_proposal(out(plan(*CHAIN_B)), world, reg, view()).options
    assert o.label.startswith(reg["bowl_on_plate"].label) and "2 more steps" in o.label


# ---------------------------------------------------------------- R14 near user


def test_r14_first_use_needs_double(reg, world):
    (o,) = validate_proposal(out(skill("bowl_to_user")), world, reg, view()).options
    assert {"near_user", "first_use"} <= set(o.badges) and o.confirm == "double"
    assert any(c.rule == R14 for c in o.checks)


def test_r14_after_first_use_single(reg, world):
    (o,) = validate_proposal(
        out(skill("bowl_to_user")), world, reg, view(near_user_used=frozenset({"bowl_to_user"}))
    ).options
    assert "near_user" in o.badges and "first_use" not in o.badges and o.confirm == "single"


# ---------------------------------------------------------------- 5.4 decision tables


def _cr(answer: str) -> CosmosResult:
    return CosmosResult(check_id="k_1", answer=answer, decision="done")


@pytest.mark.parametrize(
    "oracle,answer,decision",
    [
        (True, "yes", "done"),
        (False, "no", "failed"),
        (True, "no", "uncertain"),
        (True, "unsure", "uncertain"),
        (False, "yes", "uncertain"),
        (False, "unsure", "failed"),
        (True, None, "done"),
        (False, None, "failed"),
        (None, "yes", "done"),
        (None, "no", "failed"),
        (None, "unsure", "uncertain"),
        (None, None, "uncertain"),
    ],
)
def test_post_decision_table(oracle, answer, decision):
    assert post_decision(oracle, _cr(answer) if answer else None) == decision


@pytest.mark.parametrize(
    "hold,answer,decision,unsure",
    [
        (True, "yes", "proceed", False),
        (True, "no", "proceed", True),
        (True, "unsure", "proceed", True),
        (False, "yes", "block", False),
        (False, "no", "block", False),
        (True, None, "proceed", False),
    ],
)
def test_pre_decision_table(hold, answer, decision, unsure):
    assert pre_decision(hold, _cr(answer) if answer else None) == (decision, unsure)


def test_oracle_success(reg):
    assert oracle_success(reg["bowl_on_plate"], make_world(reg, (BOWL_ON_PLATE,))) is True
    assert oracle_success(reg["bowl_on_plate"], make_world(reg)) is False
    assert oracle_success(reg["close_middle_drawer"], make_world(reg)) is True  # only a delete effect
    assert oracle_success(reg["return_home"], make_world(reg)) is None


# ---------------------------------------------------------------- feasible_skills and the menu


def test_feasible_skills_initial_world(reg, world):
    feas = {f.skill_id: f for f in feasible_skills(world, reg)}
    assert "return_home" not in feas and "reset_scene" not in feas  # never offered
    assert feas["close_middle_drawer"].ok is False and feas["close_middle_drawer"].rule == R4
    for sid in ["bowl_on_plate", "cheese_in_bowl", "bottle_on_rack", "stove_on", "bowl_to_user"]:
        assert feas[sid].ok, sid
    assert feas["bowl_in_top_drawer"].confirm == "double"
    assert feas["lights_on"].ok and feas["lights_on"].success is None


def test_deterministic_menu(reg, world):
    feas = feasible_skills(world, reg)
    menu = deterministic_menu(feas, reg)
    res = validate_proposal(menu, world, reg, view(mode="IDLE"))
    assert len(res.options) == 3
    skills = [o for o in res.options if o.kind == "skill"]
    assert len(skills) == 2 and res.options[-1].home_action == "call_caregiver"
    # top by wilson_lo among offerable, low/no-risk skills
    best = sorted(
        (f for f in feas if f.ok and f.kind != "home" and f.risk != "high"), key=lambda f: -f.success.wilson_lo
    )[0]
    assert skills[0].skill_ids == [best.skill_id]
    assert all(o.risk != "high" for o in res.options)


def test_menu_is_deterministic(reg, world):
    feas = feasible_skills(world, reg)
    assert deterministic_menu(feas, reg) == deterministic_menu(feas, reg)


# ---------------------------------------------------------------- properties (hypothesis)


def _ids(reg: Registry) -> list[str]:
    return [*reg.skills, "microwave_snack", "fly", "drink_from_cup", ""]


@st.composite
def worlds(draw, reg: Registry):
    true_atoms = draw(st.lists(st.sampled_from(list(reg.atoms)), unique=True))
    unknown = draw(st.lists(st.sampled_from(list(reg.atoms)), unique=True, max_size=2))
    drop = draw(st.lists(st.sampled_from(list(reg.objects)), unique=True, max_size=1))
    return make_world(reg, tuple(true_atoms), unknown=tuple(unknown), drop_objects=tuple(drop))


@st.composite
def outputs(draw, reg: Registry):
    ids = _ids(reg)
    opts = []
    for _ in range(draw(st.integers(0, 3))):
        kind = draw(st.sampled_from(["skill", "plan", "home", "bad"]))
        if kind == "skill":
            opts.append(skill(draw(st.sampled_from(ids))))
        elif kind == "plan":
            opts.append(plan(*draw(st.lists(st.sampled_from(ids), min_size=2, max_size=4))))
        elif kind == "home":
            opts.append(home(draw(st.sampled_from(["lights_on", "lights_off", "call_caregiver"]))))
        else:
            opts.append(
                PlannerOption(kind="home", skill_id=draw(st.sampled_from(ids)), plan=None, home_action=None, why="")
            )
    return PlannerOutput(intent="manipulate", clarify_question=None, options=opts)


_REG = None


def _reg():
    global _REG
    if _REG is None:
        from conftest import SKILLS, STATS

        from nod_gateway.registry import load_registry

        _REG = load_registry(SKILLS, STATS)
    return _REG


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_property_no_unknown_skill_or_false_precondition_passes(data):
    reg = _reg()
    world = data.draw(worlds(reg))
    output = data.draw(outputs(reg))
    caregiver = data.draw(st.booleans())
    res = validate_proposal(output, world, reg, view(mode="IDLE", caregiver_enabled=caregiver))
    for o in res.options:
        if o.kind == "home":
            assert reg.home_skill(o.home_action) is not None
            continue
        assert o.skill_ids, o
        for sid in o.skill_ids:
            spec = reg.get(sid)
            assert spec is not None, f"unknown skill {sid} passed"
            assert (spec.enabled and spec.offer and spec.kind == "policy") or (
                o.kind == "skill" and spec.kind == "scripted"
            )
            assert all(obj in world.objects for obj in spec.objects)
        if o.kind == "skill":
            spec = reg[o.skill_ids[0]]
            assert all(holds(p, world.atoms) for p in spec.pre), "false precondition passed"
        else:
            assert check_plan(o.skill_ids, world, reg).ok, "infeasible plan passed"
        assert o.label  # always the registry label
        if any(reg[s].risk == "high" for s in o.skill_ids):
            assert o.locked == (not caregiver)
            if o.kind == "skill":
                assert o.confirm == "double"
        else:
            assert not o.locked
    assert len(res.options) <= 3
    assert len({(o.kind, tuple(o.skill_ids), o.home_action) for o in res.options}) == len(res.options)


@settings(max_examples=200, deadline=None)
@given(
    nonce=st.text(min_size=1, max_size=8),
    option=st.sampled_from(["o1", "o2", "none", "x"]),
    pid=st.sampled_from(["p_1", "p_2"]),
    now_off=st.integers(-1000, 40_000),
)
def test_property_confirm_never_accepted_without_binding(nonce, option, pid, now_off):
    reg = _reg()
    world = make_world(reg)
    p, v, c, _ = _confirmable(reg, world)
    c = c.model_copy(update={"nonce": nonce, "option_id": option, "proposal_id": pid})
    d = validate_confirm(c, p, v, T0 + now_off)
    if d.ok:
        assert c.nonce == "n_1" and c.option_id == "o1" and c.proposal_id == "p_1"
        assert T0 + 300 <= T0 + now_off < T0 + 30_000


def test_literals_parse_for_every_registry_entry(reg):
    for s in reg:
        for t in (*s.pre, *s.effects):
            parse_literal(t)
