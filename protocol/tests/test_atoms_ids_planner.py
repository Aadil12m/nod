"""Atoms, ULIDs, and the planner contract (7.4)."""

from __future__ import annotations

import json
import time

import jsonschema
import pytest
from pydantic import ValidationError

import nod_protocol as p

# ---------------------------------------------------------------- atoms


def test_parse_literal():
    lit = p.parse_literal("not on(akita_black_bowl_1,plate_1)")
    assert lit.neg and lit.pred == "on" and lit.args == ("akita_black_bowl_1", "plate_1")
    assert str(lit) == "not on(akita_black_bowl_1,plate_1)"
    assert p.parse_literal("turnon(flat_stove_1)").args == ("flat_stove_1",)
    for bad in ["on()", "on(a,)", "on a,b", "NOT on(a,b)", "on(a b)", ""]:
        with pytest.raises(ValueError):
            p.parse_literal(bad)


def test_holds_treats_unknown_as_unsatisfied():
    atoms = {"open(d)": True, "on(a,b)": False}
    assert p.holds("open(d)", atoms) and not p.holds("not open(d)", atoms)
    assert p.holds("not on(a,b)", atoms) and not p.holds("on(a,b)", atoms)
    assert not p.holds("in(x,y)", atoms) and not p.holds("not in(x,y)", atoms)


def test_apply_effects_and_rollout():
    atoms = {"on(a,b)": True}
    out = p.apply_effects(atoms, ["not on(a,b)", "on(a,c)"])
    assert out == {"on(a,b)": False, "on(a,c)": True} and atoms == {"on(a,b)": True}
    steps = [("s1", ["on(a,b)"], ["not on(a,b)", "on(a,c)"]), ("s2", ["on(a,c)"], ["open(d)"])]
    assert p.rollout(steps, atoms).ok
    r = p.rollout(list(reversed(steps)), atoms)
    assert not r.ok and r.failed_step == 0 and r.failed_skill == "s2" and r.unmet == ["on(a,c)"]


# ---------------------------------------------------------------- ids


def test_ulid_format_order_and_time():
    t0 = p.now_ms()
    ids = [p.new_ulid() for _ in range(2000)]
    assert all(p.is_ulid(i) for i in ids)
    assert ids == sorted(ids) and len(set(ids)) == len(ids)  # monotonic even within one ms
    assert abs(p.ulid_time_ms(ids[0]) - t0) < 1000
    assert p.ulid_time_ms(p.new_ulid(ts_ms=1791205200123)) == 1791205200123
    assert p.new_id("p").startswith("p_") and p.is_ulid(p.new_id("r")[2:])
    assert not p.is_ulid("01J9ZK3M8Q4T6W1Y2R5B7C9D0") and not p.is_ulid("8" * 26) and not p.is_ulid("I" * 26)
    assert p.is_ulid("01J9ZK3M8Q4T6W1Y2R5B7C9D0E")  # the design's example id


def test_now_ms():
    assert abs(p.now_ms() - int(time.time() * 1000)) < 50


def test_envelope_defaults():
    a, b = p.Ping(t0=1), p.Ping(t0=1)
    assert a.v == 1 and a.seq == 0 and a.sess is None and a.id != b.id and p.is_ulid(a.id)


# ---------------------------------------------------------------- planner

GOOD = {
    "intent": "manipulate", "clarify_question": None,
    "options": [
        {"kind": "plan", "skill_id": None, "plan": ["bowl_on_plate", "cheese_in_bowl", "bowl_to_user"],
         "home_action": None, "why": "You said you are hungry"},
        {"kind": "skill", "skill_id": "cheese_in_bowl", "plan": None, "home_action": None, "why": "Quick snack prep"},
        {"kind": "home", "skill_id": None, "plan": None, "home_action": "call_caregiver", "why": "Or ask for help"},
    ],
}


def test_planner_output_parses_and_round_trips():
    out = p.PlannerOutput.model_validate_json(json.dumps(GOOD))
    assert all(o.is_consistent() for o in out.options)
    assert p.PlannerOutput.model_validate_json(out.model_dump_json()) == out
    assert p.PlannerOutput.model_validate(p.unpackb(p.packb(out.model_dump()))) == out


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(extra=1),  # additionalProperties false
        lambda d: d.pop("clarify_question"),  # required even if null
        lambda d: d["options"].append(dict(d["options"][0])),  # max 3
        lambda d: d["options"][0].update(why="x" * 81),
        lambda d: d["options"][0].update(plan=["bowl_on_plate"]),  # plan min 2
        lambda d: d["options"][0].update(plan=["a", "b", "c", "d", "e"]),  # plan max 4
        lambda d: d["options"][2].update(home_action="unlock_door"),
        lambda d: d.update(intent="dance"),
        lambda d: d.update(clarify_question="x" * 61),
        lambda d: d["options"][1].pop("plan"),
    ],
)
def test_planner_output_rejects_schema_violations(mutate):
    d = json.loads(json.dumps(GOOD))
    mutate(d)
    with pytest.raises(ValidationError):
        p.PlannerOutput.model_validate_json(json.dumps(d))


def test_inconsistent_options_parse_but_are_flagged():
    d = json.loads(json.dumps(GOOD))
    d["options"][1]["skill_id"] = None  # kind skill without a skill id
    out = p.PlannerOutput.model_validate(d)
    assert not out.options[1].is_consistent() and out.options[0].is_consistent()


def test_response_format_matches_design_7_4():
    feasible = ["cheese_in_bowl", "bowl_on_plate"]
    plan_ids = ["bowl_on_plate", "cheese_in_bowl", "bowl_to_user"]
    rf = p.planner_response_format(feasible, plan_ids)
    assert rf["name"] == "nod_plan" and rf["strict"] is True
    schema = rf["schema"]
    assert schema["required"] == ["intent", "clarify_question", "options"]
    item = schema["properties"]["options"]["items"]
    assert item["required"] == ["kind", "skill_id", "plan", "home_action", "why"]
    assert item["properties"]["skill_id"]["enum"] == ["bowl_on_plate", "cheese_in_bowl", None]
    assert item["properties"]["plan"]["items"]["enum"] == sorted(plan_ids)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(GOOD, schema)  # the canned output satisfies the constrained schema
    bad = json.loads(json.dumps(GOOD))
    bad["options"][1]["skill_id"] = "stove_on"  # not feasible -> impossible to name
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)
    # with no plan ids, plans are impossible but the schema is still valid
    empty = p.planner_response_format([], [])
    jsonschema.Draft202012Validator.check_schema(empty["schema"])
    assert empty["schema"]["properties"]["options"]["items"]["properties"]["plan"] == {"type": "null"}


def test_plan_request_round_trip():
    world = p.WorldState(scene_id="libero_goal_v1", step=0, t_sim=0.0, atoms={"turnon(flat_stove_1)": False},
                         eef=p.Eef(pos=(0, 0, 1), gripper="open"))
    req = p.PlanRequest(
        input="I'm a bit hungry", time="2026-10-05T12:00:00+01:00", world=world,
        profile=p.Profile(preferred_name="Sam", caregiver_name="Alex", caregiver_enabled=False),
        feasible=[p.FeasibleSkill(skill_id="cheese_in_bowl", kind="policy", label="Put the cream cheese in the bowl",
                                  success_alone=p.Success(p_est=0.95, basis="alone", n=40, wilson_lo=0.83),
                                  est_s=8, risk="low")],
        infeasible=[p.InfeasibleSkill(skill_id="stove_on", rule="R7_risk", reason="needs caregiver")],
        plan_ids=["bowl_on_plate", "cheese_in_bowl", "bowl_to_user"],
        history=[p.HistoryItem(skill_id="bowl_on_plate", outcome="done", at=1)],
    )
    assert p.PlanRequest.model_validate_json(req.model_dump_json()) == req
    assert p.MENU_NO_SPEECH == "MENU_OPENED_NO_SPEECH"


def test_cosmos_reply_and_home_result():
    r = p.CosmosReply.model_validate_json('{"answer": "unsure", "reason": "bowl is occluded"}')
    assert r.answer == "unsure"
    with pytest.raises(ValidationError):
        p.CosmosReply.model_validate_json('{"answer": "maybe"}')
    h = p.HomeResult(action="lights_on", ok=True)
    assert p.HomeResult.model_validate_json(h.model_dump_json()) == h
