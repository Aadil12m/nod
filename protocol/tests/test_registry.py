"""registry/skills.yaml and registry/stats/latest.json load and behave as designed."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import nod_protocol as p
from conftest import SKILLS_YAML, STATS_JSON

# Design table 1.2: registry id -> (exact instruction, goal atom, LIBERO task name).
LIBERO_GOAL = {
    "bowl_on_plate": ("put the bowl on the plate", "on(akita_black_bowl_1,plate_1)"),
    "cheese_in_bowl": ("put the cream cheese in the bowl", "on(cream_cheese_1,akita_black_bowl_1)"),
    "bottle_on_rack": ("put the wine bottle on the rack", "on(wine_bottle_1,wine_rack_1_top_region)"),
    "bottle_on_cabinet": ("put the wine bottle on top of the cabinet", "on(wine_bottle_1,wooden_cabinet_1_top_side)"),
    "bowl_on_cabinet": ("put the bowl on top of the cabinet", "on(akita_black_bowl_1,wooden_cabinet_1_top_side)"),
    "bowl_in_top_drawer": ("open the top drawer and put the bowl inside",
                           "in(akita_black_bowl_1,wooden_cabinet_1_top_region)"),
    "open_middle_drawer": ("open the middle drawer of the cabinet", "open(wooden_cabinet_1_middle_region)"),
    "plate_to_stove_front": ("push the plate to the front of the stove", "on(plate_1,main_table_stove_front_region)"),
    "bowl_on_stove": ("put the bowl on the stove", "on(akita_black_bowl_1,flat_stove_1_cook_region)"),
    "stove_on": ("turn on the stove", "turnon(flat_stove_1)"),
}
CHAIN_A = ["bottle_on_rack", "open_middle_drawer", "cheese_in_bowl", "plate_to_stove_front"]
CHAIN_B = ["bowl_on_plate", "cheese_in_bowl", "bowl_to_user"]


@pytest.fixture(scope="module")
def by_id(specs):
    return {s.skill_id: s for s in specs}


@pytest.fixture(scope="module")
def initial(specs):
    """The initial LIBERO-Goal layout: every registry atom is false."""
    return {a: False for a in p.registry_atoms(specs)}


def test_loads_into_skill_specs(specs):
    assert all(isinstance(s, p.SkillSpec) for s in specs)
    assert p.check_registry(specs) == []
    assert len(specs) == 18


def test_ten_libero_goal_skills_match_design_table(by_id):
    for sid, (instruction, goal_atom) in LIBERO_GOAL.items():
        s = by_id[sid]
        assert s.kind == "policy" and not s.custom
        assert s.policy.instruction == instruction
        assert s.policy.checkpoint == "GR00T-N1.7-LIBERO/libero_goal"
        assert s.policy.embodiment == "LIBERO_PANDA"
        # LIBERO derives the instruction from the file name, so goal_ref is the instruction with underscores
        assert s.goal_ref == "libero_goal/" + instruction.replace(" ", "_")
        assert s.success_atoms[0] == goal_atom == s.goal_atom
        assert s.scene_id == "libero_goal_v1"
        assert s.cosmos and s.cosmos.pre and s.cosmos.post


def test_goal_refs_are_the_libero_task_names(by_id):
    names = {by_id[s].goal_ref.split("/", 1)[1] for s in LIBERO_GOAL}
    assert names == {
        "put_the_bowl_on_the_plate", "put_the_cream_cheese_in_the_bowl", "put_the_wine_bottle_on_the_rack",
        "put_the_wine_bottle_on_top_of_the_cabinet", "put_the_bowl_on_top_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside", "open_the_middle_drawer_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove", "put_the_bowl_on_the_stove", "turn_on_the_stove",
    }


def test_risk_confirm_and_near_user(by_id):
    high = {sid for sid, s in by_id.items() if s.risk == "high"}
    assert high == {"stove_on", "bowl_on_stove"}
    assert all(by_id[s].confirm == "double" for s in high)
    near = {sid for sid, s in by_id.items() if s.near_user}
    assert near == {"bowl_to_user"}


def test_custom_skill(by_id):
    s = by_id["bowl_to_user"]
    assert s.kind == "policy" and s.custom and s.near_user
    assert s.goal_ref == "nod/bring_the_bowl_to_me"
    assert s.policy.instruction == "bring the bowl to me"
    assert s.goal_atom == "on(akita_black_bowl_1,main_table_handover_region)"


def test_scripted_and_home_entries(by_id):
    scripted = {sid: s.scripted.script for sid, s in by_id.items() if s.kind == "scripted"}
    assert scripted == {sid: sid for sid in ("return_home", "reset_scene", "close_middle_drawer", "handover_retreat")}
    assert by_id["open_middle_drawer"].undo == "close_middle_drawer"
    homes = {sid: s.home.action for sid, s in by_id.items() if s.kind == "home"}
    assert homes == {"lights_on": "lights_on", "lights_off": "lights_off", "call_caregiver": "call_caregiver"}
    assert all(by_id[h].risk == "none" and by_id[h].scene_id is None for h in homes)
    assert not by_id["return_home"].offer and not by_id["reset_scene"].offer


def test_every_atom_is_well_formed_and_objects_are_named(specs):
    known_objects = set(p.registry_objects(specs))
    assert known_objects == {"akita_black_bowl_1", "plate_1", "cream_cheese_1", "wine_bottle_1",
                             "wooden_cabinet_1", "flat_stove_1", "wine_rack_1"}
    preds = {p.parse_literal(t).pred for s in specs for t in (*s.pre, *s.effects)}
    assert preds <= {"on", "in", "open", "turnon"}
    assert len(p.registry_atoms(specs)) == 12


@pytest.mark.parametrize("chain", [CHAIN_A, CHAIN_B], ids=["A", "B"])
def test_design_chains_pass_symbolic_rollout(chain, specs, initial):
    r = p.plan_rollout(chain, initial, specs)
    assert r.ok, r
    for sid in chain:  # every step's goal holds at the end
        for atom in next(s for s in specs if s.skill_id == sid).success_atoms:
            if sid != "bowl_on_plate":  # bowl_to_user takes the bowl off the plate
                assert r.atoms[atom] is True


def test_every_policy_skill_is_feasible_alone_from_the_initial_layout(specs, initial):
    for s in specs:
        if s.kind == "policy":
            assert p.unmet(s.pre, initial) == [], s.skill_id
            assert not all(p.holds(e, initial) for e in s.effects), s.skill_id  # R5: not a no-op


@pytest.mark.parametrize(
    "plan, failed_skill",
    [
        (["bowl_on_plate", "bowl_on_plate"], "bowl_on_plate"),  # already done
        (["cheese_in_bowl", "bowl_on_plate"], "bowl_on_plate"),  # never move a bowl holding cheese
        (["bowl_in_top_drawer", "cheese_in_bowl"], "cheese_in_bowl"),  # bowl is inside the drawer
        (["bowl_on_cabinet", "bottle_on_cabinet"], "bottle_on_cabinet"),  # cabinet top taken
        (["bottle_on_rack", "bottle_on_cabinet"], "bottle_on_cabinet"),  # bottle already placed
        (["stove_on", "bowl_on_stove"], "bowl_on_stove"),  # never put a bowl on a lit stove
        (["bowl_on_stove", "stove_on"], "stove_on"),  # never light a stove with a bowl on it
        (["stove_on", "plate_to_stove_front"], "plate_to_stove_front"),  # no reaching next to a lit stove
        (["bowl_on_plate", "plate_to_stove_front"], "plate_to_stove_front"),  # plate carries the bowl
        (["open_middle_drawer", "bowl_in_top_drawer"], "bowl_in_top_drawer"),  # drawers interfere
        (["bowl_in_top_drawer", "open_middle_drawer"], "open_middle_drawer"),
        (["bowl_to_user", "cheese_in_bowl"], "cheese_in_bowl"),  # no manipulation at the user
        (["bowl_to_user", "bowl_on_plate"], "bowl_on_plate"),
        (["bowl_on_cabinet", "bowl_to_user"], "bowl_to_user"),
        (["close_middle_drawer"], "close_middle_drawer"),  # nothing to close
        (["bowl_on_plate", "no_such_skill"], "no_such_skill"),
    ],
)
def test_infeasible_plans_fail(plan, failed_skill, specs, initial):
    r = p.plan_rollout(plan, initial, specs)
    assert not r.ok
    assert r.failed_skill == failed_skill
    assert r.unmet


def test_undo_and_reset_restore_state(specs, initial):
    r = p.plan_rollout(["open_middle_drawer", "close_middle_drawer", "open_middle_drawer"], initial, specs)
    assert r.ok
    r = p.plan_rollout(CHAIN_B + ["reset_scene"] + CHAIN_A, initial, specs)
    assert r.ok
    end = p.plan_rollout(CHAIN_B + ["reset_scene"], initial, specs).atoms
    assert end == initial


def test_unknown_atoms_block_preconditions(by_id):
    assert p.unmet(by_id["stove_on"].pre, {}) == by_id["stove_on"].pre


# ---------------------------------------------------------------- stats


def test_stats_file_is_a_labelled_placeholder():
    raw = json.loads(STATS_JSON.read_text())
    assert raw["placeholder"] is True and "placeholder" in raw["_note"].lower()
    sf = p.load_stats(STATS_JSON)
    assert sf.placeholder and set(sf.chains) == {"A", "B"}
    assert sf.chains["A"].skill_ids == CHAIN_A and sf.chains["B"].skill_ids == CHAIN_B
    for sid, st in sf.skills.items():
        for block in (st.alone, st.in_chain):
            assert block.n >= 20  # R6: visible in mock mode
            assert block.wilson_lo == round(p.wilson_lower(block.k, block.n), 4)
    assert sf.dump_json() == STATS_JSON.read_text()  # file is in canonical form


def test_stats_merge_into_specs(specs):
    by_id = {s.skill_id: s for s in specs}
    policy = [s for s in specs if s.kind == "policy"]
    assert all(s.stats.alone and s.stats.in_chain and s.stats.placeholder for s in policy)
    assert by_id["return_home"].stats == p.SkillStats()  # no stats for motions
    assert by_id["cheese_in_bowl"].stats.alone.k == 38 and by_id["cheese_in_bowl"].stats.alone.n == 40
    # at least one skill shows the R6 "less reliable" band so mock mode exercises the badge
    assert any(0.30 <= s.stats.alone.wilson_lo < 0.60 for s in policy)
    # loading without stats leaves them empty
    bare = p.load_skills(SKILLS_YAML)
    assert all(s.stats.alone is None for s in bare)


@pytest.mark.parametrize(
    "k, n, expected",
    [(0, 0, 0.0), (20, 20, 0.8389), (19, 20, 0.7639), (38, 40, 0.835), (10, 20, 0.2993), (0, 20, 0.0)],
)
def test_wilson_lower(k, n, expected):
    assert p.wilson_lower(k, n) == pytest.approx(expected, abs=1e-4)


def test_wilson_lower_rejects_bad_counts():
    with pytest.raises(ValueError):
        p.wilson_lower(5, 4)
    with pytest.raises(ValidationError):
        p.StatBlock(k=5, n=4, wilson_lo=0.5)


# ---------------------------------------------------------------- spec validation


def _spec(**kw):
    base = dict(skill_id="x", kind="policy", label="X", scene_id="libero_goal_v1",
                policy={"checkpoint": "c", "instruction": "do x"}, goal_ref="libero_goal/do_x")
    base.update(kw)
    return p.SkillSpec.model_validate(base)


@pytest.mark.parametrize(
    "kw",
    [
        dict(pre=["on(a, b)"]),  # spaces are not canonical
        dict(effects=["On(a,b)"]),  # uppercase predicate
        dict(pre=["not  on(a,b)"]),  # double space
        dict(goal_ref=None),  # policy needs goal_ref
        dict(policy=None),  # policy needs the policy block
        dict(home={"action": "lights_on"}),  # extra block
        dict(risk="medium"),
        dict(typo_field=1),  # extra keys are errors in the registry
        dict(skill_id="Bad-Id"),
    ],
)
def test_bad_specs_are_rejected(kw):
    with pytest.raises(ValidationError):
        _spec(**kw)


def test_registry_cross_checks():
    a = _spec(skill_id="a", undo="missing")
    b = _spec(skill_id="a")
    problems = p.check_registry([a, b])
    assert any("duplicate" in x for x in problems)
    assert any("undo" in x for x in problems)
    assert any("instruction" in x for x in problems)


def test_skill_spec_json_round_trip(specs):
    for s in specs:
        assert p.SkillSpec.model_validate_json(s.model_dump_json()) == s
        assert p.SkillSpec.model_validate(p.unpackb(p.packb(s.model_dump()))) == s
