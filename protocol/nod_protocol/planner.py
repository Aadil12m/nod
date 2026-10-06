"""Planner contract: Nemotron output (7.4) and the request the gateway builds (7.3)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from .base import HomeAction, NodModel, OptionKind, Risk, SkillKind, StrictModel
from .gateway import Success
from .world import WorldState

Intent = Literal["manipulate", "home", "caregiver", "clarify", "none"]
MENU_NO_SPEECH = "MENU_OPENED_NO_SPEECH"
HOME_ACTIONS: tuple[str, ...] = ("lights_on", "lights_off", "call_caregiver")


class PlannerOption(StrictModel):
    """One option as Nemotron returns it. All five keys are required (may be null).

    Kind/field consistency is NOT enforced here, so a single bad option does not
    fail the whole output (R1). Use `is_consistent()` and drop bad options instead.
    """

    kind: OptionKind
    skill_id: str | None
    plan: Annotated[list[str], Field(min_length=2, max_length=4)] | None
    home_action: HomeAction | None
    why: str = Field(max_length=80)

    def is_consistent(self) -> bool:
        """skill -> skill_id only; plan -> plan only; home -> home_action only."""
        if self.kind == "skill":
            return self.skill_id is not None and self.plan is None and self.home_action is None
        if self.kind == "plan":
            return self.plan is not None and self.skill_id is None and self.home_action is None
        return self.home_action is not None and self.skill_id is None and self.plan is None


class PlannerOutput(StrictModel):
    """The 7.4 output schema. R1 is `PlannerOutput.model_validate_json(text)` succeeding."""

    intent: Intent
    clarify_question: str | None = Field(max_length=60)
    options: list[PlannerOption] = Field(max_length=3)


def planner_response_format(feasible_ids: list[str], plan_ids: list[str]) -> dict[str, Any]:
    """Build the 7.4 `json_schema` object with the per-request enums filled in.

    feasible_ids: ids allowed as `skill_id` (from `feasible_skills()`).
    plan_ids: policy and custom skill ids allowed inside `plan`.
    Use as `response_format={"type": "json_schema", "json_schema": <this>}`, or pass
    `<this>["schema"]` as the parameters of the `propose_options` tool.
    """
    skill_enum: list[Any] = sorted(set(feasible_ids)) + [None]
    if plan_ids:
        plan_schema: dict[str, Any] = {
            "type": ["array", "null"], "minItems": 2, "maxItems": 4,
            "items": {"type": "string", "enum": sorted(set(plan_ids))},
        }
    else:  # an empty enum is invalid JSON Schema; no plans are possible
        plan_schema = {"type": "null"}
    return {
        "name": "nod_plan",
        "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "required": ["intent", "clarify_question", "options"],
            "properties": {
                "intent": {"type": "string", "enum": ["manipulate", "home", "caregiver", "clarify", "none"]},
                "clarify_question": {"type": ["string", "null"], "maxLength": 60},
                "options": {
                    "type": "array", "minItems": 0, "maxItems": 3,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["kind", "skill_id", "plan", "home_action", "why"],
                        "properties": {
                            "kind": {"type": "string", "enum": ["skill", "plan", "home"]},
                            "skill_id": {"type": ["string", "null"], "enum": skill_enum},
                            "plan": plan_schema,
                            "home_action": {"type": ["string", "null"], "enum": [*HOME_ACTIONS, None]},
                            "why": {"type": "string", "maxLength": 80},
                        },
                    },
                },
            },
        },
    }


# ---------------------------------------------------------------- request context (7.3)


class Profile(NodModel):
    preferred_name: str | None = None
    caregiver_name: str | None = None
    caregiver_enabled: bool = False


class FeasibleSkill(NodModel):
    skill_id: str
    kind: SkillKind
    label: str  # canonical registry label
    success_alone: Success | None = None
    success_in_chain: Success | None = None
    est_s: float | None = None
    risk: Risk


class InfeasibleSkill(NodModel):
    skill_id: str
    rule: str  # failing validator rule, e.g. "R4_preconditions"
    reason: str = ""  # e.g. "not on(cream_cheese_1,akita_black_bowl_1) is false"


class HistoryItem(NodModel):
    skill_id: str
    outcome: Literal["done", "failed", "uncertain", "aborted"]
    at: int  # gateway clock ms


class FailureInfo(NodModel):
    """Recovery context (`plan.md`, Super)."""

    skill_id: str
    steps: int
    cosmos_reason: str | None = None
    oracle: bool | None = None


class PlanRequest(NodModel):
    """Everything `Planner.propose` needs. The prompt builder renders it to JSON context."""

    mode: Literal["fast", "plan", "recovery"] = "fast"
    input: str  # transcript, or MENU_NO_SPEECH
    profile: Profile = Field(default_factory=Profile)
    time: str  # local time, ISO 8601
    world: WorldState
    feasible: list[FeasibleSkill] = Field(default_factory=list)
    infeasible: list[InfeasibleSkill] = Field(default_factory=list)
    plan_ids: list[str] = Field(default_factory=list)  # ids allowed in plans (7.4 enum)
    history: list[HistoryItem] = Field(default_factory=list, max_length=5)
    failure: FailureInfo | None = None
