"""Planner interface (4.2.2) and the prompt context builder (7.3)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from nod_protocol import PlannerOutput, PlanRequest, parse_atom
from pydantic import ValidationError

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class PlannerError(Exception):
    """Timeout, HTTP error, or output that fails R1 after every fallback. The caller shows
    the deterministic menu (5.7, 7.5)."""


@dataclass
class PlannerResult:
    """What the gateway records about one planner call (`runs/<id>/planner_trace.json`)."""

    output: PlannerOutput
    model: str
    source: str  # "nemotron" | "fake"
    mechanism: str = ""  # json_schema | tools | json_object | canned | rules
    latency_ms: float = 0.0
    raw: str | None = None
    request: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Planner(Protocol):
    async def propose(self, req: PlanRequest) -> PlannerOutput:
        """Return a parsed PlannerOutput or raise PlannerError."""
        ...


# ---------------------------------------------------------------- context (7.3)

_OBJECT_WORDS = {
    "akita_black_bowl_1": "the bowl",
    "plate_1": "the plate",
    "cream_cheese_1": "the cream cheese",
    "wine_bottle_1": "the wine bottle",
    "wine_rack_1_top_region": "the wine rack",
    "wooden_cabinet_1_top_side": "the top of the cabinet",
    "wooden_cabinet_1_top_region": "the top drawer",
    "wooden_cabinet_1_middle_region": "the middle drawer",
    "main_table_stove_front_region": "the front of the stove",
    "flat_stove_1_cook_region": "the stove",
    "flat_stove_1": "the stove",
    "main_table_handover_region": "the spot in front of you",
}


def _obj(name: str) -> str:
    return _OBJECT_WORDS.get(name, name.replace("_", " "))


def humanize_atom(atom: str) -> str:
    """'on(akita_black_bowl_1,plate_1)' -> 'the bowl is on the plate' (7.3: world in plain words)."""
    try:
        pred, args = parse_atom(atom)
    except ValueError:
        return atom
    if pred == "on" and len(args) == 2:
        return f"{_obj(args[0])} is on {_obj(args[1])}"
    if pred == "in" and len(args) == 2:
        return f"{_obj(args[0])} is in {_obj(args[1])}"
    if pred == "open" and len(args) == 1:
        return f"{_obj(args[0])} is open"
    if pred == "turnon" and len(args) == 1:
        return f"{_obj(args[0])} is switched on"
    return f"{pred} " + ", ".join(_obj(a) for a in args)


def build_context(req: PlanRequest) -> dict[str, Any]:
    """The JSON user message sent to Nemotron (7.3 item 3)."""
    ctx: dict[str, Any] = {
        "profile": req.profile.model_dump(mode="json"),
        "time": req.time,
        "world": [humanize_atom(a) for a, v in sorted(req.world.atoms.items()) if v]
        or ["everything is in its starting place"],
        "feasible": [
            {
                "id": f.skill_id,
                "label": f.label,
                "kind": f.kind,
                "success_alone": f.success_alone.model_dump(mode="json") if f.success_alone else None,
                "success_in_chain": f.success_in_chain.model_dump(mode="json") if f.success_in_chain else None,
                "est_s": f.est_s,
                "risk": f.risk,
            }
            for f in req.feasible
        ],
        "infeasible": [{"id": i.skill_id, "rule": i.rule, "reason": i.reason} for i in req.infeasible],
        "plan_ids": req.plan_ids,
        "history": [h.model_dump(mode="json") for h in req.history],
        "input": req.input,
    }
    if req.failure is not None:
        ctx["failure"] = req.failure.model_dump(mode="json")
    return ctx


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


def extract_json(text: str) -> str:
    """Strip reasoning blocks and code fences; return the outermost JSON object text."""
    t = _THINK.sub("", text or "").strip()
    t = _FENCE.sub("", t).strip()
    if t.startswith("{"):
        return t
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        return t[start : end + 1]
    return t


def parse_output(text: str) -> PlannerOutput:
    """R1: parse model text into PlannerOutput or raise PlannerError."""
    try:
        return PlannerOutput.model_validate_json(extract_json(text))
    except (ValidationError, ValueError) as exc:
        raise PlannerError(f"R1 schema failure: {str(exc)[:300]}") from exc


def output_from_obj(obj: Any) -> PlannerOutput:
    try:
        return PlannerOutput.model_validate(obj)
    except ValidationError as exc:
        raise PlannerError(f"R1 schema failure: {str(exc)[:300]}") from exc


def dumps_context(req: PlanRequest) -> str:
    return json.dumps(build_context(req), separators=(",", ":"))
