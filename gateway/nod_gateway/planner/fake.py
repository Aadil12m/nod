"""FakePlanner (mock mode, 4.4): recorded outputs keyed by transcript, deterministic regex rules otherwise.

Canned files live in `planner/canned/*.json`:

    {"transcript": "I'm a bit hungry",        # exact match after normalising case and punctuation
     "match": "hungry|snack",                 # optional regex (searched, case-insensitive)
     "mode": "fast",                          # optional: only used for this request mode
     "model": "nvidia/Nemotron-3_5-Lightning",
     "recorded": false,                       # true only for real Nemotron responses (runs/*/planner_trace.json)
     "output": { ...PlannerOutput (7.4)... }}

Canned outputs are returned verbatim, even when they name skills that are infeasible
right now: the validator then drops them, which is how mock mode shows a rejected
proposal. The regex rules, like constrained decoding, only name ids from the request's
feasible list.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from nod_protocol import MENU_NO_SPEECH, PlannerOption, PlannerOutput, PlanRequest

from .base import PlannerError, PlannerResult, output_from_obj


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


@dataclass(frozen=True)
class Canned:
    name: str
    transcript: str | None
    match: re.Pattern[str] | None
    mode: str | None
    model: str
    recorded: bool
    output: PlannerOutput


def load_canned(path: Path) -> list[Canned]:
    out: list[Canned] = []
    if not path.exists():
        return out
    for f in sorted(path.glob("*.json")):
        raw = json.loads(f.read_text())
        out.append(
            Canned(
                name=f.stem,
                transcript=raw.get("transcript"),
                match=re.compile(raw["match"], re.I) if raw.get("match") else None,
                mode=raw.get("mode"),
                model=raw.get("model", "fake"),
                recorded=bool(raw.get("recorded", False)),
                output=output_from_obj(raw["output"]),
            )
        )
    return out


def _skill(sid: str, why: str) -> PlannerOption:
    return PlannerOption(kind="skill", skill_id=sid, plan=None, home_action=None, why=why)


def _home(action: str, why: str) -> PlannerOption:
    return PlannerOption(kind="home", skill_id=None, plan=None, home_action=action, why=why)  # type: ignore[arg-type]


def _plan(ids: list[str], why: str) -> PlannerOption:
    return PlannerOption(kind="plan", skill_id=None, plan=ids, home_action=None, why=why)


Rule = tuple[re.Pattern[str], str, Callable[[PlanRequest], list[PlannerOption]]]


class FakePlanner:
    """Deterministic planner for mock mode and tests. Same interface as NemotronPlanner."""

    source = "fake"

    def __init__(self, canned: Path, *, delay_s: float = 0.0, fail: bool = False) -> None:
        self.canned = load_canned(canned)
        self.delay_s = delay_s  # simulate planner latency
        self.fail = fail  # simulate an outage (PlannerError) for tests
        self.calls: list[PlanRequest] = []
        self.rules: list[Rule] = self._rules()

    # ------------------------------------------------------------ interface

    async def propose(self, req: PlanRequest) -> PlannerOutput:
        return (await self.propose_ex(req)).output

    async def propose_ex(self, req: PlanRequest) -> PlannerResult:
        import asyncio

        t0 = time.perf_counter()
        self.calls.append(req)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail:
            raise PlannerError("fake planner set to fail")
        canned = self._find_canned(req)
        if canned is not None:
            out, mech, model = canned.output, "canned:" + canned.name, canned.model
        else:
            out, mech, model = self._by_rules(req), "rules", "fake-rules"
        return PlannerResult(
            output=out,
            model=model,
            source="fake",
            mechanism=mech,
            latency_ms=(time.perf_counter() - t0) * 1000,
            raw=out.model_dump_json(),
        )

    # ------------------------------------------------------------ canned

    def _find_canned(self, req: PlanRequest) -> Canned | None:
        if req.mode == "recovery":
            return None
        text = _norm(req.input)
        for c in self.canned:
            if c.mode and c.mode != req.mode:
                continue
            if c.transcript is not None and _norm(c.transcript) == text:
                return c
        for c in self.canned:
            if c.mode and c.mode != req.mode:
                continue
            if c.match is not None and c.match.search(req.input):
                return c
        return None

    # ------------------------------------------------------------ rules

    @staticmethod
    def _feasible(req: PlanRequest) -> list[str]:
        return [f.skill_id for f in req.feasible if f.kind != "home"]

    def _pick(self, req: PlanRequest, ids: list[str], why: str) -> list[PlannerOption]:
        feas = set(self._feasible(req))
        return [_skill(s, why) for s in ids if s in feas]

    def _rules(self) -> list[Rule]:
        def r(pattern: str, intent: str, fn: Callable[[PlanRequest], list[PlannerOption]]) -> Rule:
            return (re.compile(pattern, re.I), intent, fn)

        def snack(req: PlanRequest) -> list[PlannerOption]:
            out: list[PlannerOption] = []
            chain = ["bowl_on_plate", "cheese_in_bowl", "bowl_to_user"]
            if all(s in req.plan_ids for s in chain) and chain[0] in self._feasible(req):
                out.append(_plan(chain, "Sets up a snack and brings it to you"))
            singles = self._pick(req, ["cheese_in_bowl", "bowl_to_user"], "Quick snack prep")
            out += singles[: 1 if out else 2]
            out.append(_home("call_caregiver", "If you would rather have help"))
            return out

        return [
            r(
                r"\b(drink|thirsty|water|phone|floor|fell|dropped|toilet|pain|nurse|help)\b",
                "caregiver",
                lambda q: [_home("call_caregiver", "The arm cannot do that safely, so I can call for help")],
            ),
            r(r"\b(hungry|snack|eat|food|lunch)\b", "manipulate", snack),
            r(
                r"\b(lights? off|too bright|sleep)\b",
                "home",
                lambda q: [_home("lights_off", "Turns the room lights off")],
            ),
            r(r"\b(dark|brighter|lights?)\b", "home", lambda q: [_home("lights_on", "Turns the room lights on")]),
            r(r"\bclose\b.*\bdrawer\b", "manipulate", lambda q: self._pick(q, ["close_middle_drawer"], "Closes it")),
            r(
                r"\bdrawer\b",
                "manipulate",
                lambda q: self._pick(q, ["open_middle_drawer", "bowl_in_top_drawer"], "Drawer options"),
            ),
            r(
                r"\b(bottle|wine|tidy|put away)\b",
                "manipulate",
                lambda q: self._pick(q, ["bottle_on_rack", "bottle_on_cabinet"], "Puts the bottle away"),
            ),
            r(
                r"\b(stove|cook|heat)\b",
                "manipulate",
                lambda q: self._pick(q, ["stove_on", "bowl_on_stove"], "You asked for the stove"),
            ),
            r(
                r"\b(bring|give me|to me)\b",
                "manipulate",
                lambda q: self._pick(q, ["bowl_to_user"], "Brings the bowl to you"),
            ),
            r(
                r"\bplate\b",
                "manipulate",
                lambda q: self._pick(q, ["bowl_on_plate", "plate_to_stove_front"], "Plate options"),
            ),
            r(
                r"\b(bowl|cheese)\b",
                "manipulate",
                lambda q: self._pick(q, ["cheese_in_bowl", "bowl_on_plate", "bowl_to_user"], "Bowl options"),
            ),
        ]

    def _top_feasible(self, req: PlanRequest, k: int) -> list[str]:
        rows = [f for f in req.feasible if f.kind != "home" and f.risk != "high"]
        rows.sort(key=lambda f: -(f.success_alone.wilson_lo if f.success_alone else 0.0))
        return [f.skill_id for f in rows[:k]]

    def _by_rules(self, req: PlanRequest) -> PlannerOutput:
        if req.mode == "recovery" and req.failure is not None:
            opts = self._pick(req, [req.failure.skill_id], "Try that again?")
            opts.append(_home("call_caregiver", "Or I can call for help"))
            return PlannerOutput(intent="manipulate", clarify_question=None, options=opts)
        if req.input == MENU_NO_SPEECH:
            opts = [_skill(s, "") for s in self._top_feasible(req, 2)]
            opts.append(_home("call_caregiver", ""))
            return PlannerOutput(intent="none", clarify_question=None, options=opts)
        for pattern, intent, fn in self.rules:
            if pattern.search(req.input):
                opts = fn(req)[:3]
                if opts:
                    return PlannerOutput(intent=intent, clarify_question=None, options=opts)  # type: ignore[arg-type]
        opts = [_skill(s, "Is this what you meant?") for s in self._top_feasible(req, 2)]
        opts.append(_home("call_caregiver", "Or I can call for help"))
        return PlannerOutput(intent="clarify", clarify_question="Which of these did you mean?", options=opts)
