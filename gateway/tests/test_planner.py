"""Planners: FakePlanner (canned + rules) and NemotronPlanner against fake Token Factory responses."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from conftest import make_world, view
from nod_protocol import MENU_NO_SPEECH, FailureInfo, FeasibleSkill, PlanRequest

from nod_gateway.config import Config
from nod_gateway.planner.base import PlannerError, build_context, extract_json, humanize_atom, parse_output
from nod_gateway.planner.fake import FakePlanner
from nod_gateway.planner.nemotron import NemotronPlanner
from nod_gateway.validator import feasible_skills, validate_proposal

CANNED = Path(__file__).resolve().parents[1] / "nod_gateway" / "planner" / "canned"

GOOD = {
    "intent": "manipulate",
    "clarify_question": None,
    "options": [
        {"kind": "skill", "skill_id": "cheese_in_bowl", "plan": None, "home_action": None, "why": "Quick snack"}
    ],
}


def request(reg, text: str, mode: str = "fast", world=None, failure=None) -> PlanRequest:
    world = world or make_world(reg)
    feas = feasible_skills(world, reg)
    feasible = [
        FeasibleSkill(
            skill_id=f.skill_id,
            kind=reg[f.skill_id].kind,
            label=reg[f.skill_id].label,
            success_alone=f.success_alone,
            success_in_chain=f.success_in_chain,
            est_s=f.est_s,
            risk=reg[f.skill_id].risk,
        )
        for f in feas
        if f.ok
    ]
    return PlanRequest(
        mode=mode,
        input=text,
        time="2026-10-05T12:00:00+00:00",
        world=world,
        feasible=feasible,
        plan_ids=reg.plan_ids(),
        failure=failure,
    )


# ---------------------------------------------------------------- base helpers


def test_humanize_atom():
    assert humanize_atom("on(akita_black_bowl_1,plate_1)") == "the bowl is on the plate"
    assert humanize_atom("open(wooden_cabinet_1_middle_region)") == "the middle drawer is open"
    assert humanize_atom("turnon(flat_stove_1)") == "the stove is switched on"
    assert humanize_atom("weird") == "weird"


def test_context_has_7_3_fields(reg):
    world = make_world(reg, ("on(akita_black_bowl_1,plate_1)",))
    ctx = build_context(request(reg, "hi", world=world))
    assert set(ctx) >= {"profile", "time", "world", "feasible", "infeasible", "history", "input"}
    assert ctx["world"] == ["the bowl is on the plate"]  # true atoms only, in plain words
    assert all({"id", "label", "success_alone", "success_in_chain", "est_s", "risk"} <= set(f) for f in ctx["feasible"])


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(GOOD),
        "```json\n" + json.dumps(GOOD) + "\n```",
        "<think>hmm</think>" + json.dumps(GOOD),
        "Here you go: " + json.dumps(GOOD) + " thanks",
    ],
)
def test_parse_output_tolerates_wrapping(text):
    assert parse_output(text).options[0].skill_id == "cheese_in_bowl"
    assert extract_json(text).startswith("{")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not json",
        json.dumps({"intent": "x", "options": []}),
        json.dumps({**GOOD, "extra": 1}),
        json.dumps({**GOOD, "options": GOOD["options"] * 4}),
    ],
)
def test_parse_output_r1_failures(text):
    with pytest.raises(PlannerError, match="R1"):
        parse_output(text)


# ---------------------------------------------------------------- FakePlanner


async def test_fake_canned_exact_transcript(reg):
    fp = FakePlanner(CANNED)
    out = await fp.propose(request(reg, "I'm a bit hungry"))
    assert out.options[0].plan == ["bowl_on_plate", "cheese_in_bowl", "bowl_to_user"]
    res = await fp.propose_ex(request(reg, "i'm A BIT hungry!!"))
    assert res.mechanism == "canned:hungry" and res.source == "fake"


async def test_fake_canned_regex(reg):
    out = await FakePlanner(CANNED).propose(request(reg, "my phone is on the floor"))
    assert [o.home_action for o in out.options] == ["call_caregiver"]


async def test_fake_rules_only_name_feasible_ids(reg):
    # The cheese is already in the bowl: the snack rule may not name cheese_in_bowl or the snack plan.
    world = make_world(reg, ("on(cream_cheese_1,akita_black_bowl_1)",))
    out = await FakePlanner(Path("/nonexistent")).propose(request(reg, "I want a snack", world=world))
    ids = [o.skill_id for o in out.options if o.kind == "skill"]
    assert "cheese_in_bowl" not in ids and all(o.kind != "plan" for o in out.options)
    assert out.options[-1].home_action == "call_caregiver"


@pytest.mark.parametrize(
    "text,expect",
    [
        ("it's too dark in here", ("home", "lights_on")),
        ("turn the lights off please", ("home", "lights_off")),
        ("open the drawer please", ("skill", "open_middle_drawer")),
        ("tidy up the wine", ("skill", "bottle_on_rack")),
        ("I need water", ("home", "call_caregiver")),
        ("bring it to me", ("skill", "bowl_to_user")),
    ],
)
async def test_fake_rules(reg, text, expect):
    out = await FakePlanner(Path("/nonexistent")).propose(request(reg, text))
    first = out.options[0]
    assert (first.kind, first.home_action or first.skill_id) == expect


async def test_fake_menu_and_clarify_and_recovery(reg):
    fp = FakePlanner(Path("/nonexistent"))
    menu = await fp.propose(request(reg, MENU_NO_SPEECH))
    assert menu.intent == "none" and len(menu.options) == 3
    unclear = await fp.propose(request(reg, "hmm"))
    assert unclear.intent == "clarify" and unclear.clarify_question
    rec = await fp.propose(
        request(reg, "x", mode="recovery", failure=FailureInfo(skill_id="bottle_on_rack", steps=400, oracle=False))
    )
    assert rec.options[0].skill_id == "bottle_on_rack" and rec.options[-1].home_action == "call_caregiver"


async def test_fake_is_deterministic(reg):
    a = await FakePlanner(CANNED).propose(request(reg, "open the drawer"))
    b = await FakePlanner(CANNED).propose(request(reg, "open the drawer"))
    assert a == b


async def test_fake_outage(reg):
    with pytest.raises(PlannerError):
        await FakePlanner(CANNED, fail=True).propose(request(reg, "hi"))


async def test_canned_unsafe_demo_is_filtered_by_validator(reg):
    out = await FakePlanner(CANNED).propose(request(reg, "warm up my snack"))
    res = validate_proposal(out, make_world(reg), reg, view(mode="PLANNING"))
    assert [o.skill_ids for o in res.options] == [["stove_on"]] and res.options[0].locked
    assert sorted(d.rule for d in res.dropped) == ["R10_plan_check", "R2_known_skill"]


def test_every_canned_file_parses():
    fp = FakePlanner(CANNED)
    assert len(fp.canned) >= 5
    assert all(not c.recorded for c in fp.canned)  # hand-written until G3 produces real traces


# ---------------------------------------------------------------- NemotronPlanner (fake HTTP)


def completion(content: str | None = None, tool_args: str | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_args is not None:
        msg["tool_calls"] = [
            {"id": "c1", "type": "function", "function": {"name": "propose_options", "arguments": tool_args}}
        ]
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": msg}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class Fake:
    """httpx transport that records requests and replies from a script."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.urls: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.urls.append(str(request.url))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        if isinstance(r, httpx.Response):
            return r
        return httpx.Response(200, json=r)


def planner(fake: Fake, **cfg) -> NemotronPlanner:
    c = Config(mode="live", tf_api_key="k", **cfg)
    return NemotronPlanner(c, httpx.AsyncClient(transport=httpx.MockTransport(fake)))


async def test_nemotron_fast_request_shape(reg):
    fake = Fake(completion(json.dumps(GOOD)))
    p = planner(fake)
    res = await p.propose_ex(request(reg, "I'm hungry"))
    assert res.output.options[0].skill_id == "cheese_in_bowl" and res.mechanism == "json_schema"
    body = fake.requests[0]
    assert fake.urls[0] == "https://api.tokenfactory.nebius.com/v1/chat/completions"
    assert body["model"] == "nvidia/Nemotron-3_5-Lightning"
    assert body["temperature"] == 0.2 and body["max_tokens"] == 400
    assert body["chat_template_kwargs"] == {"enable_thinking": False}  # extra_body is merged into the body
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["name"] == "nod_plan" and rf["json_schema"]["strict"]
    enum = rf["json_schema"]["schema"]["properties"]["options"]["items"]["properties"]["skill_id"]["enum"]
    feasible = {f.skill_id for f in feasible_skills(make_world(reg), reg) if f.ok and f.kind != "home"}
    assert set(enum) - {None} == feasible  # per-request enum from feasible_skills()
    assert "close_middle_drawer" not in enum and "return_home" not in enum
    assert body["messages"][0]["role"] == "system" and "You plan for Nod" in body["messages"][0]["content"]
    ctx = json.loads(body["messages"][1]["content"])
    assert ctx["input"] == "I'm hungry"


async def test_nemotron_plan_route_uses_super_with_thinking(reg):
    fake = Fake(completion(json.dumps(GOOD)))
    await planner(fake).propose(request(reg, "set up a snack", mode="plan"))
    body = fake.requests[0]
    assert fake.urls[0].startswith("https://api.tokenfactory.us-central1.nebius.com/v1/")
    assert body["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "low_effort": True}
    assert body["temperature"] == 0.6 and body["max_tokens"] == 1500


async def test_nemotron_falls_back_to_tools_on_400_and_remembers(reg):
    fake = Fake(
        httpx.Response(400, json={"error": {"message": "json_schema not supported"}}),
        completion(None, tool_args=json.dumps(GOOD)),
        completion(None, tool_args=json.dumps(GOOD)),
    )
    p = planner(fake)
    res = await p.propose_ex(request(reg, "hi"))
    assert res.mechanism == "tools"
    body = fake.requests[1]
    assert body["tool_choice"] == "required" and body["tools"][0]["function"]["name"] == "propose_options"
    assert "response_format" not in body
    await p.propose(request(reg, "again"))
    assert "tools" in fake.requests[2]  # sticky: no json_schema attempt the second time


async def test_nemotron_schema_failure_retries_json_object_once(reg):
    fake = Fake(completion("I think you want cheese"), completion(json.dumps(GOOD)))
    res = await planner(fake).propose_ex(request(reg, "hi"))
    assert res.mechanism == "json_object"
    body = fake.requests[1]
    assert body["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in body["messages"][0]["content"]


async def test_nemotron_all_mechanisms_fail(reg):
    fake = Fake(completion("nope"), completion("still nope"))
    with pytest.raises(PlannerError, match="all mechanisms failed"):
        await planner(fake).propose(request(reg, "hi"))
    assert len(fake.requests) == 2  # json_schema, then exactly one json_object retry


async def test_nemotron_timeout_backs_off(reg):
    fake = Fake(httpx.ReadTimeout("slow"))
    p = planner(fake)
    with pytest.raises(PlannerError, match="timeout"):
        await p.propose(request(reg, "hi"))
    with pytest.raises(PlannerError, match="backing off"):
        await p.propose(request(reg, "hi"))  # fails fast: the menu shows at once
    assert len(fake.requests) == 1


async def test_nemotron_429(reg):
    fake = Fake(httpx.Response(429, json={"error": {"message": "slow down"}}))
    with pytest.raises(PlannerError, match="429"):
        await planner(fake).propose(request(reg, "hi"))


async def test_nemotron_no_key(reg):
    p = NemotronPlanner(Config(mode="live", tf_api_key=None), httpx.AsyncClient())
    with pytest.raises(PlannerError, match="API key"):
        await p.propose(request(reg, "hi"))


async def test_nemotron_forced_mechanism(reg):
    fake = Fake(completion(None, tool_args=json.dumps(GOOD)))
    res = await planner(fake, planner_structured="tools").propose_ex(request(reg, "hi"))
    assert res.mechanism == "tools" and len(fake.requests) == 1


def test_key_from_nebius_env(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "abc")
    assert Config().tf_api_key == "abc"
