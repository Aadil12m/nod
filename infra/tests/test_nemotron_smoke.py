"""docs/runbooks/nemotron_smoke.py (gate G3) against a local OpenAI-compatible stub server."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import REPO, run

SMOKE = REPO / "docs" / "runbooks" / "nemotron_smoke.py"


def _load():
    spec = importlib.util.spec_from_file_location("nemotron_smoke", SMOKE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


smoke = _load()


def test_schema_matches_nod_protocol():
    proto = pytest.importorskip("nod_protocol")
    ids, plans = smoke.FEASIBLE, smoke.PLAN_IDS
    assert smoke.planner_schema(ids, plans) == proto.planner_response_format(ids, plans)
    assert smoke.planner_schema(ids, []) == proto.planner_response_format(ids, [])


GOOD = {"intent": "manipulate", "clarify_question": None, "options": [
    {"kind": "plan", "skill_id": None, "plan": ["bowl_on_plate", "cheese_in_bowl"], "home_action": None,
     "why": "You said you are hungry"},
    {"kind": "skill", "skill_id": "cheese_in_bowl", "plan": None, "home_action": None, "why": "Quick"},
    {"kind": "home", "skill_id": None, "plan": None, "home_action": "call_caregiver", "why": "Help"}]}


def test_check_output_accepts_valid_and_matches_protocol_model():
    assert smoke.check_output(GOOD, smoke.FEASIBLE, smoke.PLAN_IDS) == []
    proto = pytest.importorskip("nod_protocol")
    proto.PlannerOutput.model_validate(GOOD)


@pytest.mark.parametrize("mutate,expect", [
    (lambda o: o.update(intent="dance"), "intent"),
    (lambda o: o["options"][1].update(skill_id="stove_on"), "not feasible"),
    (lambda o: o["options"][0].update(plan=["bowl_on_plate"]), "plan"),
    (lambda o: o["options"][2].update(home_action="open_door"), "home_action"),
    (lambda o: o["options"][1].update(why="x" * 81), "why"),
    (lambda o: o["options"].append(dict(o["options"][1])), "at most 3"),
    (lambda o: o["options"][1].pop("plan"), "wrong keys"),
    (lambda o: o.update(clarify_question="y" * 61), "clarify_question"),
])
def test_check_output_rejects(mutate, expect):
    out = json.loads(json.dumps(GOOD))
    mutate(out)
    errs = smoke.check_output(out, smoke.FEASIBLE, smoke.PLAN_IDS)
    assert any(expect in e for e in errs), errs


def test_percentile_nearest_rank():
    xs = [float(i) for i in range(1, 11)]
    assert smoke.percentile(xs, 50) == 5.0 and smoke.percentile(xs, 95) == 10.0


class _Stub(BaseHTTPRequestHandler):
    """Minimal Token Factory stand-in: /models and /chat/completions in all three modes."""

    reply: dict = GOOD
    seen: list = []

    def log_message(self, *a):  # keep pytest output clean
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"object": "list", "data": [
            {"id": "nvidia/Nemotron-3_5-Lightning", "object": "model", "created": 0, "owned_by": "nvidia"},
            {"id": "meta/other", "object": "model", "created": 0, "owned_by": "x"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(req)
        content = json.dumps(type(self).reply)
        msg = {"role": "assistant", "content": content}
        if "tools" in req:
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "propose_options", "arguments": content}}]}
        self._send({"id": "x", "object": "chat.completion", "created": 0, "model": req["model"],
                    "choices": [{"index": 0, "finish_reason": "stop", "message": msg}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


@pytest.fixture
def stub():
    pytest.importorskip("openai")

    def start(reply):
        handler = type("H", (_Stub,), {"reply": reply, "seen": []})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}/v1/", handler

    servers: list = []
    yield start
    for s in servers:
        s.shutdown()


def test_end_to_end_valid_replies(stub, tmp_path):
    url, handler = stub(GOOD)
    out = tmp_path / "g3.jsonl"
    cp = run([sys.executable, str(SMOKE), "--base-url", url, "--out", str(out)], env={"NEBIUS_API_KEY": "test"},
             timeout=120)
    assert cp.returncode == 0, cp.stderr
    for model in ("nvidia/Nemotron-3_5-Lightning", "nvidia/nemotron-3-super-120b-a12b"):
        for mode in ("json_schema", "tools", "json_object"):
            assert f"{model} {mode}: valid 10/10" in cp.stdout
    assert "['nvidia/Nemotron-3_5-Lightning']" in cp.stdout  # model listing filters to Nemotron
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 60 and all(r["ok"] for r in rows)
    # Every request switches thinking off (design 7.1) and json_schema requests carry the 7.4 schema.
    assert all(r["chat_template_kwargs"] == {"enable_thinking": False} for r in handler.seen)
    js = next(r for r in handler.seen if r.get("response_format", {}).get("type") == "json_schema")
    assert js["response_format"]["json_schema"]["name"] == "nod_plan"
    tools = next(r for r in handler.seen if "tools" in r)
    assert tools["tool_choice"] == "required"


def test_end_to_end_invalid_replies_counted(stub):
    bad = json.loads(json.dumps(GOOD))
    bad["options"][1]["skill_id"] = "stove_on"
    url, _ = stub(bad)
    cp = run([sys.executable, str(SMOKE), "--base-url", url, "--modes", "json_schema", "--repeat", "1",
              "--models", "nvidia/Nemotron-3_5-Lightning"], env={"NEBIUS_API_KEY": "test"}, timeout=60)
    assert cp.returncode == 0, cp.stderr
    assert "nvidia/Nemotron-3_5-Lightning json_schema: valid 0/5" in cp.stdout
    assert "not feasible" in cp.stdout


def test_requires_api_key():
    cp = run([sys.executable, str(SMOKE)], env={})
    assert cp.returncode == 2 and "NEBIUS_API_KEY" in cp.stderr
