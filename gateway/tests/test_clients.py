"""ASR, Cosmos and Home clients (live ones against fakes), recorder and metrics."""

from __future__ import annotations

import base64
import json
import sys
import types

import httpx
import numpy as np
import pytest
from conftest import make_world
from nod_protocol import CosmosCheck, Frame

from nod_gateway.asr.base import ASRError
from nod_gateway.asr.fake import FakeASR
from nod_gateway.asr.parakeet import ParakeetASR
from nod_gateway.config import Config
from nod_gateway.cosmos.base import decision_for, question_with_format
from nod_gateway.cosmos.fake import FakeVerifier
from nod_gateway.cosmos.vllm import VllmVerifier, data_uri, parse_reply
from nod_gateway.home.fake import FakeHome
from nod_gateway.home.hass import HassHome
from nod_gateway.metrics import Health, Histogram, MetricsHub
from nod_gateway.mock_runner import PLACEHOLDER_JPEG
from nod_gateway.recorder import Recorder

# ---------------------------------------------------------------- ASR


async def test_fake_asr_cycles():
    asr = FakeASR(["a", "b"])
    assert [(await asr.transcribe(b"\0\0")).text for _ in range(3)] == ["a", "b", "a"]
    with pytest.raises(ASRError):
        await asr.transcribe(b"\0")


async def test_parakeet_lazy_import_and_call(monkeypatch):
    calls = {}

    class Model:
        def recognize(self, audio, sample_rate):
            calls["audio"], calls["sr"] = audio, sample_rate
            return "  I'm a bit hungry "

    fake = types.ModuleType("onnx_asr")

    def load_model(name, **kw):
        calls["load"] = (name, kw)
        return Model()

    fake.load_model = load_model
    monkeypatch.setitem(sys.modules, "onnx_asr", fake)
    asr = ParakeetASR()
    pcm = (np.array([0, 16384, -32768], dtype="<i2")).tobytes()
    tr = await asr.transcribe(pcm, 16000)
    assert tr.text == "I'm a bit hungry" and tr.asr_ms is not None
    assert calls["load"] == ("nemo-parakeet-tdt-0.6b-v3", {"quantization": "int8"})
    assert calls["sr"] == 16000 and calls["audio"].dtype == np.float32
    assert np.allclose(calls["audio"], [0.0, 0.5, -1.0])


async def test_parakeet_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnx_asr", None)
    with pytest.raises(ASRError, match="onnx-asr"):
        await ParakeetASR().transcribe(b"\0\0" * 10)


# ---------------------------------------------------------------- Cosmos


def check(phase="post", skill="bowl_on_plate") -> CosmosCheck:
    return CosmosCheck(check_id="k_1", phase=phase, skill_id=skill, question="Is the black bowl resting on the plate?")


@pytest.mark.parametrize(
    "text,answer",
    [
        ('{"answer": "yes", "reason": "The bowl sits on the plate."}', "yes"),
        ('<think>looking</think>{"answer": "no", "reason": "empty plate"}', "no"),
        ("Yes, it is on the plate.", "yes"),
        ("no.", "no"),
        ("Unsure: the view is blocked", "unsure"),
        ("maybe?", "unsure"),
    ],
)
def test_parse_reply(text, answer):
    assert parse_reply(text).answer == answer


def test_decision_and_question():
    assert decision_for("post", "yes") == "done" and decision_for("post", "no") == "failed"
    assert decision_for("post", "unsure") == "uncertain" and decision_for("pre", "no") == "proceed"
    assert question_with_format("Q?").endswith('Reply with JSON {"answer": "yes"|"no"|"unsure", "reason": "..."}')


def cosmos_reply(content: str) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
    }


async def test_vllm_request_shape_and_schema_fallback():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "guided decoding unsupported"}})
        return httpx.Response(200, json=cosmos_reply("Yes. The bowl is on the plate."))

    cfg = Config(mode="live", cosmos_url="http://127.0.0.1:8001/v1")
    v = VllmVerifier(cfg, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    frame = Frame(camera="frontview", w=8, h=8, jpeg=PLACEHOLDER_JPEG, step=3)
    res = await v.check(check(), [frame, frame])
    assert res.answer == "yes" and res.decision == "done" and res.check_id == "k_1"
    first = seen[0]
    assert first["model"] == "nvidia/Cosmos-Reason2-8B" and first["temperature"] == 0 and first["max_tokens"] == 128
    assert first["response_format"]["type"] == "json_schema"
    content = first["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image_url", "image_url", "text"]
    url = content[0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,") and base64.b64decode(url.split(",", 1)[1]) == PLACEHOLDER_JPEG
    assert "response_format" not in seen[1]
    await v.check(check())  # remembers: no schema attempt any more
    assert "response_format" not in seen[2] and len(seen) == 3


async def test_vllm_errors_raise():
    from nod_gateway.cosmos.base import VerifierError

    def handler(req):
        raise httpx.ConnectError("down")

    v = VllmVerifier(
        Config(mode="live", cosmos_url="http://x/v1"), httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(VerifierError):
        await v.check(check())


def test_data_uri():
    assert data_uri(Frame(camera="c", w=1, h=1, jpeg=b"\xff\xd8")).endswith(base64.b64encode(b"\xff\xd8").decode())


async def test_fake_verifier_follows_oracle(reg):
    world = {"w": make_world(reg)}
    v = FakeVerifier(reg, lambda: world["w"])
    assert (await v.check(check("pre"))).answer == "yes"
    assert (await v.check(check("post"))).answer == "no"
    world["w"] = make_world(reg, ("on(akita_black_bowl_1,plate_1)",))
    assert (await v.check(check("post"))).answer == "yes"
    assert (await v.check(check("pre"))).answer == "no"
    v.script = ["unsure"]
    assert (await v.check(check("post"))).answer == "unsure"
    v2 = FakeVerifier(reg, lambda: None, recorded={("bowl_on_plate", "post"): ("no", "recorded")})
    r = await v2.check(check("post"))
    assert r.answer == "no" and r.reason == "recorded"


# ---------------------------------------------------------------- Home


async def test_hass_requests():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, str(req.url), req.headers["authorization"], json.loads(req.content)))
        return httpx.Response(200, json=[])

    cfg = Config(mode="live", hass_url="http://ha:8123/", hass_token="tok")
    h = HassHome(cfg, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert (await h.call("lights_on")).ok
    assert (await h.call("lights_off")).ok
    assert (await h.call("call_caregiver")).ok
    assert seen[0] == (
        "POST",
        "http://ha:8123/api/services/light/turn_on",
        "Bearer tok",
        {"entity_id": "light.ceiling_lights"},
    )
    assert seen[1][1].endswith("/api/services/light/turn_off")
    assert seen[2][1] == "http://ha:8123/api/services/notify/persistent_notification"
    assert seen[2][3]["title"] == "Nod"


async def test_hass_failure_reported():
    h = HassHome(
        Config(mode="live", hass_url="http://ha:8123", hass_token="t"),
        httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(401, text="no"))),
    )
    r = await h.call("lights_on")
    assert not r.ok and "401" in r.detail and h.last_error


def test_hass_needs_config():
    with pytest.raises(ValueError):
        HassHome(Config(mode="live"), httpx.AsyncClient())


async def test_fake_home():
    h = FakeHome()
    assert (await h.call("lights_on")).ok and h.lights_on
    assert (await h.call("lights_off")).ok and not h.lights_on
    assert (await h.call("call_caregiver")).detail.startswith("Caregiver")
    assert not (await h.call("dance")).ok


# ---------------------------------------------------------------- recorder, metrics, health


def test_recorder_roundtrip(tmp_path):
    r = Recorder(tmp_path)
    r.start("r_1", "run", skill_id="bowl_on_plate")
    r.event("r_1", "run", {"phase": "executing"})
    r.file("r_1", "planner_trace.json", {"model": "m"})
    r.update("r_1", outcome="done")
    assert (tmp_path / "r_1" / "manifest.json").exists()
    assert json.loads((tmp_path / "r_1" / "events.jsonl").read_text().splitlines()[0])["data"] == {"phase": "executing"}
    assert r.list()[0]["outcome"] == "done"
    fresh = Recorder(tmp_path)  # read back from disk
    got = fresh.get("r_1")
    assert got["outcome"] == "done" and got["events"][0]["kind"] == "run"
    assert got["files"]["planner_trace.json"] == {"model": "m"}
    assert fresh.get("../etc") is None and fresh.get("missing") is None


def test_recorder_memory_only_and_bounded():
    r = Recorder(None, keep=3)
    for i in range(5):
        r.start(f"r_{i}", "run")
    assert [x["id"] for x in r.list()] == ["r_4", "r_3", "r_2"]


def test_histogram_and_metrics():
    h = Histogram()
    for v in range(1, 101):
        h.add(v)
    h.add(None)
    h.add(float("nan"))
    assert h.pct(50) == 50 and h.pct(95) == 95 and h.summary()["n"] == 100
    m = MetricsHub()
    m.add("estop_ms", 60)
    m.add("llm_ms", 640)
    msg = m.message()
    assert msg.estop_ms_p50 == 60 and msg.llm_ms_p50 == 640 and msg.view_lag_ms is None


def test_health():
    h = Health()
    h.error("sim", "down")
    h.ok("sim", scene_id="x")
    v = h.view()["sim"]
    assert v["ok"] and v["last_error"] == "down" and v["scene_id"] == "x"
