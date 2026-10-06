"""nod_eval on the CPU fakes: run (alone + chains), publish (StatsFile), cosmos_agree (fake verifier)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nod_protocol import StatsFile, load_stats, wilson_lower

from nod_eval.chains import CHAINS, available_prefix, parse_chains
from nod_eval.cosmos_agree import FakeVerifier, VllmVerifier, cohen_kappa, parse_reply, score
from nod_eval.publish import build_stats, publish
from nod_eval.run import EvalConfig, evaluate
from nod_sim.fakes import FakeEnv, FakePolicy

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "registry" / "skills.yaml"


def test_parse_chains():
    assert parse_chains("A,B") == {"A": CHAINS["A"], "B": CHAINS["B"]}
    assert parse_chains("x=bowl_on_plate+cheese_in_bowl") == {"x": ["bowl_on_plate", "cheese_in_bowl"]}
    assert parse_chains("") == {}
    with pytest.raises(ValueError):
        parse_chains("Z")
    assert available_prefix(["a", "b", "c"], {"a", "c"}) == ["a"]


@pytest.fixture(scope="module")
def scripted_run(tmp_path_factory):
    """Scripted actor on FakeEnv: 3 skills alone and both chains, 2 episodes each, with frames."""
    out = tmp_path_factory.mktemp("eval")
    cfg = EvalConfig(episodes=2, chains=parse_chains("A,B"), actor="scripted", frame_w=96, frame_h=72,
                     skills=["bowl_on_plate", "cheese_in_bowl", "stove_on"])
    meta = evaluate(FakeEnv(), cfg, out, registry=REGISTRY)
    return out, meta


def test_run_writes_episodes_and_meta(scripted_run):
    out, meta = scripted_run
    rows = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    assert meta["skills"] == ["bowl_on_plate", "cheese_in_bowl"]
    assert "stove_on" in meta["skipped"]  # no scripted controller for it
    assert meta["chains"] == {"A": CHAINS["A"], "B": CHAINS["B"]}
    alone = [r for r in rows if r["kind"] == "alone"]
    assert len(alone) == 4 and all(r["success"] and r["effects_ok"] and r["pre_ok"] for r in alone)
    assert [r["init_index"] for r in alone[:2]] == [0, 1]
    summaries = [r for r in rows if r["kind"] == "chain_summary"]
    assert len(summaries) == 4 and all(s["success"] for s in summaries)
    steps_b = [r for r in rows if r["kind"] == "chain" and r["chain"] == "B" and r["episode"] == 0]
    assert [r["skill_id"] for r in steps_b] == CHAINS["B"]
    assert steps_b[0]["home"] is None and steps_b[1]["home"]["success"]
    # chain B step 2 starts with the bowl on the plate: still within cheese_in_bowl's preconditions
    assert steps_b[1]["atoms_pre"]["on(akita_black_bowl_1,plate_1)"] and steps_b[1]["pre_ok"]
    # recordings and frames are where the rows say
    r = steps_b[2]
    assert (out / r["record_dir"] / "manifest.json").exists()
    assert (out / r["frames"]["post"]["frontview"]).read_bytes()[:2] == b"\xff\xd8"


def test_publish_matches_statsfile(scripted_run, tmp_path):
    out, _ = scripted_run
    stats = publish(out, tmp_path / "stats.json", job_id="job1", git_sha="abc")
    again = load_stats(tmp_path / "stats.json")
    assert isinstance(again, StatsFile) and not again.placeholder
    s = again.skills["cheese_in_bowl"]
    assert (s.alone.k, s.alone.n) == (2, 2) and s.alone.wilson_lo == round(wilson_lower(2, 2), 4)
    assert (s.in_chain.k, s.in_chain.n) == (4, 4)  # step 3 of A and step 2 of B, 2 episodes each
    assert "bowl_on_plate" in again.skills and again.skills["bowl_on_plate"].in_chain is None  # only ever step 1
    a = again.chains["A"]
    assert a.skill_ids == CHAINS["A"] and (a.k, a.n) == (2, 2) and len(a.steps) == 4
    assert a.median_s and a.median_s > again.skills["cheese_in_bowl"].alone.median_s
    assert "scripted" in (again.note or "") and again.job_id == "job1" and again.git_sha == "abc"
    assert stats.dump_json() == (tmp_path / "stats.json").read_text()


def test_build_stats_failures_and_conditional_steps():
    rows = [
        {"kind": "alone", "skill_id": "s", "success": True, "sim_s": 4.0},
        {"kind": "alone", "skill_id": "s", "success": False, "sim_s": 20.0},
        {"kind": "chain", "chain": "X", "step": 0, "skill_id": "s", "success": True, "sim_s": 4.0},
        {"kind": "chain", "chain": "X", "step": 1, "skill_id": "t", "success": False, "sim_s": 20.0},
        {"kind": "chain_summary", "chain": "X", "skill_ids": ["s", "t", "u"], "success": False, "sim_s": 24.0},
        {"kind": "chain", "chain": "X", "step": 0, "skill_id": "s", "success": False, "sim_s": 20.0},
        {"kind": "chain_summary", "chain": "X", "skill_ids": ["s", "t", "u"], "success": False, "sim_s": 20.0},
    ]
    st = build_stats(rows, {"actor": "policy"}, at="2026-10-06T00:00:00Z")
    assert (st.skills["s"].alone.k, st.skills["s"].alone.n, st.skills["s"].alone.median_s) == (1, 2, 4.0)
    assert st.skills["t"].in_chain.k == 0 and st.skills["t"].alone is None
    x = st.chains["X"]
    assert (x.k, x.n, x.median_s) == (0, 2, None)
    assert [(b.k, b.n) for b in x.steps] == [(1, 2), (0, 1), (0, 0)]
    assert st.note is None


def test_policy_actor_with_fake_policy_and_missing_checkpoint(tmp_path):
    """Policy actor: skills whose checkpoint has no server are skipped; chain B is truncated."""
    from nod_sim.scripted import scripted_skill

    env = FakeEnv()

    class ScriptAsPolicy(FakePolicy):
        """Closed-loop stand-in for GR00T: replays the scripted controller one step per call."""

        def __init__(self):
            super().__init__(lambda obs: None, horizon=1)
            self.gen = None

        def get_action(self, obs):
            import numpy as np

            lang = obs["annotation.human.action.task_description"]
            sid = {"put the bowl on the plate": "bowl_on_plate",
                   "put the cream cheese in the bowl": "cheese_in_bowl"}[lang]
            if self.gen is None or self.lang != lang:
                self.gen, self.lang = scripted_skill(sid)(env), lang
            try:
                a = next(self.gen)
            except StopIteration:
                a = np.zeros(7, np.float32)
                a[6] = 1.0
            return np.asarray(a, np.float32)[None]

    pol = ScriptAsPolicy()
    cfg = EvalConfig(episodes=1, chains=parse_chains("B"), skills=["bowl_on_plate", "cheese_in_bowl", "bowl_to_user"],
                     frames="none", record=False, max_steps=300)
    meta = evaluate(env, cfg, tmp_path, registry=REGISTRY, policies={"GR00T-N1.7-LIBERO/libero_goal": pol},
                    runner_kw={"policy_timeout_ms": 5000})
    assert "bowl_to_user" in meta["skipped"] and meta["chains"] == {"B": ["bowl_on_plate", "cheese_in_bowl"]}
    assert "truncated" in meta["chain_notes"]["B"]
    rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert all(r["actor"] == "policy" for r in rows if r["kind"] != "chain_summary")
    st = build_stats(rows, meta)
    assert st.skills["bowl_on_plate"].alone.k == 1
    assert "truncated" in st.note


def test_cohen_kappa_and_parse_reply():
    assert cohen_kappa(10, 10, 0, 0) == 1.0
    assert cohen_kappa(5, 5, 5, 5) == 0.0
    assert cohen_kappa(0, 0, 0, 0) is None
    assert parse_reply('<think>hmm</think>{"answer": "yes", "reason": "on plate"}').answer == "yes"
    assert parse_reply("No, the bowl is on the table").answer == "no"
    assert parse_reply("I cannot tell").answer == "unsure"


def test_cosmos_agree_with_fake_verifier(scripted_run):
    out, _ = scripted_run
    perfect = score(out, REGISTRY, FakeVerifier())
    o = perfect["overall"]
    assert o["n"] > 0 and o["accuracy"] == 1.0 and o["false_yes"] == 0 and o["coverage"] == 1.0
    assert set(perfect["by_phase"]) == {"post", "pre"}
    assert "bowl_on_plate" in perfect["by_skill"]
    noisy = score(out, REGISTRY, FakeVerifier(error_rate=0.5, unsure_rate=0.2, seed=3))
    assert noisy["overall"]["accuracy"] < 1.0 and noisy["overall"]["unsure"] > 0
    rows = [json.loads(line) for line in (out / "cosmos_checks.jsonl").read_text().splitlines()]
    assert len(rows) == o["n"] and {"question", "answer", "oracle", "agree"} <= set(rows[0])


def test_vllm_verifier_against_a_local_http_server(scripted_run):
    """The stdlib client against a tiny OpenAI-style server: schema rejected once, then plain text."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            if "response_format" in body:
                self.send_response(400)
                self.end_headers()
                return
            data = json.dumps({"choices": [{"message": {"content": "Yes. The bowl is on the plate."}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        v = VllmVerifier(f"http://127.0.0.1:{srv.server_port}/v1", timeout_s=5)
        r = v.ask([b"\xff\xd8fake"], "Is the black bowl resting on the plate?")
        assert r.answer == "yes" and not v.use_schema and len(seen) == 2
        content = seen[1]["messages"][0]["content"]
        assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert content[-1]["text"].endswith('Reply with JSON {"answer": "yes"|"no"|"unsure", "reason": "..."}')
    finally:
        srv.shutdown()
