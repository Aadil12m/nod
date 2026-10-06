"""Score Cosmos verdicts against the oracle on the eval's recorded frames (design 4.2.5, 3.5.7).

    python -m nod_eval.cosmos_agree --in /mnt/results/<date> --registry registry/skills.yaml \
        --url http://127.0.0.1:8000/v1 --model nvidia/Cosmos-Reason2-8B

For every episode row of a `nod_eval.run` output that has frames, it asks the skill's registry
questions exactly as the live gateway does (same frames camera, same reply-format suffix):

* post: `cosmos.post` on the frames after the run; ground truth = the skill's effects hold
  (oracle atoms after the run, `effects_ok`).
* pre: `cosmos.pre` on the frames before the run; ground truth = the skill's preconditions
  hold (`pre_ok`). In chains this includes states the training demos never show.

Writes `cosmos_checks.jsonl` (one row per question) and `cosmos_agree.json`:
per skill and phase, and overall: n, yes/no/unsure counts, coverage (share answered yes or no),
accuracy over answered checks, `false_yes` (Cosmos says yes, oracle says no: for post checks
this is the dangerous "looks done but is not" case), `false_no`, and Cohen's kappa.

Verifiers: `vllm` (OpenAI-compatible chat completions with one image per frame, temperature 0,
json_schema response format with a plain-text retry, as gateway/nod_gateway/cosmos/vllm.py)
and `fake` (the oracle with seeded noise; tests and dry runs only).
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nod_protocol import CosmosReply, load_skills
from pydantic import ValidationError

from .publish import load_rows

_THINK = re.compile(r"<think>.*?</think>", re.S)
_LEAD = re.compile(r"^\W*(yes|no|unsure)\b", re.I)


def question_with_format(question: str) -> str:
    """Same suffix as the gateway (nod_gateway.cosmos.base.question_with_format)."""
    return question + ' Reply with JSON {"answer": "yes"|"no"|"unsure", "reason": "..."}'


def parse_reply(text: str) -> CosmosReply:
    """JSON {answer, reason} if possible, else a leading yes/no/unsure, else unsure."""
    t = _THINK.sub("", text or "").strip()
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        try:
            return CosmosReply.model_validate_json(t[start : end + 1])
        except (ValidationError, ValueError):
            pass
    m = _LEAD.match(t)
    if m:
        return CosmosReply(answer=m.group(1).lower(), reason=t[m.end():].strip(" .,:;-")[:200])  # type: ignore[arg-type]
    return CosmosReply(answer="unsure", reason=("unparsed: " + t)[:200])


class Verifier(Protocol):
    def ask(self, images: list[bytes], question: str, *, truth: bool | None = None) -> CosmosReply: ...


class VllmVerifier:
    """Blocking OpenAI-compatible client (stdlib only, so it runs in the LIBERO venv)."""

    def __init__(self, url: str, model: str = "nvidia/Cosmos-Reason2-8B", timeout_s: float = 60.0,
                 api_key: str = "EMPTY") -> None:
        self.url = url.rstrip("/") + "/chat/completions"
        self.model, self.timeout_s, self.api_key = model, timeout_s, api_key
        self.use_schema = True

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read())

    def ask(self, images: list[bytes], question: str, *, truth: bool | None = None) -> CosmosReply:
        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}}
                   for b in images]
        content.append({"type": "text", "text": question_with_format(question)})
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "temperature": 0, "max_tokens": 128}
        if self.use_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "cosmos_reply", "strict": True,
                "schema": {"type": "object", "additionalProperties": False, "required": ["answer", "reason"],
                           "properties": {"answer": {"type": "string", "enum": ["yes", "no", "unsure"]},
                                          "reason": {"type": "string", "maxLength": 200}}}}}
        try:
            resp = self._post(body)
        except urllib.error.HTTPError as exc:
            if exc.code == 400 and self.use_schema:  # schema not supported with this server/parser
                self.use_schema = False
                return self.ask(images, question)
            raise
        return parse_reply(resp["choices"][0]["message"].get("content") or "")


class FakeVerifier:
    """The oracle with seeded noise: flips `error_rate` of answers, says unsure for `unsure_rate`."""

    def __init__(self, error_rate: float = 0.0, unsure_rate: float = 0.0, seed: int = 0) -> None:
        self.error_rate, self.unsure_rate = error_rate, unsure_rate
        self.rng = random.Random(seed)

    def ask(self, images: list[bytes], question: str, *, truth: bool | None = None) -> CosmosReply:
        if truth is None or self.rng.random() < self.unsure_rate:
            return CosmosReply(answer="unsure", reason="fake")
        ans = truth if self.rng.random() >= self.error_rate else not truth
        return CosmosReply(answer="yes" if ans else "no", reason="fake")


# ---------------------------------------------------------------------------- scoring


@dataclass
class Tally:
    n: int = 0
    yes: int = 0
    no: int = 0
    unsure: int = 0
    tp: int = 0  # yes, oracle true
    tn: int = 0  # no, oracle false
    false_yes: int = 0  # yes, oracle false
    false_no: int = 0  # no, oracle true
    latency_ms: float = 0.0

    def add(self, answer: str, truth: bool, ms: float = 0.0) -> None:
        self.n += 1
        self.latency_ms += ms
        if answer == "unsure":
            self.unsure += 1
            return
        said = answer == "yes"
        self.yes += said
        self.no += not said
        if said and truth:
            self.tp += 1
        elif not said and not truth:
            self.tn += 1
        elif said:
            self.false_yes += 1
        else:
            self.false_no += 1

    def summary(self) -> dict:
        answered = self.yes + self.no
        acc = (self.tp + self.tn) / answered if answered else None
        return {"n": self.n, "yes": self.yes, "no": self.no, "unsure": self.unsure,
                "coverage": round(answered / self.n, 4) if self.n else None,
                "accuracy": None if acc is None else round(acc, 4),
                "false_yes": self.false_yes, "false_no": self.false_no,
                "kappa": cohen_kappa(self.tp, self.tn, self.false_yes, self.false_no),
                "mean_latency_ms": round(self.latency_ms / self.n, 1) if self.n else None}


def cohen_kappa(tp: int, tn: int, fp: int, fn: int) -> float | None:
    """Cohen's kappa between verifier (yes/no) and oracle (true/false) on answered checks."""
    n = tp + tn + fp + fn
    if n == 0:
        return None
    po = (tp + tn) / n
    p_yes = ((tp + fp) / n) * ((tp + fn) / n)
    p_no = ((tn + fn) / n) * ((tn + fp) / n)
    pe = p_yes + p_no
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return round((po - pe) / (1 - pe), 4)


def score(run_dir: Path, registry: str | Path, verifier: Verifier, *, phases: tuple[str, ...] = ("post", "pre"),
          limit: int | None = None) -> dict:
    specs = {s.skill_id: s for s in load_skills(registry)}
    rows = [r for r in load_rows(run_dir) if r.get("kind") in ("alone", "chain") and r.get("frames")]
    tallies: dict[tuple[str, str], Tally] = defaultdict(Tally)
    checks_path = run_dir / "cosmos_checks.jsonl"
    checks_path.write_text("")
    count = 0
    for r in rows:
        spec = specs.get(r["skill_id"])
        if spec is None or spec.cosmos is None:
            continue
        for phase in phases:
            question = getattr(spec.cosmos, phase, None)
            paths = (r["frames"].get(phase) or {}).values()
            if not question or not paths:
                continue
            truth = bool(r["effects_ok"] if phase == "post" else r["pre_ok"])
            images = [(run_dir / p).read_bytes() for p in paths]
            t0 = time.perf_counter()
            reply = verifier.ask(images, question, truth=truth)
            ms = (time.perf_counter() - t0) * 1000
            for key in ((r["skill_id"], phase), ("*", phase), ("*", "*")):
                tallies[key].add(reply.answer, truth, ms)
            with open(checks_path, "a") as f:
                f.write(json.dumps({"label": r["label"], "skill_id": r["skill_id"], "kind": r["kind"],
                                    "chain": r.get("chain"), "phase": phase, "question": question,
                                    "answer": reply.answer, "reason": reply.reason, "oracle": truth,
                                    "agree": None if reply.answer == "unsure" else (reply.answer == "yes") == truth,
                                    "latency_ms": round(ms, 1)}) + "\n")
            count += 1
            if limit and count >= limit:
                break
        if limit and count >= limit:
            break
    out = {"overall": tallies[("*", "*")].summary(),
           "by_phase": {ph: tallies[("*", ph)].summary() for ph in phases if ("*", ph) in tallies},
           "by_skill": {}}
    for (sid, ph), t in sorted(tallies.items()):
        if sid != "*":
            out["by_skill"].setdefault(sid, {})[ph] = t.summary()
    (run_dir / "cosmos_agree.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Cosmos verdicts vs oracle atoms on recorded eval frames")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--registry", default="registry/skills.yaml")
    ap.add_argument("--verifier", choices=["vllm", "fake"], default="vllm")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI-compatible base URL")
    ap.add_argument("--model", default="nvidia/Cosmos-Reason2-8B")
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument("--phases", default="post,pre")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fake-error-rate", type=float, default=0.1)
    ap.add_argument("--fake-unsure-rate", type=float, default=0.05)
    a = ap.parse_args(argv)
    v: Verifier = (VllmVerifier(a.url, a.model, a.timeout_s) if a.verifier == "vllm"
                   else FakeVerifier(a.fake_error_rate, a.fake_unsure_rate))
    res = score(Path(a.inp), a.registry, v, phases=tuple(p for p in a.phases.split(",") if p), limit=a.limit)
    print(json.dumps({"overall": res["overall"], "by_phase": res["by_phase"]}, indent=1))


if __name__ == "__main__":
    main()
