#!/usr/bin/env python
"""Nod Day-1 gate G3: does Nemotron on Token Factory return valid structured output? (design 8.6)

Run on a laptop or the VM. Only the `openai` package is needed:

    export NEBIUS_API_KEY=...            # https://tokenfactory.nebius.com/project/api-keys
    uv run --no-project --with openai python docs/runbooks/nemotron_smoke.py | tee ~/nod_spike/g3.log

What it does:
1. Lists the Nemotron models each base URL serves (global and us-central1), which settles the
   [unverified] question of whether the global URL serves Lightning (design 7.1, 10.2).
2. For each model and each mode (json_schema, tools, json_object), sends the prompts twice and
   checks every reply against the planner schema of design 7.4, with the per-request enums filled
   in (the same schema the gateway builds with nod_protocol.planner_response_format).
3. Prints one summary line per model and mode in the form the 8.8 table asks for:
   "valid 9/10, p50 0.84s, p95 1.71s". G3 passes with 9/10 or more and fast-path p95 <= 2.5 s.

Put the winning mode into .env as NOD_PLANNER_STRUCTURED and copy the table into
docs/runbooks/day1-results.md. Pass --out to keep every raw reply as JSON lines.

Differences from the inline script in design 8.6: it uses the full 7.4 schema (plan and
clarify_question included), adds the json_object fallback mode of 7.5, and validates each reply
field by field instead of only checking skill ids.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

URLS = {
    "global": "https://api.tokenfactory.nebius.com/v1/",
    "us": "https://api.tokenfactory.us-central1.nebius.com/v1/",
}
# (model id, which URL serves it) from design 7.1.
MODELS = [
    ("nvidia/Nemotron-3_5-Lightning", "global"),
    ("nvidia/nemotron-3-super-120b-a12b", "us"),
]
MODES = ["json_schema", "tools", "json_object"]

# A realistic request: the feasible skills from the starting layout, and the policy skills a plan may use.
FEASIBLE = ["bowl_on_plate", "cheese_in_bowl", "bottle_on_rack", "open_middle_drawer", "plate_to_stove_front"]
PLAN_IDS = FEASIBLE + ["bowl_to_user"]
HOME_ACTIONS = ["lights_on", "lights_off", "call_caregiver"]
INTENTS = ["manipulate", "home", "caregiver", "clarify", "none"]

SYSTEM = (
    "You plan for Nod, an assistive robot arm used by a person who cannot use their hands. "
    "You never move the arm. Pick up to 3 options using only ids from the enums. "
    "For anything the arm cannot do (drink, food to mouth, things on the floor, people), offer "
    "call_caregiver and say why. 'why' is at most 80 characters, plain words, addressed to the person. "
    "Reply with JSON only."
)
PROMPTS = ["I'm a bit hungry", "I dropped my phone", "it's too dark in here", "tidy up the bottle",
           "open the drawer please"]


def planner_schema(feasible_ids: list[str], plan_ids: list[str]) -> dict[str, Any]:
    """The design 7.4 schema. Kept identical to nod_protocol.planner_response_format (a test checks it)."""
    skill_enum: list[Any] = sorted(set(feasible_ids)) + [None]
    if plan_ids:
        plan_schema: dict[str, Any] = {
            "type": ["array", "null"], "minItems": 2, "maxItems": 4,
            "items": {"type": "string", "enum": sorted(set(plan_ids))},
        }
    else:
        plan_schema = {"type": "null"}
    return {
        "name": "nod_plan",
        "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "required": ["intent", "clarify_question", "options"],
            "properties": {
                "intent": {"type": "string", "enum": list(INTENTS)},
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


def check_output(out: Any, feasible: list[str], plan_ids: list[str]) -> list[str]:
    """Return a list of problems (empty means valid). Mirrors validator rule R1 plus the enums."""
    errs: list[str] = []
    if not isinstance(out, dict):
        return ["reply is not a JSON object"]
    if set(out) != {"intent", "clarify_question", "options"}:
        errs.append(f"top-level keys {sorted(out)}")
    if out.get("intent") not in INTENTS:
        errs.append(f"intent {out.get('intent')!r}")
    cq = out.get("clarify_question")
    if cq is not None and (not isinstance(cq, str) or len(cq) > 60):
        errs.append("clarify_question not null or a string of at most 60 characters")
    opts = out.get("options")
    if not isinstance(opts, list) or len(opts) > 3:
        return [*errs, "options is not a list of at most 3"]
    for i, o in enumerate(opts):
        if not isinstance(o, dict) or set(o) != {"kind", "skill_id", "plan", "home_action", "why"}:
            errs.append(f"option {i}: wrong keys")
            continue
        if o["kind"] not in ("skill", "plan", "home"):
            errs.append(f"option {i}: kind {o['kind']!r}")
        if o["skill_id"] is not None and o["skill_id"] not in feasible:
            errs.append(f"option {i}: skill_id {o['skill_id']!r} not feasible")
        plan = o["plan"]
        if plan is not None and (not isinstance(plan, list) or not 2 <= len(plan) <= 4
                                 or any(s not in plan_ids for s in plan)):
            errs.append(f"option {i}: plan {plan!r}")
        if o["home_action"] is not None and o["home_action"] not in HOME_ACTIONS:
            errs.append(f"option {i}: home_action {o['home_action']!r}")
        if not isinstance(o["why"], str) or len(o["why"]) > 80:
            errs.append(f"option {i}: why longer than 80 characters")
    return errs


def percentile(xs: list[float], p: float) -> float:
    """Nearest-rank percentile; nan for an empty list."""
    if not xs:
        return math.nan
    s = sorted(xs)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def one_call(client: Any, model: str, mode: str, prompt: str, schema: dict[str, Any], extra: dict[str, Any],
             max_tokens: int, timeout: float) -> tuple[Any, str]:
    """Send one request in the given mode; return (parsed JSON, raw text)."""
    kw: dict[str, Any] = dict(model=model, max_tokens=max_tokens, temperature=0.2, extra_body=extra, timeout=timeout)
    if mode == "json_schema":
        r = client.chat.completions.create(
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": schema}, **kw)
        raw = r.choices[0].message.content or ""
    elif mode == "tools":
        r = client.chat.completions.create(
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            tools=[{"type": "function", "function": {"name": "propose_options",
                                                     "description": "Offer 0-3 options to the person.",
                                                     "parameters": schema["schema"]}}],
            tool_choice="required", **kw)
        raw = r.choices[0].message.tool_calls[0].function.arguments
    else:  # json_object with the schema in the prompt (design 7.5 retry path)
        sys_prompt = SYSTEM + "\nThe reply must match this JSON Schema:\n" + json.dumps(schema["schema"])
        r = client.chat.completions.create(
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, **kw)
        raw = r.choices[0].message.content or ""
    return json.loads(raw), raw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modes", default=",".join(MODES), help="comma list of json_schema,tools,json_object")
    ap.add_argument("--models", default="", help="comma list of model ids (default: Lightning and Super)")
    ap.add_argument("--repeat", type=int, default=2, help="times each of the 5 prompts is sent (default 2 -> 10 calls)")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--thinking", action="store_true",
                    help="leave reasoning on (enable_thinking true, low_effort true), as the Super plan path does")
    ap.add_argument("--base-url", default="", help="send everything to this OpenAI-compatible URL (tests, proxies)")
    ap.add_argument("--out", default="", help="append every raw reply to this JSON lines file")
    args = ap.parse_args(argv)

    from openai import OpenAI  # imported here so --help works without the package

    key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("NOD_TF_API_KEY")
    if not key:
        print("NEBIUS_API_KEY is not set (https://tokenfactory.nebius.com/project/api-keys)", file=sys.stderr)
        return 2
    urls = {name: (args.base_url or url) for name, url in URLS.items()}
    models = MODELS
    if args.models:
        models = [(m, "us" if "super" in m.lower() or "ultra" in m.lower() else "global") for m in args.models.split(",")]
    modes = [m for m in args.modes.split(",") if m]
    bad = [m for m in modes if m not in MODES]
    if bad:
        ap.error(f"unknown modes {bad}")

    extra = {"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}} if args.thinking \
        else {"chat_template_kwargs": {"enable_thinking": False}}
    schema = planner_schema(FEASIBLE, PLAN_IDS)
    out_f = open(args.out, "a") if args.out else None  # noqa: SIM115 - closed below

    print("== models served")
    for name, url in urls.items():
        try:
            ids = sorted(m.id for m in OpenAI(base_url=url, api_key=key).models.list().data
                         if "emotron" in m.id.lower())
        except Exception as e:  # noqa: BLE001 - report and continue
            ids = [f"error: {e!r}"[:160]]
        print(f"{name} ({url}): {ids}")

    print("== structured output")
    summary = []
    for model, where in models:
        client = OpenAI(base_url=urls[where], api_key=key, max_retries=0)
        for mode in modes:
            ok, lat = 0, []
            for prompt in PROMPTS * args.repeat:
                t0 = time.perf_counter()
                raw, errs = "", []
                try:
                    out, raw = one_call(client, model, mode, prompt, schema, extra, args.max_tokens, args.timeout)
                    errs = check_output(out, FEASIBLE, PLAN_IDS)
                except Exception as e:  # noqa: BLE001 - every failure counts as invalid
                    errs = [repr(e)[:200]]
                dt = time.perf_counter() - t0
                lat.append(dt)
                if errs:
                    print(f"  fail {model} {mode} {prompt!r}: {'; '.join(errs)[:200]}")
                else:
                    ok += 1
                if out_f:
                    out_f.write(json.dumps({"model": model, "mode": mode, "prompt": prompt, "ok": not errs,
                                            "errors": errs, "latency_s": round(dt, 3), "raw": raw}) + "\n")
            n = len(lat)
            line = f"{model} {mode}: valid {ok}/{n}, p50 {percentile(lat, 50):.2f}s, p95 {percentile(lat, 95):.2f}s"
            summary.append(line)
            print(line, flush=True)
    if out_f:
        out_f.close()

    print("== summary (paste into docs/runbooks/day1-results.md, G3)")
    for line in summary:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
