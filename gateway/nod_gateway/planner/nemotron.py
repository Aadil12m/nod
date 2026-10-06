"""Nemotron planner on Nebius Token Factory (design 7.1, 7.2, 7.5).

Two routes:
  fast  (utterance -> 1-3 options, prefetch): Nemotron-3.5 Lightning, `enable_thinking: false`,
        temperature 0.2, max_tokens 400, timeout 2.5 s.
  plan  (multi-step plans, recovery):          Nemotron-3 Super, `enable_thinking: true,
        low_effort: true`, temperature 0.6, max_tokens 1500, timeout 8 s.

Structured output is not confirmed for Nemotron on Token Factory (7.1), so the planner tries,
in order: `response_format` json_schema with the per-request enums, then a single tool
`propose_options` with `tool_choice: "required"`, then `json_object` with the schema in the
prompt; Pydantic validates every result (R1). The first mechanism that works is remembered
per model (the fast and plan routes use different models, which may support different
mechanisms) and tried first next time. An HTTP 400 moves to the next mechanism in the same
call, also when the remembered one is rejected; a schema failure goes straight to json_object
(one retry, 7.5). Timeouts, 429s and connection errors raise PlannerError at once and start
a backoff during which calls fail fast, so the user sees the deterministic menu without waiting.

Only exercised against recorded/fake HTTP responses here; the G3 smoke test (design 8.6)
must confirm which mechanism Token Factory accepts.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx
from nod_protocol import PlannerOutput, PlanRequest, planner_response_format

from .base import PlannerError, PlannerResult, dumps_context, load_prompt, output_from_obj, parse_output

log = logging.getLogger(__name__)

MECHANISMS = ("json_schema", "tools", "json_object")
TOOL_NAME = "propose_options"


class NemotronPlanner:
    source = "nemotron"

    def __init__(self, cfg, http: httpx.AsyncClient) -> None:
        from openai import AsyncOpenAI  # imported lazily so mock mode never needs the SDK configured

        self.cfg = cfg
        key = cfg.tf_api_key or "missing-key"
        # max_retries=0: our own deadline and backoff apply; the SDK must not retry behind our back.
        self.fast_client = AsyncOpenAI(base_url=cfg.tf_fast_url, api_key=key, http_client=http, max_retries=0)
        self.plan_client = AsyncOpenAI(base_url=cfg.tf_plan_url, api_key=key, http_client=http, max_retries=0)
        # A configured mechanism is the only one tried; with "auto" the one that worked last is
        # remembered per model and tried first.
        self.fixed: str | None = None if cfg.planner_structured == "auto" else cfg.planner_structured
        self.sticky: dict[str, str] = {}
        self.backoff_until = 0.0
        self.backoff_s = 0.0
        self.last_error: str | None = None
        self.prompts = {"fast": load_prompt("fast.md"), "plan": load_prompt("plan.md")}

    # ------------------------------------------------------------ interface

    async def propose(self, req: PlanRequest) -> PlannerOutput:
        return (await self.propose_ex(req)).output

    def _route(self, req: PlanRequest) -> dict[str, Any]:
        if req.mode == "fast":
            return {
                "client": self.fast_client,
                "model": self.cfg.fast_model,
                "prompt": self.prompts["fast"],
                "temperature": 0.2,
                "max_tokens": 400,
                "timeout": self.cfg.fast_timeout_s,
                "thinking": {"enable_thinking": False},
            }
        return {
            "client": self.plan_client,
            "model": self.cfg.plan_model,
            "prompt": self.prompts["plan"],
            "temperature": 0.6,
            "max_tokens": 1500,
            "timeout": self.cfg.plan_timeout_s,
            "thinking": {"enable_thinking": True, "low_effort": True},
        }

    def _start_backoff(self, why: str) -> None:
        self.backoff_s = min(30.0, max(1.0, self.backoff_s * 2))
        self.backoff_until = time.monotonic() + self.backoff_s
        self.last_error = why

    async def propose_ex(self, req: PlanRequest) -> PlannerResult:
        import openai

        if not self.cfg.tf_api_key:
            self.last_error = "no Token Factory API key (NOD_TF_API_KEY or NEBIUS_API_KEY)"
            raise PlannerError(self.last_error)
        if time.monotonic() < self.backoff_until:
            raise PlannerError(f"planner offline, backing off ({self.last_error})")

        route = self._route(req)
        feasible_ids = [f.skill_id for f in req.feasible if f.kind != "home"]
        schema = planner_response_format(feasible_ids, req.plan_ids)
        context = dumps_context(req)
        t0 = time.monotonic()
        deadline = t0 + route["timeout"]
        model = route["model"]
        if self.fixed:
            order = [self.fixed]
        elif model in self.sticky:
            first = self.sticky[model]
            order = [first, *(m for m in MECHANISMS if m != first)]
        else:
            order = list(MECHANISMS)
        last: Exception | None = None
        i = 0
        while i < len(order):
            mech = order[i]
            remaining = deadline - time.monotonic()
            if remaining <= 0.05:
                break
            try:
                raw, out = await self._call(mech, route, schema, context, remaining)
            except openai.BadRequestError as exc:  # mechanism not supported by this model
                last = exc
                log.info("planner %s rejected %s: %s", route["model"], mech, exc)
                if self.sticky.get(model) == mech:
                    del self.sticky[model]  # no longer accepted: fall through to the others
                i += 1
                continue
            except PlannerError as exc:  # R1 failure: one retry with json_object (7.5)
                last = exc
                if mech != "json_object":
                    order = [*order[: i + 1], "json_object"]
                i += 1
                continue
            except openai.RateLimitError as exc:
                self._start_backoff("HTTP 429")
                raise PlannerError("rate limited (429)") from exc
            except openai.APITimeoutError as exc:
                self._start_backoff("timeout")
                raise PlannerError("planner timeout") from exc
            except (openai.APIConnectionError, openai.APIStatusError) as exc:
                self._start_backoff(type(exc).__name__)
                raise PlannerError(f"planner unavailable: {exc}") from exc
            if not self.fixed:
                self.sticky[model] = mech
            self.backoff_s = 0.0
            self.last_error = None
            return PlannerResult(
                output=out,
                model=route["model"],
                source="nemotron",
                mechanism=mech,
                latency_ms=(time.monotonic() - t0) * 1000,
                raw=raw,
                request={"model": route["model"], "mode": req.mode, "mechanism": mech, "context": json.loads(context)},
            )
        if last is None:
            self._start_backoff("timeout")
            raise PlannerError("planner timeout")
        self.last_error = f"all mechanisms failed: {last}"
        raise PlannerError(self.last_error) from last

    # ------------------------------------------------------------ one request

    async def _call(
        self, mech: str, route: dict[str, Any], schema: dict[str, Any], context: str, timeout: float
    ) -> tuple[str, PlannerOutput]:
        system = route["prompt"]
        kwargs: dict[str, Any] = {
            "model": route["model"],
            "temperature": route["temperature"],
            "max_tokens": route["max_tokens"],
            "timeout": timeout,
            "extra_body": {"chat_template_kwargs": route["thinking"]},
        }
        if mech == "json_schema":
            kwargs["response_format"] = {"type": "json_schema", "json_schema": schema}
        elif mech == "tools":
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": TOOL_NAME,
                        "description": "Propose 0-3 options for the person.",
                        "parameters": schema["schema"],
                    },
                }
            ]
            kwargs["tool_choice"] = "required"
        else:
            kwargs["response_format"] = {"type": "json_object"}
            system = (
                system
                + "\n\nReply with one JSON object that matches this JSON Schema exactly:\n"
                + json.dumps(schema["schema"])
            )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": context}]
        resp = await route["client"].chat.completions.create(messages=messages, **kwargs)
        if not resp.choices:
            raise PlannerError("empty response")
        msg = resp.choices[0].message
        if mech == "tools":
            calls = msg.tool_calls or []
            if not calls:
                raise PlannerError("no tool call in response")
            raw = calls[0].function.arguments or ""
            try:
                return raw, output_from_obj(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise PlannerError(f"R1: tool arguments are not JSON: {exc}") from exc
        raw = msg.content or ""
        return raw, parse_output(raw)
