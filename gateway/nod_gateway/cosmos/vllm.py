"""Cosmos-Reason2-8B on vLLM's OpenAI-compatible server (design 3.1, 3.5.7).

A standard chat completion with one `image_url` data URI per frame, temperature 0,
max_tokens 128 and `response_format` json_schema {answer, reason}. Whether vLLM's guided
decoding works together with Cosmos-Reason2's reasoning parser is unverified (3.5.7); if the
server rejects the schema (HTTP 400) we retry without it and parse a leading yes/no.
Only exercised here against fake HTTP responses.
"""

from __future__ import annotations

import base64
import re
import time
from collections.abc import Sequence

import httpx
from nod_protocol import CosmosCheck, CosmosReply, CosmosResult, Frame
from pydantic import ValidationError

from .base import VerifierError, decision_for, question_with_format

_THINK = re.compile(r"<think>.*?</think>", re.S)
_LEAD = re.compile(r"^\W*(yes|no|unsure)\b", re.I)


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
        return CosmosReply(answer=m.group(1).lower(), reason=t[m.end() :].strip(" .,:;-")[:200])  # type: ignore[arg-type]
    return CosmosReply(answer="unsure", reason=("unparsed: " + t)[:200])


def data_uri(frame: Frame) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(frame.jpeg).decode("ascii")


class VllmVerifier:
    def __init__(self, cfg, http: httpx.AsyncClient) -> None:
        from openai import AsyncOpenAI

        if not cfg.cosmos_url:
            raise VerifierError("NOD_COSMOS_URL is not set")
        self.cfg = cfg
        self.model = cfg.cosmos_model
        self.client = AsyncOpenAI(base_url=cfg.cosmos_url, api_key="EMPTY", http_client=http, max_retries=0)
        self.use_schema = True
        self.last_error: str | None = None

    async def check(self, req: CosmosCheck, images: Sequence[Frame] = ()) -> CosmosResult:
        import openai

        content: list[dict] = [{"type": "image_url", "image_url": {"url": data_uri(f)}} for f in images]
        content.append({"type": "text", "text": question_with_format(req.question)})
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 128,
            "timeout": self.cfg.cosmos_timeout_s,
        }
        t0 = time.perf_counter()
        try:
            try:
                if self.use_schema:
                    schema = {
                        "name": "cosmos_reply",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["answer", "reason"],
                            "properties": {
                                "answer": {"type": "string", "enum": ["yes", "no", "unsure"]},
                                "reason": {"type": "string", "maxLength": 200},
                            },
                        },
                    }
                    resp = await self.client.chat.completions.create(
                        response_format={"type": "json_schema", "json_schema": schema}, **kwargs
                    )
                else:
                    resp = await self.client.chat.completions.create(**kwargs)
            except openai.BadRequestError:
                if not self.use_schema:
                    raise
                self.use_schema = False  # guided JSON unsupported with the reasoning parser: parse text
                resp = await self.client.chat.completions.create(**kwargs)
        except (openai.APIError, httpx.HTTPError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise VerifierError(self.last_error) from exc
        text = resp.choices[0].message.content if resp.choices else ""
        reply = parse_reply(text or "")
        self.last_error = None
        return CosmosResult(
            check_id=req.check_id,
            answer=reply.answer,
            reason=reply.reason,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            decision=decision_for(req.phase, reply.answer),  # type: ignore[arg-type]
        )


__all__ = ["VllmVerifier", "data_uri", "parse_reply"]
