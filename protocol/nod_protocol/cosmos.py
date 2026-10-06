"""Cosmos checks (3.5.7) and presenter messages (3.5.8)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .base import CosmosAnswer, CosmosDecision, Envelope, NodModel


class FrameRef(NodModel):
    """Which captured frame a check looked at. The image bytes travel separately."""

    camera: str
    w: int
    h: int
    step: int


class CosmosCheck(Envelope):
    """Request to the verifier (`Verifier.check`). Pre-checks run before a run exists,
    so `run_id` may be null; `proposal_id`/`option_id` identify the card instead."""

    type: Literal["cosmos.check"] = "cosmos.check"
    check_id: str
    run_id: str | None = None
    proposal_id: str | None = None  # addition: pre-checks are per displayed option
    option_id: str | None = None  # addition
    phase: Literal["pre", "post"]
    skill_id: str
    question: str
    frames: list[FrameRef] = Field(default_factory=list)
    model: str = "nvidia/Cosmos-Reason2-8B"


class OracleAtom(NodModel):
    atom: str
    value: bool


class CosmosResult(Envelope):
    type: Literal["cosmos.result"] = "cosmos.result"
    check_id: str
    answer: CosmosAnswer
    reason: str = ""
    latency_ms: float | None = None
    oracle: OracleAtom | None = None
    agree: bool | None = None
    decision: CosmosDecision


class CosmosReply(NodModel):
    """The JSON object Cosmos is asked to return (used as its response_format schema)."""

    answer: CosmosAnswer
    reason: str = Field(default="", max_length=200)


TraceKind = Literal["planner", "validator", "cosmos", "run", "estop", "asr", "home"]


class Trace(Envelope):
    type: Literal["trace"] = "trace"
    kind: TraceKind
    ref: str | None = None  # proposal, run or check id
    data: dict[str, Any] = Field(default_factory=dict)


class Metrics(Envelope):
    type: Literal["metrics"] = "metrics"
    estop_ms_p50: float | None = None
    estop_ms_p95: float | None = None
    view_lag_ms: float | None = None
    policy_ms_p95: float | None = None
    asr_ms_p50: float | None = None
    llm_ms_p50: float | None = None


class HomeResult(NodModel):
    """Return value of `Home.call` (4.2.2). Not a wire message."""

    action: Literal["lights_on", "lights_off", "call_caregiver"]
    ok: bool
    detail: str | None = None
    latency_ms: float | None = None
