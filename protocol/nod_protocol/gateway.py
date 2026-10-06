"""Gateway to phone messages (3.5.3). The presenter receives these too, mirrored."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import (
    CosmosAnswer,
    CosmosDecision,
    Envelope,
    Fault,
    HomeAction,
    NodModel,
    OptionConfirm,
    OptionKind,
    Risk,
    RunPhase,
    SessionMode,
)

# Badges a card can carry. Additive to the design's JSON (see docs/DEVIATIONS.md).
Badge = Literal[
    "less_reliable",  # R6: wilson_lo between 0.30 and 0.60
    "needs_caregiver",  # R7: risk high without caregiver_enabled
    "unsure_from_camera",  # 5.4: oracle ok, Cosmos pre-check no/unsure
    "near_user",  # R14
    "scripted",  # scripted motion, not a learned policy
    "first_use",  # R14: first use of a near-user skill this session
]


class Welcome(Envelope):
    type: Literal["welcome"] = "welcome"
    resume_token: str
    lease: Literal["user", "caregiver", "none"]
    server_time: int


class Pong(Envelope):
    type: Literal["pong"] = "pong"
    t0: int
    t_srv: int


class State(Envelope):
    type: Literal["state"] = "state"
    mode: SessionMode
    run_id: str | None = None
    scene_id: str
    estop_seq: int


class Transcript(Envelope):
    """ASR result. Also the return type of `ASR.transcribe` (4.2.2)."""

    type: Literal["transcript"] = "transcript"
    utt_id: int
    text: str
    asr_ms: float | None = None


class Success(NodModel):
    """Success estimate shown on a card ("19/20 in tests").

    For a plan, `p_est` is the product of in-chain rates (R10), and `n`, `k` and
    `wilson_lo` describe the weakest step. `k` is an addition to the design JSON.
    """

    p_est: float = Field(ge=0.0, le=1.0)
    basis: Literal["alone", "in_chain"]
    n: int = Field(ge=0)
    wilson_lo: float = Field(ge=0.0, le=1.0)
    k: int | None = Field(default=None, ge=0)


class Check(NodModel):
    """One validator rule outcome (5.2), e.g. {"rule": "R4_preconditions", "ok": true}."""

    rule: str
    ok: bool
    detail: str | None = None


class Precheck(NodModel):
    """Cosmos pre-check status of one option (5.4)."""

    status: Literal["pending", "ok", "unsure", "block", "error"]
    reason: str | None = None


class Option(NodModel):
    """One card. `label` always comes from the registry (R13); `why` is LLM text, <= 80 chars."""

    option_id: str
    kind: OptionKind
    skill_ids: list[str] = Field(default_factory=list)  # empty for kind "home"
    home_action: HomeAction | None = None
    label: str
    why: str = Field(default="", max_length=80)
    success: Success | None = None
    est_s: float | None = None
    risk: Risk
    confirm: OptionConfirm
    checks: list[Check] = Field(default_factory=list)
    precheck: Precheck | None = None
    # Additions (docs/DEVIATIONS.md): visible badges and the R7 lock.
    badges: list[Badge] = Field(default_factory=list)
    locked: bool = False  # shown but cannot be confirmed (e.g. "Needs caregiver")


class FixedOption(NodModel):
    """Cards the UI always adds, e.g. {"option_id": "none", "label": "None of these"}."""

    option_id: str
    label: str


class Proposal(Envelope):
    type: Literal["proposal"] = "proposal"
    proposal_id: str
    expires_at: int  # gateway clock ms
    arm_delay_ms: int = 300
    source: Literal["nemotron", "menu", "fake"]  # menu = deterministic menu (5.7)
    model: str | None = None
    transcript: str | None = None
    options: list[Option] = Field(max_length=3)
    fixed: list[FixedOption] = Field(default_factory=lambda: [FixedOption(option_id="none", label="None of these")])
    focus: str | None = None
    nonce: str


class CosmosVerdict(NodModel):
    """Compact Cosmos outcome embedded in a `run` update."""

    check_id: str
    phase: Literal["pre", "post"]
    answer: CosmosAnswer
    reason: str = ""
    decision: CosmosDecision


class RunUpdate(Envelope):
    """Progress of the run in flight (`type: "run"`)."""

    type: Literal["run"] = "run"
    run_id: str
    proposal_id: str | None = None
    option_id: str | None = None
    plan_step: int = 0
    plan_len: int = 1
    skill_id: str
    phase: RunPhase
    sim_step: int = 0
    max_steps: int
    elapsed_ms: int = 0
    cosmos: CosmosVerdict | None = None
    oracle: bool | None = None  # oracle success atom value after the run
    fault: Fault | None = None  # addition: sim fault that ended the run, if any


class LatMs(NodModel):
    """E-stop latency breakdown in ms (5.5)."""

    phone_gw: float | None = None
    gw_sim: float | None = None
    sim_halt: float | None = None


class EStopAck(Envelope):
    type: Literal["estop.ack"] = "estop.ack"
    estop_seq: int
    stopped_at_step: int | None = None
    lat_ms: LatMs = Field(default_factory=LatMs)


class Notice(Envelope):
    type: Literal["notice"] = "notice"
    level: Literal["info", "warn", "error"] = "info"
    text: str
    speak: bool = False


class Thumb(Envelope):
    """Header for the binary JPEG frame that follows it (at most 5 fps)."""

    type: Literal["thumb"] = "thumb"
    camera: str
    w: int
    h: int
    step: int
