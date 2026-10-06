"""Phone to gateway messages: `/ws/phone` (3.5.1) and `/ws/safety` (3.5.2)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Envelope, NodModel

GestureName = Literal[
    "long_blink", "double_blink", "brow_raise", "turn_left", "turn_right",
    "jaw_hold", "face_lost", "face_found",
]
EStopSource = Literal["jaw_hold", "voice_stop", "button", "presenter"]


class Calib(NodModel):
    """Calibration summary sent in `hello` (6.3). The profile itself stays on the phone."""

    profile_id: str
    long_blink_ms: int
    fps: float


# ---------------------------------------------------------------- /ws/phone (3.5.1)


class Hello(Envelope):
    type: Literal["hello"] = "hello"
    device: str
    app_ver: str
    role: Literal["user", "caregiver"] = "user"
    resume_token: str | None = None
    calib: Calib | None = None


class Gesture(Envelope):
    """Telemetry: the phone's UI state machine has already acted on the gesture."""

    type: Literal["gesture"] = "gesture"
    name: GestureName
    phase: Literal["fired", "cancelled"]
    held_ms: int | None = None
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    fps: float | None = None


class UiMenu(Envelope):
    type: Literal["ui.menu"] = "ui.menu"
    source: str  # "brow_raise", "keyboard", "button", ...


class UiFocus(Envelope):
    type: Literal["ui.focus"] = "ui.focus"
    proposal_id: str
    option_id: str


class UiConfirm(Envelope):
    """Confirm of one option. Checked by `validate_confirm` (rule R9)."""

    type: Literal["ui.confirm"] = "ui.confirm"
    proposal_id: str
    option_id: str
    nonce: str
    gesture: str  # "long_blink" normally; alternative gesture sets from 6.3 use their own names
    held_ms: int
    shown_at: int  # phone clock ms when the card was shown
    focused_at: int  # phone clock ms when focus landed on this option


class UiCancel(Envelope):
    type: Literal["ui.cancel"] = "ui.cancel"
    proposal_id: str | None = None
    source: str  # "jaw_hold", "voice_stop", "button", ...


class UtterStart(Envelope):
    """Start of an utterance. Audio follows as binary frames: [u32 LE utt_id][pcm16le samples]."""

    type: Literal["utter.start"] = "utter.start"
    utt_id: int
    sr: int = 16000
    fmt: Literal["pcm16le"] = "pcm16le"


class UtterEnd(Envelope):
    type: Literal["utter.end"] = "utter.end"
    utt_id: int
    dur_ms: int


class UtterText(Envelope):
    """Typed text in place of speech (caregiver, or mock mode)."""

    type: Literal["utter.text"] = "utter.text"
    utt_id: int
    text: str


class Ping(Envelope):
    type: Literal["ping"] = "ping"
    t0: int


# ---------------------------------------------------------------- /ws/safety (3.5.2)


class Heartbeat(Envelope):
    """Sent every 100 ms on /ws/safety.

    `resume_token` is the token from this phone's `welcome` on /ws/phone. It binds the
    heartbeat to the phone session: only the heartbeat of the phone holding the `user` lease
    feeds the watchdogs and the R9 face check (a `latched` heartbeat from any device still
    counts as a stop). None until the phone has been welcomed.
    """

    type: Literal["hb"] = "hb"
    face_present: bool
    track_conf: float
    fps: float
    latched: bool
    resume_token: str | None = None


class EStop(Envelope):
    """Resent every 100 ms until the matching `estop.ack`. Also sent by the presenter."""

    type: Literal["estop"] = "estop"
    estop_seq: int
    source: EStopSource
    t_detect: int  # phone (or presenter) clock ms at detection
    held_ms: int | None = None


class EStopReset(Envelope):
    """Clears the latched stop `estop_seq` (the latest one the phone knows of). Accepted only
    from the phone holding the `user` lease (long blink of at least the minimum hold, or a
    caregiver touch on that phone) or a `caregiver` phone, identified by `resume_token`."""

    type: Literal["estop.reset"] = "estop.reset"
    estop_seq: int
    gesture: str  # "long_blink" or "touch" (caregiver)
    held_ms: int | None = None
    resume_token: str | None = None
