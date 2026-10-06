"""Base classes and shared enumerations.

Two families of messages exist:

* `Envelope` subclasses travel over WebSockets (design 3.5.1-3.5.3, 3.5.7, 3.5.8).
  Every one carries the common envelope `v, type, id, ts, sess, seq` at the top
  level next to its payload fields.
* `ZmqMessage` subclasses travel between the gateway and the sim over ZMQ as
  msgpack (3.5.4-3.5.6). They carry `v` and `type` only; correlation uses the
  per-message `req_id`, `run_id` and `epoch` fields from the design.

`type` is a `Literal` on every concrete class and is the discriminator used by
`nod_protocol.codec`.
"""

from __future__ import annotations

import base64
import binascii
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, SerializationInfo, WithJsonSchema

from .ids import new_ulid, now_ms

PROTOCOL_VERSION = 1

# Shared vocabularies (design section in brackets).
Risk = Literal["none", "low", "high"]  # 4.2.3; high = caregiver-gated (R7)
ConfirmPolicy = Literal["single", "double"]  # 4.2.3 registry value
OptionConfirm = Literal["single", "double", "per_step"]  # 3.5.3 proposal option
SkillKind = Literal["policy", "scripted", "home"]  # 4.2.3
OptionKind = Literal["skill", "plan", "home"]  # 3.5.3, 7.4
HomeAction = Literal["lights_on", "lights_off", "call_caregiver"]  # 4.2.2 Home.call, 7.4
ScriptName = Literal["return_home", "reset_scene", "close_middle_drawer", "handover_retreat"]  # 3.5.4
SessionMode = Literal[  # 3.4 session state machine
    "IDLE", "LISTENING", "PLANNING", "AWAITING_CONFIRM", "COUNTDOWN",
    "EXECUTING", "POSTCHECK", "PAUSED", "ESTOPPED",
]
RunPhase = Literal[  # 3.5.3 run.phase
    "countdown", "precheck", "executing", "returning_home", "postcheck",
    "done", "failed", "uncertain", "aborted", "paused",
]
Fault = Literal["eef_out_of_box", "policy_timeout", "nan_action", "hb_lost", "budget"]  # 3.5.5
CosmosAnswer = Literal["yes", "no", "unsure"]  # 3.5.7
CosmosDecision = Literal["done", "failed", "uncertain", "proceed", "block"]  # 3.5.7 / 5.4
PostDecision = Literal["done", "failed", "uncertain"]  # 4.2.2 post_decision


def _b64_in(value: Any) -> Any:
    if isinstance(value, str):
        try:
            # Standard alphabet; URL-safe input is accepted too.
            return base64.b64decode(value.replace("-", "+").replace("_", "/"), validate=True)
        except binascii.Error as exc:
            raise ValueError("invalid base64") from exc
    return value


def _b64_out(value: bytes, info: SerializationInfo) -> Any:
    return base64.b64encode(value).decode("ascii") if info.mode == "json" else value


B64Bytes = Annotated[
    bytes,
    BeforeValidator(_b64_in),
    PlainSerializer(_b64_out),
    WithJsonSchema({"type": "string", "contentEncoding": "base64"}),
]
"""Bytes: msgpack `bin` on ZMQ, standard base64 text (RFC 4648 section 4, what `atob` reads) in JSON."""


class NodModel(BaseModel):
    """Base for every model: unknown fields are ignored on input.

    Declare binary fields as `B64Bytes` (not plain `bytes`) so JSON uses standard base64.
    """

    model_config = ConfigDict(extra="ignore", json_schema_serialization_defaults_required=True)


class StrictModel(NodModel):
    """Base for documents we author ourselves (registry, planner output): typos are errors."""

    model_config = ConfigDict(extra="forbid")


class Envelope(NodModel):
    """Common WebSocket envelope (3.5).

    `id` is a ULID, `ts` the sender wall clock in ms, `seq` increases per sender per
    socket. Defaults let a sender build a message without boilerplate; the sender
    (gateway or phone) must still set `seq` and `sess` correctly before sending.
    """

    v: int = PROTOCOL_VERSION
    type: str
    id: str = Field(default_factory=new_ulid)
    ts: int = Field(default_factory=now_ms)
    sess: str | None = None
    seq: int = 0


class ZmqMessage(NodModel):
    """Base for gateway <-> sim messages carried as msgpack over ZMQ (3.5.4-3.5.6)."""

    v: int = PROTOCOL_VERSION
    type: str
