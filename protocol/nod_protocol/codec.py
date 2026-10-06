"""Discriminated unions and (de)serialisation helpers.

* WebSocket: JSON text frames. `parse_ws(text)` returns the concrete model chosen
  by `type`; `to_ws(model)` returns the JSON text.
* ZMQ: msgpack. `dumps(model)` returns bytes; `loads(data)` returns the model.
  ormsgpack is used when installed (it is faster), else msgpack. Both produce
  standard msgpack and each can read the other's output.

Narrower adapters (`PHONE_IN`, `SAFETY_IN`, `PRESENTER_IN`, `CTL`, `SIM_OUT`)
let a socket handler reject message types that do not belong on it.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

from .cosmos import CosmosCheck, CosmosResult, Metrics, Trace
from .gateway import EStopAck, Notice, Pong, Proposal, RunUpdate, State, Thumb, Transcript, Welcome
from .phone import (
    EStop,
    EStopReset,
    Gesture,
    Heartbeat,
    Hello,
    Ping,
    UiCancel,
    UiConfirm,
    UiFocus,
    UiMenu,
    UtterEnd,
    UtterStart,
    UtterText,
)
from .sim import (
    Abort,
    Capture,
    CtlEStop,
    CtlHeartbeat,
    LoadScene,
    Pause,
    Pose,
    Resume,
    RunScripted,
    RunSkill,
    SimAck,
    SimEpisodeHeader,
    SimEStopAck,
    SimFrames,
    SimStatus,
    SimWorld,
    WorldRequest,
)

# Message classes grouped by link (design 3.3 / 3.5).
PHONE_IN_TYPES = (Hello, Gesture, UiMenu, UiFocus, UiConfirm, UiCancel, UtterStart, UtterEnd, UtterText, Ping)
SAFETY_IN_TYPES = (Heartbeat, EStop, EStopReset, Ping)
PRESENTER_IN_TYPES = (EStop, Ping)
GATEWAY_OUT_TYPES = (Welcome, Pong, State, Transcript, Proposal, RunUpdate, EStopAck, Notice, Thumb)
PRESENTER_OUT_TYPES = (Trace, Metrics, CosmosCheck, CosmosResult)
CTL_TYPES = (LoadScene, RunSkill, RunScripted, Pause, Resume, Abort, Capture, WorldRequest, CtlHeartbeat, CtlEStop)
SIM_OUT_TYPES = (SimAck, SimStatus, SimEStopAck, SimWorld, SimFrames, SimEpisodeHeader)
POSE_TYPES = (Pose,)

WS_TYPES = tuple(dict.fromkeys(PHONE_IN_TYPES + SAFETY_IN_TYPES + GATEWAY_OUT_TYPES + PRESENTER_OUT_TYPES))
ZMQ_TYPES = CTL_TYPES + SIM_OUT_TYPES + POSE_TYPES
ALL_TYPES = WS_TYPES + ZMQ_TYPES


def _union(types: tuple[type[BaseModel], ...]) -> Any:
    if len(types) == 1:
        return types[0]
    return Annotated[Union[types], Field(discriminator="type")]  # noqa: UP007


def _type_name(cls: type[BaseModel]) -> str:
    return cls.model_fields["type"].default


TYPE_TO_MODEL: dict[str, type[BaseModel]] = {_type_name(c): c for c in ALL_TYPES}

WsMessage = _union(WS_TYPES)
ZmqAny = _union(ZMQ_TYPES)
AnyMessage = _union(ALL_TYPES)
PhoneIn = _union(PHONE_IN_TYPES)
SafetyIn = _union(SAFETY_IN_TYPES)
PresenterIn = _union(PRESENTER_IN_TYPES)
GatewayOut = _union(GATEWAY_OUT_TYPES)
CtlMessage = _union(CTL_TYPES)
SimOut = _union(SIM_OUT_TYPES)

WS = TypeAdapter(WsMessage)
PHONE_IN = TypeAdapter(PhoneIn)
SAFETY_IN = TypeAdapter(SafetyIn)
PRESENTER_IN = TypeAdapter(PresenterIn)
GATEWAY_OUT = TypeAdapter(GatewayOut)
CTL = TypeAdapter(CtlMessage)
SIM_OUT = TypeAdapter(SimOut)
ANY = TypeAdapter(AnyMessage)


# ---------------------------------------------------------------- JSON (WebSocket)


def parse_ws(text: str | bytes, adapter: TypeAdapter | None = None) -> BaseModel:
    """Parse one WebSocket JSON text frame into its model (discriminated on `type`).

    Pass a narrower adapter (e.g. `SAFETY_IN`) to reject types that do not belong
    on that socket. Raises `pydantic.ValidationError` on anything invalid.
    """
    return (adapter or WS).validate_json(text)


def to_ws(msg: BaseModel) -> str:
    """Serialise a model to a compact JSON text frame (bytes become base64)."""
    return msg.model_dump_json()


# ---------------------------------------------------------------- msgpack (ZMQ)

Backend = Literal["auto", "msgpack", "ormsgpack"]

try:  # optional fast path
    import ormsgpack as _ormsgpack
except ImportError:  # pragma: no cover - depends on the environment
    _ormsgpack = None

import msgpack as _msgpack  # noqa: E402


def _pick(backend: Backend) -> str:
    if backend == "auto":
        return "ormsgpack" if _ormsgpack is not None else "msgpack"
    if backend == "ormsgpack" and _ormsgpack is None:
        raise ImportError("ormsgpack is not installed: pip install 'nod-protocol[fast]'")
    return backend


def packb(obj: Any, backend: Backend = "auto") -> bytes:
    """msgpack-encode plain Python data (dict/list/str/int/float/bool/None/bytes)."""
    if _pick(backend) == "ormsgpack":
        return _ormsgpack.packb(obj)
    return _msgpack.packb(obj, use_bin_type=True)


def unpackb(data: bytes, backend: Backend = "auto") -> Any:
    if _pick(backend) == "ormsgpack":
        return _ormsgpack.unpackb(data)
    return _msgpack.unpackb(data, raw=False)


def dumps(msg: BaseModel, backend: Backend = "auto") -> bytes:
    """Encode a message as msgpack for ZMQ. Arrays become {dtype, shape, data(bin)}."""
    return packb(msg.model_dump(mode="python"), backend)


def loads(data: bytes, adapter: TypeAdapter | None = None, backend: Backend = "auto") -> BaseModel:
    """Decode a msgpack message into its model, discriminated on `type`.

    Accepts every message type (WebSocket ones too); pass `CTL` or `SIM_OUT` to
    restrict. Raises `pydantic.ValidationError` on invalid content.
    """
    return (adapter or ANY).validate_python(unpackb(data, backend))
