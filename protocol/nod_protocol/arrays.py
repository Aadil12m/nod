"""float32 array field type used by the `pose` message (design 3.5.6).

Wire format (msgpack and JSON alike) for one array is a plain map:

    {"dtype": "<f4", "shape": [nb, 3], "data": <raw little-endian float32 bytes, C order>}

In msgpack `data` is a `bin` value. In JSON (`model_dump_json`) `data` is base64.
A consumer without this package decodes it with numpy only:

    np.frombuffer(m["data"], dtype="<f4").reshape(m["shape"])

numpy is imported lazily, so the rest of the package works without it. Install
the `numpy` extra (`pip install nod-protocol[numpy]`) to use `Pose`.
"""

from __future__ import annotations

import base64
from typing import Annotated, Any

from pydantic import BeforeValidator, PlainSerializer, SerializationInfo, WithJsonSchema

DTYPE = "<f4"


def _np():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only without numpy
        raise ImportError("nod_protocol pose arrays need numpy: pip install 'nod-protocol[numpy]'") from exc
    return np


def to_f32(value: Any):
    """Coerce an ndarray, nested list, or wire map into a C-contiguous float32 ndarray."""
    np = _np()
    if isinstance(value, dict):
        try:
            dtype, shape, data = value["dtype"], value["shape"], value["data"]
        except KeyError as exc:
            raise ValueError(f"array map is missing {exc.args[0]!r}") from exc
        if dtype != DTYPE:
            raise ValueError(f"array dtype must be {DTYPE!r}, got {dtype!r}")
        if isinstance(data, str):
            data = base64.b64decode(data, validate=True)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValueError("array data must be bytes (msgpack) or base64 text (JSON)")
        shape = tuple(int(s) for s in shape)
        count = 1
        for s in shape:
            count *= s
        if len(data) != 4 * count:
            raise ValueError(f"array data has {len(data)} bytes, shape {shape} needs {4 * count}")
        # frombuffer returns a read-only view; copy so callers own a normal array.
        return np.frombuffer(bytes(data), dtype=DTYPE).reshape(shape).copy()
    arr = np.asarray(value, dtype=DTYPE)
    if not np.all(np.isfinite(arr)):
        raise ValueError("array contains NaN or inf")
    return np.ascontiguousarray(arr)


def _serialize(arr: Any, info: SerializationInfo) -> dict:
    np = _np()
    a = np.ascontiguousarray(arr, dtype=DTYPE)
    raw = a.tobytes()
    data: Any = base64.b64encode(raw).decode("ascii") if info.mode == "json" else raw
    return {"dtype": DTYPE, "shape": list(a.shape), "data": data}


_JSON_SCHEMA = {
    "type": "object",
    "title": "Float32Array",
    "description": "Raw little-endian float32 array: decode with np.frombuffer(data, '<f4').reshape(shape). "
    "data is msgpack bin on ZMQ and base64 text in JSON.",
    "properties": {
        "dtype": {"const": DTYPE, "type": "string"},
        "shape": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "data": {"type": "string", "contentEncoding": "base64"},
    },
    "required": ["dtype", "shape", "data"],
    "additionalProperties": False,
}

F32Array = Annotated[
    Any,
    BeforeValidator(to_f32),
    PlainSerializer(_serialize, return_type=dict),
    WithJsonSchema(_JSON_SCHEMA),
]
"""A numpy float32 array in Python, the `{dtype, shape, data}` map on the wire."""
