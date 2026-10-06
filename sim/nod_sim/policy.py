"""Client for the Isaac-GR00T policy server (design 3.3: ZMQ REQ/REP on :5555, msgpack-numpy).

Speaks the wire protocol of `gr00t/policy/server_client.py` at Isaac-GR00T 51d4c89 without
importing gr00t, so the sim's LIBERO venv only needs pyzmq, msgpack and msgpack-numpy:

    request  = {"endpoint": "get_action", "data": {"observation": obs, "options": None}}
    response = [action_dict, info]            or {"error": "..."} on a server-side exception
    ping     = {"endpoint": "ping"}  ->  {"status": "ok", ...}

Arrays travel as msgpack-numpy maps (the server's MsgSerializer decodes those and its own
`__ndarray_class__` form, and refuses object dtypes). The observation is the GR00T sim format
with a batch of one and a horizon of one, exactly as docs/runbooks/chain_test.py builds it:
`video.*` uint8 [1, 1, H, W, 3], `state.*` float32 [1, 1, D], language a 1-tuple of str.
With `--use-sim-policy-wrapper` the server answers `action.<key>` arrays of shape [1, T, 1];
`get_action` returns them stacked as float32 [T, 7] in ACTION_KEYS order (policy units).
"""

from __future__ import annotations

import io
import threading
from typing import Any
from urllib.parse import urlparse

import numpy as np
import zmq

from .env_api import ACTION_KEYS, PolicyError, PolicyTimeout


def _codec():
    try:
        import msgpack
        import msgpack_numpy as mnp
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError("the GR00T client needs msgpack and msgpack-numpy (pip install 'nod-sim[groot]')") from exc
    return msgpack, mnp


def pack(data: Any) -> bytes:
    msgpack, mnp = _codec()

    def default(obj):
        if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
            raise TypeError("object arrays are refused by the GR00T server")
        return mnp.encode(obj)

    return msgpack.packb(data, default=default, use_bin_type=True)


def unpack(data: bytes) -> Any:
    msgpack, mnp = _codec()

    def hook(obj):
        if isinstance(obj, dict):
            marker = obj.get("__ndarray_class__", obj.get(b"__ndarray_class__"))
            if marker:
                return np.load(io.BytesIO(obj.get("as_npy", obj.get(b"as_npy"))), allow_pickle=False)
            nd, kind = obj.get(b"nd", obj.get("nd")), obj.get(b"kind", obj.get("kind"))
            if nd and kind in (b"O", "O"):
                raise ValueError("refusing a pickled object array from the policy server")
        return mnp.decode(obj)

    return msgpack.unpackb(data, object_hook=hook, raw=False)


def batch_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Flat GR00T sim observation -> batch of one, horizon one (chain_test.py `batch`)."""
    b: dict[str, Any] = {}
    for k, v in obs.items():
        if k.startswith("video."):
            b[k] = np.ascontiguousarray(v, dtype=np.uint8)[None, None]
        elif k.startswith("state."):
            b[k] = np.asarray(v, dtype=np.float32).reshape(1, 1, -1)
        else:
            b[k] = (v,)
    return b


def chunk_from_response(action: dict[str, Any]) -> np.ndarray:
    """{"action.x": [1, T, 1], ...} -> float32 [T, 7] in ACTION_KEYS order."""
    cols = []
    for key in ACTION_KEYS:
        name = f"action.{key}"
        if name not in action:
            raise PolicyError(f"policy reply has no {name} (keys: {sorted(action)})")
        a = np.asarray(action[name], dtype=np.float32)
        if a.ndim == 3:
            a = a[0]
        cols.append(a.reshape(a.shape[0], -1)[:, :1])
    lengths = {c.shape[0] for c in cols}
    if len(lengths) != 1:
        raise PolicyError(f"action keys have different horizons {sorted(lengths)}")
    return np.concatenate(cols, axis=1)


class Gr00tPolicy:
    """`nod_sim.env_api.Policy` over the Isaac-GR00T ZMQ server.

    One REQ socket, guarded by a lock: the runner calls from one worker thread, and a call
    abandoned after a timeout finishes (or times out) before the next one may use the socket.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 2000,
                 api_token: str | None = None, context: zmq.Context | None = None) -> None:
        self.host, self.port, self.timeout_ms, self.api_token = host, port, timeout_ms, api_token
        self._ctx = context or zmq.Context.instance()
        self._lock = threading.Lock()
        self._sock: zmq.Socket | None = None
        self._init_socket()

    @classmethod
    def from_url(cls, url: str, **kw) -> "Gr00tPolicy":
        u = urlparse(url if "://" in url else f"tcp://{url}")
        return cls(u.hostname or "127.0.0.1", u.port or 5555, **kw)

    def _init_socket(self) -> None:
        if self._sock is not None:
            self._sock.close(0)
        s = self._ctx.socket(zmq.REQ)
        s.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        s.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(f"tcp://{self.host}:{self.port}")
        self._sock = s

    def call(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        req: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        if self.api_token:
            req["api_token"] = self.api_token
        with self._lock:
            assert self._sock is not None
            try:
                self._sock.send(pack(req))
                reply = self._sock.recv()
            except zmq.Again as exc:
                self._init_socket()  # a REQ socket that missed its reply cannot send again
                raise PolicyTimeout(f"no reply from {self.host}:{self.port} within {self.timeout_ms} ms") from exc
            except zmq.ZMQError as exc:
                self._init_socket()
                raise PolicyError(f"zmq error talking to the policy server: {exc}") from exc
        if reply == b"ERROR":
            raise PolicyError("server error: make sure the policy server runs the LIBERO checkpoint")
        msg = unpack(reply)
        if isinstance(msg, dict) and "error" in msg:
            raise PolicyError(f"server error: {msg['error']}")
        return msg

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        reply = self.call("get_action", {"observation": batch_observation(obs), "options": None})
        if not isinstance(reply, (list, tuple)) or not reply or not isinstance(reply[0], dict):
            raise PolicyError(f"unexpected get_action reply of type {type(reply).__name__}")
        return chunk_from_response(reply[0])

    def ping(self) -> bool:
        try:
            self.call("ping", requires_input=False)
            return True
        except PolicyError:
            return False

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(0)
            self._sock = None
