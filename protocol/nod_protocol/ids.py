"""ULIDs and wall-clock helpers.

A ULID is 26 Crockford base32 characters: 48 bits of Unix time in ms followed by
80 random bits. ULIDs sort by creation time as plain strings. `new_ulid()` is
monotonic within one process: two ids made in the same millisecond still sort in
creation order (the random part is incremented, as the ULID spec suggests).
"""

from __future__ import annotations

import os
import threading
import time

CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_DECODE = {c: i for i, c in enumerate(CROCKFORD)}
_MAX_TS = (1 << 48) - 1
_MAX_RAND = (1 << 80) - 1

_lock = threading.Lock()
_last_ms = -1
_last_rand = 0


def now_ms() -> int:
    """Sender wall clock in integer milliseconds since the Unix epoch (envelope `ts`)."""
    return time.time_ns() // 1_000_000


def _encode(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(out))


def ulid_from_parts(ts_ms: int, rand: int) -> str:
    """Build a ULID string from a 48-bit timestamp and an 80-bit random value."""
    if not 0 <= ts_ms <= _MAX_TS:
        raise ValueError("ULID timestamp out of range")
    if not 0 <= rand <= _MAX_RAND:
        raise ValueError("ULID random part out of range")
    return _encode((ts_ms << 80) | rand, 26)


def new_ulid(ts_ms: int | None = None) -> str:
    """Return a new ULID. Monotonic within this process for ids made without `ts_ms`."""
    global _last_ms, _last_rand
    if ts_ms is not None:
        return ulid_from_parts(ts_ms, int.from_bytes(os.urandom(10), "big"))
    with _lock:
        ms = now_ms()
        if ms <= _last_ms:
            # Same (or earlier, if the clock stepped back) millisecond: keep the last
            # timestamp and increment the random part so ordering is preserved.
            ms = _last_ms
            rand = _last_rand + 1
            if rand > _MAX_RAND:  # astronomically unlikely; move to the next ms
                ms, rand = ms + 1, int.from_bytes(os.urandom(10), "big")
        else:
            rand = int.from_bytes(os.urandom(10), "big")
        _last_ms, _last_rand = ms, rand
        return ulid_from_parts(ms, rand)


def is_ulid(text: str) -> bool:
    """True if `text` is a syntactically valid ULID (26 Crockford chars, first char <= 7)."""
    return len(text) == 26 and all(c in _DECODE for c in text) and text[0] <= "7"


def ulid_time_ms(ulid: str) -> int:
    """Extract the millisecond timestamp from a ULID."""
    if not is_ulid(ulid):
        raise ValueError(f"not a ULID: {ulid!r}")
    value = 0
    for c in ulid[:10]:
        value = (value << 5) | _DECODE[c]
    return value


def new_id(prefix: str) -> str:
    """Prefixed id as used in the design: p_ (proposal), r_ (run), k_ (cosmos check),
    q_ (ZMQ request), n_ (nonce), s_ (session), o (option ids are short and local)."""
    return f"{prefix}_{new_ulid()}"
