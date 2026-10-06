"""Latency histograms (GET /api/metrics, `metrics` messages) and component health (GET /healthz)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from nod_protocol import Metrics, now_ms


class Histogram:
    """Keeps the last `size` samples; percentiles by nearest rank."""

    def __init__(self, size: int = 1000) -> None:
        self.samples: deque[float] = deque(maxlen=size)
        self.count = 0

    def add(self, value: float | None) -> None:
        if value is None or not math.isfinite(value):
            return
        self.samples.append(float(value))
        self.count += 1

    def pct(self, q: float) -> float | None:
        if not self.samples:
            return None
        s = sorted(self.samples)
        k = max(0, min(len(s) - 1, math.ceil(q / 100.0 * len(s)) - 1))
        return round(s[k], 1)

    def summary(self) -> dict:
        return {
            "n": self.count,
            "p50": self.pct(50),
            "p95": self.pct(95),
            "max": round(max(self.samples), 1) if self.samples else None,
            "recent": [round(v, 1) for v in list(self.samples)[-20:]],
        }


class MetricsHub:
    NAMES = (
        "estop_ms",
        "estop_phone_gw_ms",
        "estop_gw_sim_ms",
        "estop_sim_halt_ms",
        "estop_gw_fwd_ms",
        "view_lag_ms",
        "policy_ms",
        "asr_ms",
        "llm_ms",
        "cosmos_ms",
    )

    def __init__(self) -> None:
        self.h = {n: Histogram() for n in self.NAMES}

    def add(self, name: str, value: float | None) -> None:
        self.h[name].add(value)

    def summary(self) -> dict:
        return {n: h.summary() for n, h in self.h.items()}

    def message(self) -> Metrics:
        return Metrics(
            estop_ms_p50=self.h["estop_ms"].pct(50),
            estop_ms_p95=self.h["estop_ms"].pct(95),
            view_lag_ms=self.h["view_lag_ms"].pct(50),
            policy_ms_p95=self.h["policy_ms"].pct(95),
            asr_ms_p50=self.h["asr_ms"].pct(50),
            llm_ms_p50=self.h["llm_ms"].pct(50),
        )


@dataclass
class ComponentHealth:
    ok: bool | None = None  # None: not used / not seen yet
    last_error: str | None = None
    last_error_ms: int | None = None
    last_ok_ms: int | None = None
    detail: dict = field(default_factory=dict)


class Health:
    COMPONENTS = ("sim", "policy", "cosmos", "token_factory", "hass", "asr")

    def __init__(self) -> None:
        self.c = {n: ComponentHealth() for n in self.COMPONENTS}

    def ok(self, name: str, **detail) -> None:
        h = self.c[name]
        h.ok, h.last_ok_ms = True, now_ms()
        h.detail.update(detail)

    def error(self, name: str, err: str, **detail) -> None:
        h = self.c[name]
        h.ok, h.last_error, h.last_error_ms = False, err, now_ms()
        h.detail.update(detail)

    def view(self) -> dict:
        return {
            n: {
                "ok": h.ok,
                "last_error": h.last_error,
                "last_error_ms": h.last_error_ms,
                "last_ok_ms": h.last_ok_ms,
                **h.detail,
            }
            for n, h in self.c.items()
        }
