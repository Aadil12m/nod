"""Presenter fan-out (design 3.5.8): trace events and periodic metrics to /ws/presenter.

Every trace also goes to the recorder under its `ref` (proposal or run id), so the
presenter view and `GET /api/runs/{id}` tell the same story.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from nod_protocol import Trace

from .metrics import MetricsHub
from .peers import Outbox
from .recorder import Recorder

log = logging.getLogger(__name__)


class Presenter:
    def __init__(self, out: Outbox, recorder: Recorder, metrics: MetricsHub) -> None:
        self.out = out
        self.recorder = recorder
        self.metrics = metrics
        self.traces: list[Trace] = []

    def trace(self, kind: str, ref: str | None, data: dict[str, Any]) -> None:
        msg = Trace(kind=kind, ref=ref, data=data)  # type: ignore[arg-type]
        self.traces.append(msg)
        if len(self.traces) > 1000:
            del self.traces[:200]
        self.out.to_presenter(msg)
        if ref:
            self.recorder.event(ref, kind, data)

    async def metrics_loop(self, period_s: float = 1.0) -> None:
        while True:
            await asyncio.sleep(period_s)
            if self.out.presenters:
                self.out.to_presenter(self.metrics.message())
