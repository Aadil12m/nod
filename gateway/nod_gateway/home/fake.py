"""FakeHome (mock mode, 4.4): logs the action, keeps a light state, always succeeds."""

from __future__ import annotations

import logging

from nod_protocol import HomeResult

log = logging.getLogger(__name__)


class FakeHome:
    def __init__(self) -> None:
        self.lights_on = False
        self.calls: list[str] = []

    async def call(self, action: str) -> HomeResult:
        self.calls.append(action)
        if action == "lights_on":
            self.lights_on = True
            detail = "Lights on (mock)"
        elif action == "lights_off":
            self.lights_on = False
            detail = "Lights off (mock)"
        elif action == "call_caregiver":
            detail = "Caregiver notified (mock)"
        else:
            return HomeResult(action="call_caregiver", ok=False, detail=f"unknown action {action!r}")
        log.info("FakeHome: %s", detail)
        return HomeResult(action=action, ok=True, detail=detail, latency_ms=0.0)  # type: ignore[arg-type]
