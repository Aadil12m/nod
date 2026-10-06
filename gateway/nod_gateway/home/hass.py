"""Home Assistant REST client (design 3.1): demo lights plus a notify target.

  lights_on / lights_off -> POST /api/services/light/turn_on|turn_off {"entity_id": <light>}
  call_caregiver         -> POST /api/services/notify/<service> {"title": "Nod", "message": ...}

Authenticated with a long-lived access token (`Authorization: Bearer`). Exercised here only
against a fake HTTP transport.
"""

from __future__ import annotations

import time

import httpx
from nod_protocol import HomeResult

CAREGIVER_MESSAGE = "Nod: the user is asking for help."


class HassHome:
    def __init__(self, cfg, http: httpx.AsyncClient) -> None:
        if not cfg.hass_url or not cfg.hass_token:
            raise ValueError("Home Assistant needs NOD_HASS_URL and NOD_HASS_TOKEN")
        self.base = cfg.hass_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {cfg.hass_token}", "Content-Type": "application/json"}
        self.light = cfg.hass_light_entity
        self.notify = cfg.hass_notify_service
        self.http = http
        self.last_error: str | None = None

    def _request(self, action: str) -> tuple[str, dict]:
        if action == "lights_on":
            return "/api/services/light/turn_on", {"entity_id": self.light}
        if action == "lights_off":
            return "/api/services/light/turn_off", {"entity_id": self.light}
        if action == "call_caregiver":
            return f"/api/services/notify/{self.notify}", {"title": "Nod", "message": CAREGIVER_MESSAGE}
        raise ValueError(f"unknown home action {action!r}")

    async def call(self, action: str) -> HomeResult:
        path, body = self._request(action)
        t0 = time.perf_counter()
        try:
            r = await self.http.post(self.base + path, json=body, headers=self.headers, timeout=5.0)
            ok = 200 <= r.status_code < 300
            detail = "ok" if ok else f"HTTP {r.status_code}: {r.text[:200]}"
        except httpx.HTTPError as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        self.last_error = None if ok else detail
        return HomeResult(
            action=action,
            ok=ok,
            detail=detail,  # type: ignore[arg-type]
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
