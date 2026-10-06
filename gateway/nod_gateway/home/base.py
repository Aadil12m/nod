"""Home interface (4.2.2)."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from nod_protocol import HomeResult

HomeActionName = Literal["lights_on", "lights_off", "call_caregiver"]


@runtime_checkable
class Home(Protocol):
    async def call(self, action: HomeActionName) -> HomeResult: ...
