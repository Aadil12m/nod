"""FakeVerifier (mock mode, 4.4).

Answers from the oracle world (the same atoms the sim publishes), so mock runs show the
done / failed path end to end. `script` forces answers for the next checks (tests, or to
demo the "uncertain" path); `recorded` maps (skill_id, phase) to a verdict recorded with a
real run in the mock library.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from nod_protocol import CosmosCheck, CosmosResult, Frame, WorldState, unmet

from ..registry import Registry
from ..validator import oracle_success
from .base import decision_for


class FakeVerifier:
    model = "fake-cosmos"

    def __init__(
        self,
        reg: Registry,
        world: Callable[[], WorldState | None],
        recorded: dict[tuple[str, str], tuple[str, str]] | None = None,
    ) -> None:
        self.reg = reg
        self.world = world
        self.recorded = dict(recorded or {})
        self.script: list[str] = []  # answers to return next, in order (yes/no/unsure)
        self.calls: list[CosmosCheck] = []

    async def check(self, req: CosmosCheck, images: Sequence[Frame] = ()) -> CosmosResult:
        t0 = time.perf_counter()
        self.calls.append(req)
        if self.script:
            answer, reason = self.script.pop(0), "scripted answer (mock)"
        elif (req.skill_id, req.phase) in self.recorded:
            answer, reason = self.recorded[(req.skill_id, req.phase)]
        else:
            answer, reason = self._from_oracle(req)
        return CosmosResult(
            check_id=req.check_id,
            answer=answer,
            reason=reason,  # type: ignore[arg-type]
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            decision=decision_for(req.phase, answer),  # type: ignore[arg-type]
        )

    def _from_oracle(self, req: CosmosCheck) -> tuple[str, str]:
        spec = self.reg.get(req.skill_id)
        world = self.world()
        if spec is None or world is None:
            return "unsure", "no view of the scene (mock)"
        if req.phase == "pre":
            missing = unmet(spec.pre, world.atoms)
            return (
                ("yes", "The scene matches (mock, from the oracle).")
                if not missing
                else ("no", "The scene does not match (mock, from the oracle).")
            )
        ok = oracle_success(spec, world) is not False
        return (
            ("yes", "It looks done (mock, from the oracle).")
            if ok
            else ("no", "It does not look done (mock, from the oracle).")
        )
