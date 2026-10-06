"""Verifier interface (4.2.2, 3.5.7).

`check` takes the `cosmos.check` request plus the captured frames it refers to (`images`,
an addition: `FrameRef` carries no pixels). The returned `CosmosResult.decision` is the
verifier-only reading; the session recomputes it with the oracle through
`validator.pre_decision` / `validator.post_decision` and fills `oracle` and `agree`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from nod_protocol import CosmosCheck, CosmosResult, Frame


class VerifierError(Exception):
    pass


@runtime_checkable
class Verifier(Protocol):
    async def check(self, req: CosmosCheck, images: Sequence[Frame] = ()) -> CosmosResult: ...


def decision_for(phase: str, answer: str) -> str:
    """Cosmos-only decision: post yes/no/unsure -> done/failed/uncertain; pre -> proceed."""
    if phase == "post":
        return {"yes": "done", "no": "failed"}.get(answer, "uncertain")
    return "proceed"


def question_with_format(question: str) -> str:
    """The 3.5.7 question text with the reply-format instruction appended."""
    return question + ' Reply with JSON {"answer": "yes"|"no"|"unsure", "reason": "..."}'
