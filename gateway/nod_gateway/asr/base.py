"""ASR interface (4.2.2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nod_protocol import Transcript


class ASRError(Exception):
    pass


@runtime_checkable
class ASR(Protocol):
    async def transcribe(self, pcm16: bytes, sr: int = 16000) -> Transcript:
        """Transcribe little-endian PCM16 mono audio. `Transcript.utt_id` is set by the caller."""
        ...
