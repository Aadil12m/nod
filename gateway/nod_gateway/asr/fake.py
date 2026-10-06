"""FakeASR (mock mode, 4.4): canned transcripts, returned in order and cycled.

Typed text (`utter.text`) bypasses ASR entirely, so the "text box" path needs nothing here.
"""

from __future__ import annotations

import time

from nod_protocol import Transcript

from .base import ASRError


class FakeASR:
    def __init__(self, transcripts: list[str] | None = None) -> None:
        self.transcripts = list(transcripts or ["I'm a bit hungry"])
        self.i = 0
        self.calls = 0

    async def transcribe(self, pcm16: bytes, sr: int = 16000) -> Transcript:
        t0 = time.perf_counter()
        self.calls += 1
        if len(pcm16) % 2:
            raise ASRError("PCM16 audio must have an even number of bytes")
        text = self.transcripts[self.i % len(self.transcripts)]
        self.i += 1
        return Transcript(utt_id=0, text=text, asr_ms=round((time.perf_counter() - t0) * 1000, 2))
