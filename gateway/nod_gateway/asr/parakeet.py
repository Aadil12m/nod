"""Parakeet TDT 0.6B v3 (int8) on CPU through onnx-asr (design 3.1, 7.6).

onnx-asr is imported lazily the first time audio arrives, so the gateway starts without it
in mock mode. Install with `pip install 'nod-gateway[asr]'` (onnx-asr[cpu,hub]); the model
(about 670 MB) downloads from Hugging Face on first load.

The call shape `load_model("nemo-parakeet-tdt-0.6b-v3", quantization="int8")` and
`model.recognize(waveform, sample_rate=sr)` follows the onnx-asr 0.12 README; it is
exercised here only against a fake module. CPU latency for a 3 s command is unverified
(design 10.2).
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import numpy as np
from nod_protocol import Transcript

from .base import ASRError

MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
MAX_SECONDS = 25.0  # onnx-asr handles clips up to about 20-30 s


class ParakeetASR:
    def __init__(self, model_name: str = MODEL_NAME, quantization: str | None = "int8") -> None:
        self.model_name = model_name
        self.quantization = quantization
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    import onnx_asr
                except ImportError as exc:
                    raise ASRError("onnx-asr is not installed: pip install 'nod-gateway[asr]'") from exc
                kwargs = {"quantization": self.quantization} if self.quantization else {}
                self._model = onnx_asr.load_model(self.model_name, **kwargs)
            return self._model

    def _recognize(self, pcm16: bytes, sr: int) -> str:
        model = self._load()
        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        if audio.size == 0:
            return ""
        if audio.size > MAX_SECONDS * sr:
            audio = audio[: int(MAX_SECONDS * sr)]
        out = model.recognize(audio, sample_rate=sr)
        if isinstance(out, (list, tuple)):  # batch API returns a list
            out = out[0] if out else ""
        return str(getattr(out, "text", out)).strip()

    async def transcribe(self, pcm16: bytes, sr: int = 16000) -> Transcript:
        if len(pcm16) % 2:
            raise ASRError("PCM16 audio must have an even number of bytes")
        t0 = time.perf_counter()
        # CPU-bound: run in a thread so the event loop (and the e-stop path) never waits on it.
        text = await asyncio.to_thread(self._recognize, pcm16, sr)
        return Transcript(utt_id=0, text=text, asr_ms=round((time.perf_counter() - t0) * 1000, 1))
