"""Speech recognition: Parakeet TDT 0.6B v3 via onnx-asr (live) and FakeASR (mock)."""

from .base import ASR, ASRError
from .fake import FakeASR

__all__ = ["ASR", "ASRError", "FakeASR"]
