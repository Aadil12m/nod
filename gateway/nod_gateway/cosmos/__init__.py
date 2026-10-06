"""Cosmos-Reason2 verifier: vLLM OpenAI-compatible server (live) and FakeVerifier (mock)."""

from .base import Verifier, VerifierError, decision_for
from .fake import FakeVerifier

__all__ = ["FakeVerifier", "Verifier", "VerifierError", "decision_for"]
