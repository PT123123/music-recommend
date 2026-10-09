"""Source-separation adapter (Demucs or equivalent) — P2, optional, off by default.

Reliable vocal activity, melody range and long-note statistics need a real
vocal stem; the existing voice-band heuristics stay the low-confidence layer.
This adapter defines the single integration point; it runs nothing until a
separation backend is installed locally (Windows/CPU/GPU fit is evaluated
BEFORE any model is chosen, per expansion doc section 9).

Downstream outputs that only become honest WITH a separated vocal stem:
vocal activity intervals, candidate lead-melody pitch, register, long-tone
ratio, syllable onset density, pitch slides. Multi-voice material still
disturbs single-f0 estimation: harmony-layer counts and singer identity stay
out of scope for this adapter, with or without it.
"""
from __future__ import annotations

import importlib.util

from . import DeepModelAdapter, NotAvailable


class DemucsSourceSeparator(DeepModelAdapter):
    name = "demucs_separator"
    version = "demucs-v0"

    def available(self) -> bool:
        return importlib.util.find_spec("demucs") is not None

    def analyze(self, audio_path: str, sr: int) -> dict:
        if not self.available():
            raise NotAvailable("demucs not installed")
        try:
            import demucs  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on the optional install
            raise NotAvailable(f"demucs import failed: {exc}") from exc
        raise NotAvailable("runtime wiring for demucs is not part of this build; "
                           "wire the model call here after a local performance check")
