"""Structure function-label adapter (All-In-One Music Structure Analyzer, or later a
locally trained equivalent) — P2, optional, off by default.

What the adapter is ALLOWED to promise once a model is actually installed:
  - boundary times, functional labels from the model's own label set, per-label
    confidence, and the model version;
  - labels absent from the model's public label set are reported as unsupported
    (e.g. a model without pre-chorus must never emit one).

What it must NEVER do: invent labels, claim instrument identities, or override
the A/B/C repeat grouping, which stays the independent fact layer.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from . import DeepModelAdapter, NotAvailable

# All-In-One's public label set (mir-aidj/all-in-one). Anything outside this
# set stays "unknown" here — we do not translate model jargon into instrument
# or section claims the model never made.
SUPPORTED_LABELS = {"intro", "verse", "chorus", "bridge", "inst", "outro", "pre-chorus"}
_WEIGHT_CANDIDATES = ("checkpoint.ckpt", "model.ckpt")


class AllInOneStructureLabeler(DeepModelAdapter):
    name = "allinone_structure"
    version = "allinone-v0"

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = Path(models_dir) if models_dir else self._default_dir()

    @staticmethod
    def _default_dir() -> Path:
        from ..utils.config import get_config
        cfg = get_config()
        return Path(cfg.deep.get("structure_models_dir", "models/allinone")) \
            if hasattr(cfg, "deep") else Path("models/allinone")

    def available(self) -> bool:
        if importlib.util.find_spec("allinone") is None:
            return False
        return any((self.models_dir / w).exists() for w in _WEIGHT_CANDIDATES)

    def analyze(self, audio_path: str, sr: int) -> dict:
        """Run the labelled-structure model. Raises NotAvailable when unusable."""
        if not self.available():
            raise NotAvailable("allinone package or weights not installed")
        try:
            import allinone  # noqa: F401  (only meaningful when actually installed)
        except Exception as exc:  # pragma: no cover - depends on the optional install
            raise NotAvailable(f"allinone import failed: {exc}") from exc
        raise NotAvailable("runtime wiring for allinone is not part of this build; "
                           "wire the model call here after a local performance check")
