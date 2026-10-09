"""Optional deep-analysis model adapters (expansion doc section 9 / P2).

Contract, shared by every adapter:

  - `available()` must answer WITHOUT downloading anything: it only checks
    imports / weight files / config. Network installs are a user decision.
  - `analyze()` raises `NotAvailable` (caught by the extractor) instead of
    half-failing; a deep-analysis failure keeps the base result intact.
  - results always carry `label_source`, model version and per-label
    confidence; labels the model does not support are reported as unsupported,
    never guessed.

Deep analysis runs only on selected tracks (`analysis.deep.enabled` is false
by default): no library-wide model downloads or runs.
"""
from __future__ import annotations


class NotAvailable(RuntimeError):
    """The adapter is installed but its model/weights are missing, or the
    environment (GPU/runtime) cannot serve a request right now."""


class DeepModelAdapter:
    name = "abstract"
    version = "v0"

    def available(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def status(self) -> dict:
        if self.available():
            return {"module": self.name, "status": "ready", "model_version": self.version}
        return {"module": self.name, "status": "not_installed",
                "note": "模型未安装或依赖缺失；本引擎不做隐式下载"}
