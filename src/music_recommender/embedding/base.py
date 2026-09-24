"""AudioEmbedder abstraction (spec section 24, principle 2).

Concrete backends (PCA-stats, PANNs, MERT) are swappable; core code depends
only on this interface so no model is hard-wired.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AudioEmbedder(ABC):
    name: str = "abstract"

    @abstractmethod
    def fit(self, vectors: np.ndarray) -> "AudioEmbedder":
        ...

    @abstractmethod
    def embed(self, vector: np.ndarray) -> np.ndarray:
        """Return a single L2-normalized embedding for one feature vector."""
        ...

    @abstractmethod
    def embed_many(self, vectors: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...
