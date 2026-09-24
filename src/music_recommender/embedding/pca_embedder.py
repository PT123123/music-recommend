"""Lightweight placeholder embedder: standardized statistical features -> PCA.

No large model download; runs on CPU. Produces a low-dim embedding used for
FAISS coarse recall. Swap for PANNs/MERT by implementing AudioEmbedder.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import AudioEmbedder


def _l2_norm(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (emb / norms).astype("float32")


class PcaStatsEmbedder(AudioEmbedder):
    def __init__(self, n_components: int = 24, name: str = "librosa_pca"):
        self.n_components = n_components
        self.name = name
        self._pipeline: Pipeline | None = None

    @property
    def dim(self) -> int:
        self._ensure()
        return int(self._pipeline.named_steps["pca"].n_components_)

    def fit(self, vectors: np.ndarray) -> "PcaStatsEmbedder":
        vectors = np.atleast_2d(vectors).astype("float64")
        if vectors.shape[0] > 1:
            n_comp = max(1, int(min(self.n_components, vectors.shape[0] - 1, vectors.shape[1])))
        else:
            n_comp = 1
        self._pipeline = Pipeline([
            ("scaler", StandardScaler(with_mean=vectors.shape[0] > 1)),
            ("pca", PCA(n_components=n_comp, whiten=vectors.shape[0] > 1, random_state=0)),
        ])
        self._pipeline.fit(vectors)
        return self

    def _ensure(self):
        if self._pipeline is None:
            raise RuntimeError("embedder not fitted; call fit() or load()")

    def embed_many(self, vectors: np.ndarray) -> np.ndarray:
        self._ensure()
        return _l2_norm(self._pipeline.transform(np.atleast_2d(vectors).astype("float64")))

    def embed(self, vector: np.ndarray) -> np.ndarray:
        return self.embed_many(np.atleast_2d(vector))[0]

    def save(self, path: Path) -> None:
        self._ensure()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline, "name": self.name,
                     "n_components": self.n_components}, str(path))

    @classmethod
    def load(cls, path: Path) -> "PcaStatsEmbedder":
        obj = joblib.load(str(path))
        inst = cls(n_components=obj["n_components"], name=obj["name"])
        inst._pipeline = obj["pipeline"]
        return inst
