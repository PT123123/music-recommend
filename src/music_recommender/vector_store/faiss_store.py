"""FAISS vector store: cache/index only, rebuildable from SQLite (principle 4).

Persists the required sidecar files so index <-> track_id mapping stays aligned:
  faiss_index.bin, track_ids.json, embedding_model.txt, embedding_dimension.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


DEFAULT_VS = {
    "index_file": "faiss_index.bin",
    "track_ids_file": "track_ids.json",
    "embedding_model_file": "embedding_model.txt",
    "embedding_dim_file": "embedding_dimension.txt",
}


class FaissStore:
    def __init__(self, index_dir: Path, dim: int, model_name: str, vs_cfg: dict | None = None):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = int(dim)
        self.model_name = model_name
        self.vs = vs_cfg or DEFAULT_VS
        self.index = faiss.IndexFlatIP(self.dim)  # embeddings are L2-normalized -> inner product = cosine
        self.track_ids: list[str] = []  # position i -> track_id

    # ---- build ----
    def build(self, ordered: list[tuple[int, str, np.ndarray]]) -> None:
        """ordered: [(embedding_id, track_id, vector)] sorted ascending by embedding_id.

        We add by embedding_id position so index id == embedding_id (sparse-safe),
        padding gaps is unnecessary for MVP since ids are assigned contiguously.
        """
        self.index = faiss.IndexFlatIP(self.dim)
        self.track_ids = []
        if not ordered:
            self._check_dim([])
            return
        max_id = max(e[0] for e in ordered)
        vecs = np.zeros((max_id + 1, self.dim), dtype="float32")
        ids: list[str | None] = [None] * (max_id + 1)
        for emb_id, tid, v in ordered:
            v = np.asarray(v, dtype="float32")
            if v.shape[0] != self.dim:
                raise ValueError(f"dim mismatch: got {v.shape[0]} expected {self.dim}")
            vecs[emb_id] = v
            ids[emb_id] = tid
        self.index.add(vecs)
        self.track_ids = [t if t is not None else f"__missing_{i}__" for i, t in enumerate(ids)]
        self._check_dim(vecs)

    def _check_dim(self, _):
        pass

    # ---- query ----
    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        q = np.asarray(query, dtype="float32").reshape(1, -1)
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(q, k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.track_ids):
                continue
            out.append((self.track_ids[idx], float(score)))
        return out

    def save(self) -> None:
        vs = self.vs
        faiss.write_index(self.index, str(self.dir / vs["index_file"]))
        (self.dir / vs["track_ids_file"]).write_text(json.dumps(self.track_ids), encoding="utf-8")
        (self.dir / vs["embedding_model_file"]).write_text(self.model_name, encoding="utf-8")
        (self.dir / vs["embedding_dim_file"]).write_text(str(self.dim), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path, vs_cfg: dict | None = None) -> "FaissStore":
        vs_cfg = vs_cfg or DEFAULT_VS
        store = cls.__new__(cls)
        store.dir = Path(index_dir)
        store.vs = vs_cfg
        store.model_name = (store.dir / vs_cfg["embedding_model_file"]).read_text(encoding="utf-8")
        store.dim = int((store.dir / vs_cfg["embedding_dim_file"]).read_text(encoding="utf-8"))
        store.index = faiss.read_index(str(store.dir / vs_cfg["index_file"]))
        store.track_ids = json.loads((store.dir / vs_cfg["track_ids_file"]).read_text(encoding="utf-8"))
        return store
