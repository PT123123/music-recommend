"""Public Recommender façade: loads SQLite + FAISS + embedding pipeline and
exposes high-level recommendation. HTTP API and scripts use this (spec 53).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .database import repository
from .database.schema import connect
from .embedding.pca_embedder import PcaStatsEmbedder
from .recommendation.core import Recommendation  # noqa: F401
from .recommendation.core import Recommender as _CoreRecommender
from .utils.config import Config, get_config
from .utils.logging import get_logger
from .vector_store.faiss_store import FaissStore

log = get_logger()


class Recommender:
    def __init__(self, cfg: Config | None = None, conn: sqlite3.Connection | None = None):
        self.cfg = cfg or get_config()
        self.conn = conn or connect(self.cfg.db_path)
        self.faiss = self._maybe_load_faiss()
        self.embedder = self._maybe_load_embedder()
        self._core = _CoreRecommender(self.conn, self.faiss, cfg=self.cfg)

    def _maybe_load_faiss(self) -> FaissStore | None:
        idx = self.cfg.faiss_dir / self.cfg.vector_store.get("index_file", "faiss_index.bin")
        if idx.exists():
            try:
                return FaissStore.load(self.cfg.faiss_dir, self.cfg.vector_store)
            except Exception as exc:  # FAISS is only a cache; rebuild is allowed (principle 4)
                log.warning("FAISS load failed, falling back to metadata ranking: %s", exc)
        return None

    def _maybe_load_embedder(self) -> PcaStatsEmbedder | None:
        store = self.cfg.abs(self.cfg.embedding.get("model_store", "data/faiss/embedding_pipeline.joblib"))
        return PcaStatsEmbedder.load(store) if Path(store).exists() else None

    # ---- song -> song ----
    def similar(self, track_id: str, limit: int = 20) -> list[Recommendation]:
        return self._core.similar(track_id, limit=limit)

    def similar_by_path(self, file_path: str | Path, limit: int = 20) -> list[Recommendation]:
        """Ad-hoc recommendation from a file not necessarily in the library."""
        from .database.repository import get_track_by_path
        row = get_track_by_path(self.conn, str(Path(file_path).resolve()))
        if row is not None:
            return self.similar(row["track_id"], limit=limit)
        if self.embedder is None or self.faiss is None:
            raise RuntimeError("index/embedder not built; cannot recommend for an unindexed file")
        from .preprocess.audio import normalize
        from .features import extract
        wav, duration, sr = normalize(Path(file_path))
        feats = extract.extract(str(wav), sr, duration)
        qvec = self.embedder.embed(feats.stat_vector)
        hits = self.faiss.search(qvec, limit + 1)
        out = []
        for tid, cos in hits:
            if tid.startswith("__missing"):
                continue
            r = self._core._score_pair(tid, tid, float(np.clip((cos + 1) / 2, 0, 1)))
            out.append(r)
        return out[:limit]

    # ---- category (Phase 4) ----
    def categories(self) -> list[dict]:
        return [{"id": c["id"], "name": c.get("name", c["id"])} for c in self.cfg.categories]

    def _category_engine(self):
        from .recommendation.category import CategoryEngine
        if getattr(self, "_cat_engine", None) is None:
            self._cat_engine = CategoryEngine(self._core.space)
        return self._cat_engine

    def category(self, category_id: str, limit: int = 30) -> tuple[list[dict], dict]:
        cat = next((c for c in self.cfg.categories if c["id"] == category_id), None)
        if cat is None:
            raise KeyError(f"unknown category_id: {category_id}")
        return self._run_category(cat, limit)

    def category_query(self, query: dict, hard_filter: dict | None = None, limit: int = 30) -> tuple[list[dict], dict]:
        return self._run_category({"id": "custom", "query": query, "hard_filter": hard_filter or {}}, limit)

    def _run_category(self, cat: dict, limit: int) -> tuple[list[dict], dict]:
        from pathlib import Path as _P
        engine = self._category_engine()
        results, meta = engine.recommend(cat, limit=limit)
        out = []
        for r in results:
            row = repository.get_track(self.conn, r.track_id)
            out.append({"track_id": r.track_id, "score": r.score,
                        "file_name": _P(row["file_path"]).name if row else r.track_id,
                        "matched_dims": r.matched_dims})
        return out, meta

    # ---- dynamic feed (Phase 5) ----
    def _feed_engine(self):
        from .interest.feed import FeedEngine
        if getattr(self, "_feed", None) is None:
            self._feed = FeedEngine(self.conn, self.faiss, self._core.space, self.cfg)
        return self._feed

    def feedback(self, user_id: str, track_id: str, event: str, timestamp: float | None = None,
                 play_seconds: float | None = None, completion_ratio: float | None = None) -> int:
        from .database import behavior
        return behavior.log_interaction(self.conn, user_id, track_id, event, timestamp,
                                        play_seconds, completion_ratio)

    def feed_next(self, user_id: str, limit: int = 20, exclude_track_ids: list[str] | None = None) -> list[dict]:
        return self._feed_engine().next(user_id, limit=limit, exclude_track_ids=exclude_track_ids)

    def feed_state(self, user_id: str) -> dict:
        return self._feed_engine().model.state_summary(user_id)

    def feed_reset(self, user_id: str, scope: str = "short") -> int:
        from .database import behavior
        return behavior.reset_scope(self.conn, user_id, scope)

