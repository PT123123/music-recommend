"""Recommendation core: shared by song->song, category and (later) feed (spec 42).

Flow (spec 26-28):
  seed -> FAISS coarse recall (Top 200) -> metadata re-rank -> weighted total -> Top N.
The only thing that changes between modes is where the query vector comes from.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np

from ..utils.config import Config, get_config
from ..utils.logging import get_logger
from .space import GROUP_REASON, get_music_space

log = get_logger()

REASON_MIN_SIM = 0.60


@dataclass
class Recommendation:
    track_id: str
    score: float
    reasons: list[str] = field(default_factory=list)
    breakdown: dict[str, float] = field(default_factory=dict)


class Recommender:
    def __init__(self, conn: sqlite3.Connection, faiss_store, weights: dict[str, float] | None = None,
                 cfg: Config | None = None):
        self.conn = conn
        self.faiss = faiss_store
        self.cfg = cfg or get_config()
        self.weights = weights or self.cfg.active_weights()
        self.space = get_music_space(conn)

    # ---- query construction ----
    def _seed_vector(self, track_id: str) -> np.ndarray | None:
        if self.faiss is None:
            return None
        try:
            idx = self.faiss.track_ids.index(track_id)
        except ValueError:
            return None
        return self.faiss.index.reconstruct(int(idx))

    def _candidate_pool(self, track_id: str, top_k: int) -> dict[str, float]:
        """Return {candidate_track_id: embedding_similarity[0,1]} via FAISS coarse recall."""
        q = self._seed_vector(track_id)
        if q is None:
            return {}
        hits = self.faiss.search(q, top_k + 1)
        pool = {}
        for tid, cos in hits:
            if tid == track_id or tid.startswith("__missing"):
                continue
            pool[tid] = float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))
        return pool

    def _score_pair(self, seed_id: str, cand_id: str, emb_sim: float | None) -> Recommendation:
        # Each group similarity is computed exactly once and reused for the score, the
        # breakdown and the reasons; scoring it twice per pair doubled every rerank cost.
        detail: dict[str, float | None] = {}
        contributions: dict[str, float] = {}
        for group, weight in self.weights.items():
            if group == "embedding":
                if emb_sim is None:
                    continue
                detail["embedding"] = float(emb_sim)
            else:
                sim = self.space.group_similarity(seed_id, cand_id, group)
                if sim is None:
                    continue
                detail[group] = sim
            contributions[group] = weight * detail[group]
        total_weight = sum(self.weights[g] for g in contributions)
        score = (sum(contributions.values()) / total_weight) if total_weight > 0 else 0.0

        reasons = []
        ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
        for g, _ in ranked:
            if g == "embedding":
                reasons.append("整体音色/听感接近")
            elif detail.get(g) is not None and detail[g] >= REASON_MIN_SIM:
                reasons.append(GROUP_REASON.get(g, g))
        return Recommendation(track_id=cand_id, score=round(float(score), 4),
                              reasons=reasons[:4],
                              breakdown={k: round(v, 3) for k, v in detail.items() if v is not None})

    # ---- public: song -> song ----
    def similar(self, track_id: str, limit: int = 20) -> list[Recommendation]:
        top_k = int(self.cfg.retrieval.get("faiss_top_k", 200))
        pool = self._candidate_pool(track_id, top_k)
        if pool:
            cands = list(pool.items())
        else:
            # FAISS unavailable -> rank whole library by metadata only
            from ..database import repository
            cands = [(r["track_id"], None) for r in repository.all_tracks(self.conn)
                     if r["track_id"] != track_id]
        recs = [self._score_pair(track_id, cand, emb) for cand, emb in cands]
        recs.sort(key=lambda r: r.score, reverse=True)
        return recs[:limit]
