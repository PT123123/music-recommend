"""Dynamic Feed engine (spec 39-41, 65-66).

Reuses the same retrieval/ranking core as song->song and category; the only
difference is that the query vector comes from the time-decayed interest model.

Ranking combines, all in [0,1]:
  preference (affinity to current interest)
  + similarity (metadata region closeness)
  + novelty   (distance from recent history)
  + diversity (MMR: distance from already-selected items this batch)
so the feed never collapses into one tiny region (spec 39).
"""
from __future__ import annotations

import sqlite3
import time

import numpy as np

from ..database import behavior
from ..recommendation.space import MusicSpace
from ..utils.config import Config, get_config
from ..utils.logging import get_logger
from .model import INTEREST_DIMS, InterestModel

log = get_logger()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class FeedEngine:
    def __init__(self, conn: sqlite3.Connection, faiss_store, space: MusicSpace, cfg: Config | None = None):
        self.conn = conn
        self.faiss = faiss_store
        self.space = space
        self.cfg = cfg or get_config()
        self.feed = _feed_cfg(self.cfg)
        if not self.feed:
            raise RuntimeError("missing 'feed' section in config.yaml")
        self.model = InterestModel(conn, space, self.cfg)
        self.embeddings = self.model.embeddings

    # ---- helpers ----
    def _metadata_affinity(self, tid: str, meta_interest: dict) -> float | None:
        terms = []
        for d, target in meta_interest.items():
            col = INTEREST_DIMS.get(d)
            if not col:
                continue
            p = self.space.percentile(tid, col)
            if p is not None:
                terms.append(1.0 - abs(p - target))
        return float(np.mean(terms)) if terms else None

    def next(self, user_id: str, limit: int = 20,
             exclude_track_ids: list[str] | None = None) -> list[dict]:
        now = time.time()
        interest = self.model.current(user_id, now=now)
        exclude = set(exclude_track_ids or [])
        recent = behavior.recent_track_ids(self.conn, user_id, int(self.feed["recent_history_window"]))
        exclude |= set(recent)

        if not interest["has_history"]:
            picks = self._cold_start(limit + len(exclude))
            picks = [t for t in picks if t not in exclude][:limit]
            return [{"track_id": t, "score": None, "reason": "cold_start_diverse"} for t in picks]

        # candidate pool
        pool: dict[str, float] = {}
        emb_i = interest["embedding"]
        if self.faiss is not None and emb_i is not None:
            for tid, cosv in self.faiss.search(emb_i, int(self.cfg.retrieval.get("faiss_top_k", 200))):
                if not tid.startswith("__missing"):
                    pool[tid] = (cosv + 1) / 2
        else:
            pool = {tid: 0.5 for tid in self.space.rows}

        rw = self.feed["ranking_weights"]
        scored = []
        for tid, pref_emb in pool.items():
            if tid in exclude:
                continue
            meta_aff = self._metadata_affinity(tid, interest["metadata"])
            preference = pref_emb if meta_aff is None else 0.6 * pref_emb + 0.4 * meta_aff
            sim = meta_aff if meta_aff is not None else pref_emb
            novelty = 1.0 - max([_cos(self.embeddings[tid], self.embeddings[r])
                                 for r in recent if r in self.embeddings and tid in self.embeddings], default=0.0) \
                if tid in self.embeddings else 1.0
            scored.append([tid, preference, sim, novelty, novelty])  # 5th slot = diversity, recomputed by MMR

        # MMR-style greedy selection with a diversity term
        selected: list[dict] = []
        sel_ids: list[str] = []
        base = {t: (p, s, n) for t, p, s, n, _ in scored}
        remaining = [t for t, *_ in scored]
        while len(selected) < limit and remaining:
            best, best_score = None, -1e9
            for tid in remaining:
                p, s, n = base[tid]
                div = 1.0 - max([_cos(self.embeddings[tid], self.embeddings[x])
                                 for x in sel_ids if tid in self.embeddings and x in self.embeddings], default=0.0)
                score = rw["preference"] * p + rw["similarity"] * s + rw["novelty"] * n + rw["diversity"] * div
                if score > best_score:
                    best, best_score = tid, score
            if best is None:
                break
            selected.append({"track_id": best, "score": round(best_score, 4), "reason": "interest"})
            sel_ids.append(best)
            remaining.remove(best)
        return selected

    def _cold_start(self, k: int) -> list[str]:
        """Diverse farthest-point sample across embedding space; NOT external popularity (spec 66)."""
        ids = [t for t in self.embeddings]
        if len(ids) <= k:
            return ids
        # seed with the track closest to the centroid, then farthest-point traversal
        M = np.stack([self.embeddings[t] for t in ids])
        centroid = M.mean(axis=0)
        start = ids[int(np.argmin([np.linalg.norm(M[i] - centroid) for i in range(len(ids))]))]
        chosen = [start]
        for _ in range(k - 1):
            # maximize (min cosine distance to chosen) == minimize (min cosine similarity)
            farthest = max((t for t in ids if t not in chosen),
                           key=lambda t: -min(_cos(self.embeddings[t], self.embeddings[c]) for c in chosen))
            chosen.append(farthest)
        return chosen


def _feed_cfg(cfg: Config) -> dict:
    from ..utils.config import _load_yaml
    return _load_yaml("config.yaml").get("feed", {})
