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

    def _matrix(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Embeddings stacked in a stable track order, plus each row's norm.

        The MMR loop used to call np.linalg.norm once per (candidate, already-selected)
        pair -- 150k numpy scalar calls for one 20-item batch on a 93-track library, so
        numpy call overhead rather than arithmetic was the whole cost. One row of dots
        per picked item replaces it while keeping the same per-pair expression.
        """
        ids = list(self.embeddings)
        if not ids:
            return [], np.zeros((0, 0)), np.zeros(0)
        M = np.stack([np.asarray(self.embeddings[t], float) for t in ids])
        return ids, M, np.linalg.norm(M, axis=1)

    @staticmethod
    def _sims_to(M: np.ndarray, nrm: np.ndarray, rows: np.ndarray, k: int) -> np.ndarray:
        """Cosine of each row in `rows` against matrix row k (the pairwise formula)."""
        return (M[rows] @ M[k]) / (nrm[rows] * nrm[k] + 1e-9)

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

        ids, M, nrm = self._matrix()
        pos = {t: i for i, t in enumerate(ids)}
        cands = [t for t in pool if t not in exclude]
        if not cands:
            return []
        pref_emb = np.asarray([pool[t] for t in cands], float)
        meta_aff = [self._metadata_affinity(t, interest["metadata"]) for t in cands]
        preference = np.asarray([p if m is None else 0.6 * p + 0.4 * m for p, m in zip(pref_emb, meta_aff)], float)
        similarity = np.asarray([m if m is not None else p for m, p in zip(meta_aff, pref_emb)], float)

        # candidate slot -> embedding-matrix slot; tracks without an embedding keep
        # novelty/diversity at 1.0, exactly as the per-pair loop's default did.
        in_matrix = np.asarray([t in pos for t in cands])
        cidx = np.asarray([pos.get(t, 0) for t in cands], int)
        novelty = np.ones(len(cands))
        recent_idx = [pos[t] for t in recent if t in pos]
        if recent_idx and M.shape[0] and in_matrix.any():
            rows = cidx[in_matrix]
            sims = (M[rows] @ M[recent_idx].T) / (np.outer(nrm[rows], nrm[recent_idx]) + 1e-9)
            novelty[in_matrix] = 1.0 - sims.max(axis=1)

        rw = self.feed["ranking_weights"]
        alive = np.ones(len(cands), bool)
        dmax = np.zeros(len(ids))       # max cosine to the items already selected
        have_sel = False                # before the first pick, diversity is exactly 1.0
        selected: list[dict] = []
        while len(selected) < limit and alive.any():
            # 1 - cos can exceed 1 when a candidate is anti-correlated with the picks;
            # clamping the running max at 0 here silently flattened whole batches.
            div = np.where(in_matrix & have_sel, 1.0 - dmax[cidx], 1.0)
            scores = (rw["preference"] * preference + rw["similarity"] * similarity
                      + rw["novelty"] * novelty + rw["diversity"] * div)
            scores[~alive] = -np.inf
            j = int(np.argmax(scores))
            selected.append({"track_id": cands[j], "score": round(float(scores[j]), 4), "reason": "interest"})
            alive[j] = False
            if M.shape[0] and in_matrix[j]:
                v = self._sims_to(M, nrm, np.arange(len(ids)), cidx[j])
                dmax = v if not have_sel else np.maximum(dmax, v)
                have_sel = True
        return selected

    def _cold_start(self, k: int) -> list[str]:
        """Diverse farthest-point sample across embedding space; NOT external popularity (spec 66)."""
        ids = list(self.embeddings)
        if len(ids) <= k:
            return ids
        mid, M, nrm = self._matrix()
        centroid = M.mean(axis=0)
        # seed with the track closest to the centroid, then farthest-point traversal
        j = int(np.argmin(np.linalg.norm(M - centroid, axis=1)))
        chosen = [ids[j]]
        minsim = self._sims_to(M, nrm, np.arange(len(ids)), j)
        unchosen = np.ones(len(ids), bool)
        unchosen[j] = False
        for _ in range(k - 1):
            cand = np.where(unchosen, minsim, np.inf)
            j = int(np.argmin(cand))              # smallest similarity to the chosen set == farthest point
            chosen.append(ids[j])
            unchosen[j] = False
            minsim = np.minimum(minsim, self._sims_to(M, nrm, np.arange(len(ids)), j))
        return chosen


def _feed_cfg(cfg: Config) -> dict:
    from ..utils.config import _load_yaml
    return _load_yaml("config.yaml").get("feed", {})
