"""Time-decayed interest model (spec 35-38).

Weighted aggregation of recent interactions, where each event decays as
    weight = base_weight * exp(-age / tau)
and scope windows split behaviour into short / medium / long. The current interest
is    l1*short + l2*medium + l3*long   with l1 > l2 > l3, so recent preference wins
without erasing long-term taste (spec 38).
"""
from __future__ import annotations

import math
import sqlite3
import time

import numpy as np

from ..database import behavior, repository
from ..recommendation.space import MusicSpace
from ..utils.config import Config, get_config
from ..utils.logging import get_logger

log = get_logger()

# Interpretable music-space dimensions: name -> scalar column (percentile-normalised).
INTEREST_DIMS = {
    "energy": "rms_mean",
    "brightness": "spectral_centroid_mean",
    "tempo": "bpm",
    "rhythm_intensity": "onset_density",
    "danceability": "danceability",
    "vocal_presence": "vocal_ratio",
    "vocal_intensity": "vocal_intensity",
    "vocal_pitch": "vocal_pitch_mean",
    "bass_energy": "bass_energy_ratio",
    "drum_energy": "drum_energy_ratio",
    "pitch": "pitch_mean",
}


class InterestModel:
    def __init__(self, conn: sqlite3.Connection, space: MusicSpace, cfg: Config | None = None):
        self.conn = conn
        self.space = space
        self.cfg = cfg or get_config()
        self.feed = _feed_cfg(self.cfg)
        if not self.feed:
            raise RuntimeError("missing 'feed' section in config.yaml")
        # embedding lookup
        self.embeddings: dict[str, np.ndarray] = {
            tid: vec for _, tid, vec in repository.ordered_embeddings(conn)
        }

    # ---- weights ----
    def _event_weight(self, row, now: float) -> float:
        ev = self.feed["events"].get(row["event"], {})
        base = float(ev.get("base_weight", 0.0))
        tau = float(ev.get("tau", 86400)) or 86400.0
        age = max(0.0, now - float(row["timestamp"]))
        if row["event"] == "play" and row["completion_ratio"] is not None:
            base *= float(row["completion_ratio"])   # partial plays count less (spec 36)
        return base * math.exp(-age / tau)

    def _scope_weights(self, user_id: str, now: float) -> dict[str, dict[str, float]]:
        """Return {scope: {track_id: aggregated_weight}} using cumulative windows."""
        rows = behavior.get_interactions(self.conn, user_id)
        windows = self.feed["scope_windows"]
        short_w, med_w, long_w = {}, {}, {}
        for r in rows:
            age = now - float(r["timestamp"])
            w = self._event_weight(r, now)
            for bucket, win in ((long_w, windows["long"]), (med_w, windows["medium"]), (short_w, windows["short"])):
                if age <= win:
                    bucket[r["track_id"]] = bucket.get(r["track_id"], 0.0) + w
        return {"short": short_w, "medium": med_w, "long": long_w}

    # ---- vectors ----
    def _embedding_interest(self, weights: dict[str, float]) -> np.ndarray | None:
        vecs = [self.embeddings[t] * w for t, w in weights.items() if t in self.embeddings and abs(w) > 1e-9]
        if not vecs:
            return None
        v = np.sum(vecs, axis=0)
        n = np.linalg.norm(v)
        return (v / n) if n > 0 else None

    def _metadata_interest(self, weights: dict[str, float]) -> dict[str, float]:
        num = {d: 0.0 for d in INTEREST_DIMS}
        den = 0.0
        for t, w in weights.items():
            if abs(w) < 1e-9 or t not in self.space.rows:
                continue
            den += abs(w)
            for d, col in INTEREST_DIMS.items():
                p = self.space.percentile(t, col)
                if p is not None:
                    num[d] += w * p
        if den == 0:
            return {}
        return {d: round(num[d] / den, 3) for d in INTEREST_DIMS}

    def current(self, user_id: str, now: float | None = None) -> dict:
        now = now or time.time()
        scopes = self._scope_weights(user_id, now)
        lam = self.feed["scope_lambda"]
        emb_vecs, emb_lams = [], []
        for s, l in (("short", lam["short"]), ("medium", lam["medium"]), ("long", lam["long"])):
            v = self._embedding_interest(scopes[s])
            if v is not None:
                emb_vecs.append(v * l)
        embedding_interest = None
        if emb_vecs:
            e = np.sum(emb_vecs, axis=0)
            n = np.linalg.norm(e)
            embedding_interest = e / n if n > 0 else None
        # metadata: blend scope vectors
        meta = {}
        for s, l in (("short", lam["short"]), ("medium", lam["medium"]), ("long", lam["long"])):
            mv = self._metadata_interest(scopes[s])
            for d, val in mv.items():
                meta[d] = meta.get(d, 0.0) + l * val
        return {
            "embedding": embedding_interest,
            "metadata": meta,
            "scopes": {s: self._metadata_interest(scopes[s]) for s in ("short", "medium", "long")},
            "has_history": bool(sum(len(v) for v in scopes.values())),
        }

    def state_summary(self, user_id: str) -> dict:
        cur = self.current(user_id)
        return {"short_term": cur["scopes"]["short"], "medium_term": cur["scopes"]["medium"],
                "long_term": cur["scopes"]["long"], "has_history": cur["has_history"]}


def _feed_cfg(cfg: Config) -> dict:
    from ..utils.config import _load_yaml
    return _load_yaml("config.yaml").get("feed", {})
