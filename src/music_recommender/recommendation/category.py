"""Category recommendation as a query region in Music Space (spec 30-32, 63-64).

A category is NOT a hard tag; it is:
  hard_filter (must-satisfy boolean conditions)
  + soft ranking over Music Space target dimensions (percentile closeness)
  + support reporting and near-duplicate suppression (small-library adaptation).

Only dimensions we can actually derive locally are used. Query terms whose
feature we do not have (e.g. external genre/arousal) are reported as *ignored*
rather than fabricated (spec principle 1, section 33).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lexicon import DIM_MAP, BY_COL
from .space import MusicSpace

# hard_filter field -> predicate on the raw row. Anything else is either a numeric
# "<column>_min" / "<column>_max" bound or a tag equality "<field>_is"; both are
# handled generically in `_passes_hard` so a new dimension needs no code change.
HARD_FILTERS = {
    "has_vocal": lambda row: bool(row.get("has_vocal")),
    "instrumental": lambda row: not bool(row.get("has_vocal")),
}

# tag-derived filters: value comes from the file's own embedded tag (ADR-14), never
# from audio. A track with no such tag can never match, and `unavailable` counts them.
TAG_FILTERS = {
    "language_is": "language",
    "genre_is": "genre",
}

# Filters whose backing value is a thresholded guess rather than a measurement:
# `has_vocal` is "voice-band energy ratio over the analysed window", so a densely
# mastered pop track can land on the instrumental side. Answering with such a filter
# is legitimate; presenting its result as certain is not, so it is reported.
ESTIMATE_FILTERS = frozenset({"has_vocal", "instrumental"})


@dataclass
class CategoryResult:
    track_id: str
    score: float
    matched_dims: dict[str, float] = field(default_factory=dict)


@dataclass
class CategoryConfig:
    """Everything tunable for the category path, from `config.yaml: categories`."""
    support_max_distance: float = 0.25
    support_min_share: float = 0.05
    support_min_count: int = 5
    dedupe: bool = True
    dedupe_lambda: float = 0.8
    dedupe_max_similarity: float = 0.97

    @classmethod
    def from_settings(cls, section: dict | None) -> "CategoryConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (section or {}).items() if k in known})


class CategoryEngine:
    def __init__(self, space: MusicSpace, ccfg: CategoryConfig | None = None):
        self.space = space
        self.ccfg = ccfg or CategoryConfig()

    def _percentile(self, tid: str, col: str) -> float | None:
        return self.space.percentile(tid, col)

    def resolve(self, category: dict) -> dict:
        """Return {available:{col:target}, ignored:[fields], hard:{...}} for a category."""
        query = category.get("query", {}) or {}
        available, ignored = {}, []
        for f, target in query.items():
            col = DIM_MAP.get(f)
            if col and col in self.space.sorted_vals:
                available[col] = float(target)
            else:
                ignored.append(f)
        return {"available": available, "ignored": ignored, "hard": category.get("hard_filter", {}) or {}}

    def _passes_hard(self, tid: str, hard: dict) -> bool:
        row = self.space.rows.get(tid, {})
        for k, v in hard.items():
            pred = HARD_FILTERS.get(k)
            if pred is not None:
                if v and not pred(row):
                    return False
                continue
            if k in TAG_FILTERS:
                col = TAG_FILTERS[k]
                actual = row.get(col)
                if not actual or str(actual).strip().lower() != str(v).strip().lower():
                    return False
                continue
            if k.endswith("_min") or k.endswith("_max"):
                col, bound = k[:-4], "min" if k.endswith("_min") else "max"
                if col not in BY_COL and col not in self.space.rows.get(tid, {}):
                    continue  # unknown bound: reported in meta, not silently applied
                value = row.get(col)
                if value is None:
                    return False
                value = float(value)
                if bound == "min" and value < float(v):
                    return False
                if bound == "max" and value > float(v):
                    return False
        return True

    def _hard_report(self, hard: dict) -> tuple[list[str], dict[str, int], list[str]]:
        """(unknown conditions, {tag field: tracks with no value}, estimate-backed conditions)."""
        unknown = []
        for k in hard:
            if k in HARD_FILTERS or k in TAG_FILTERS:
                continue
            col = k[:-4] if k.endswith("_min") or k.endswith("_max") else k
            if col not in BY_COL and col not in next(iter(self.space.rows.values()), {}):
                unknown.append(k)
        unavailable = {}
        for k, col in TAG_FILTERS.items():
            if k in hard:
                missing = sum(1 for r in self.space.rows.values() if not r.get(col))
                unavailable[col] = missing
        return unknown, unavailable, sorted(k for k, v in hard.items() if v and k in ESTIMATE_FILTERS)

    def _member_ids(self, category: dict) -> set[str] | None:
        """Discovered categories carry their cluster membership; use it as the pool."""
        members = category.get("member_track_ids")
        return set(members) if members else None

    def _dedupe(self, results: list[CategoryResult], limit: int) -> tuple[list[CategoryResult], int]:
        """Greedy lambda*relevance - (1-lambda)*max-cosine selection.

        On a 93-track library a category's top-30 was mostly the same song family
        repeated; scores say nothing about redundancy, embeddings do. Candidates
        above `dedupe_max_similarity` to an already-picked track are dropped, and
        if that starves the list the remainder is topped up by pure score.

        Stored embeddings are not unit length, so similarity uses L2-normalised rows: a
        raw dot product would make the 0.97 cap mean nothing (it is a magnitude, not an
        angle). One matrix per call + one matrix-vector product per pick keeps this off
        the category path's critical time budget.
        """
        vectors = self.space.vectors()
        if not self.ccfg.dedupe or limit <= 1 or not vectors:
            return results[:limit], 0
        if len(results) <= limit:
            return results, 0
        dim = len(next(iter(vectors.values())))
        rows, idx = [], []
        for i, r in enumerate(results):
            v = vectors.get(r.track_id)
            if v is None:  # no embedding: never suppressed, contributes no redundancy
                rows.append(np.zeros(dim))
            else:
                v = np.asarray(v, float).ravel()
                n = float(np.linalg.norm(v))
                rows.append(v / n if n > 0 else np.zeros(dim))
            idx.append(i)
        M = np.vstack(rows)
        score = np.array([results[i].score for i in idx], float)
        lam, cap = self.ccfg.dedupe_lambda, self.ccfg.dedupe_max_similarity
        alive = np.ones(len(idx), bool)
        rejected = np.zeros(len(idx), bool)
        max_red = np.zeros(len(idx))
        picked: list[CategoryResult] = []
        while len(picked) < limit:
            usable = alive & ~rejected
            if not usable.any():
                break
            obj = np.where(usable, lam * score - (1.0 - lam) * max_red, -np.inf)
            j = int(np.argmax(obj))
            alive[j] = False
            picked.append(results[idx[j]])
            max_red = np.maximum(max_red, M @ M[j])
            rejected |= (max_red > cap) & alive
        suppressed = int(rejected.sum())
        if len(picked) < limit:  # starvation must not silently shorten the answer
            seen = {r.track_id for r in picked}
            for r in results:
                if len(picked) >= limit:
                    break
                if r.track_id not in seen:
                    picked.append(r)
                    seen.add(r.track_id)
        return picked, suppressed

    def recommend(self, category: dict, limit: int = 30) -> tuple[list[CategoryResult], dict]:
        spec = self.resolve(category)
        available = spec["available"]
        members = self._member_ids(category)
        hard = spec["hard"]
        unknown_hard, tag_unavailable, estimated_hard = self._hard_report(hard)
        if not available and not (hard or members):
            # nothing to rank on and nothing to filter: refuse instead of shuffling
            return [], {"ignored": spec["ignored"], "usable": 0, "unknown_filters": unknown_hard,
                        "filter_estimated": estimated_hard}
        scored = []
        for tid in self.space.rows:
            if members is not None and tid not in members:
                continue
            if not self._passes_hard(tid, hard):
                continue
            dists, matched = [], {}
            for col, target in available.items():
                p = self._percentile(tid, col)
                if p is None:
                    continue
                dists.append(abs(p - target))
                matched[col] = round(p, 3)
            if available and not dists:
                continue
            mean_dist = (sum(dists) / len(dists)) if dists else 0.0
            scored.append((round(1.0 - mean_dist, 4), tid, matched, mean_dist))
        scored.sort(key=lambda t: (-t[0], t[1]))

        support = sum(1 for _, _, _, d in scored if d <= self.ccfg.support_max_distance)
        # A dimension can be asked for and still carry no information: vocal percentiles
        # are NULL on instrumentals, so "女声 + 纯音乐" ranks on nothing and every track
        # ties at 1.0. Saying which targets actually scored is what makes that visible.
        scored_cols = {c for _, _, matched, _ in scored for c in matched}
        pool_size = len(members) if members is not None else len(self.space.rows)
        min_support = max(self.ccfg.support_min_count,
                          int(round(self.ccfg.support_min_share * pool_size)))
        results = [CategoryResult(tid, s, m) for s, tid, m, _ in scored]
        picked, suppressed = self._dedupe(results, limit)
        meta = {
            "ignored": spec["ignored"],
            "usable": len(available),
            "used_dims": list(available.keys()),
            "scored_dims": sorted(scored_cols),
            "unscored_dims": sorted(set(available) - scored_cols),
            "unknown_filters": unknown_hard,
            "filters_applied": sorted(k for k, v in hard.items() if v),
            "filter_estimated": estimated_hard,
            "tag_missing": tag_unavailable,
            "support": support,
            "candidate_pool": pool_size,
            "low_support": support < min_support,
            "min_support": min_support,
            "filter_only": not scored_cols,
            "deduped": suppressed > 0,
            "duplicates_suppressed": suppressed,
        }
        return picked, meta
