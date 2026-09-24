"""Fixed-category recommendation as a query region in Music Space (spec 30-32, 63-64).

A category is NOT a hard tag; it is:
  hard_filter (must-satisfy boolean conditions)
  + soft ranking over Music Space target dimensions (percentile closeness).

Only dimensions we can actually derive locally are used. Query terms whose
feature we do not have (e.g. external genre/arousal) are reported as *ignored*
rather than fabricated (spec principle 1, section 33).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .space import MusicSpace

# query field -> (sqlite column, higher-is-target?)
DIM_MAP = {
    "energy": "rms_mean",
    "loudness": "rms_mean",
    "brightness": "spectral_centroid_mean",
    "timbre_brightness": "spectral_centroid_mean",
    "rhythm_intensity": "onset_density",
    "danceability": "danceability",
    "tempo": "bpm",
    "bpm": "bpm",
    "vocal_presence": "vocal_ratio",
    "vocal_ratio": "vocal_ratio",
    "vocal_pitch": "vocal_pitch_mean",
    "vocal_pitch_mean": "vocal_pitch_mean",
    "vocal_range": "vocal_pitch_range",
    "vocal_pitch_range": "vocal_pitch_range",
    "vocal_intensity": "vocal_intensity",
    "pitch_high": "high_pitch_ratio",
    "bass_energy": "bass_energy_ratio",
    "drum_energy": "drum_energy_ratio",
}

# hard_filter field -> predicate on the raw row
HARD_FILTERS = {
    "has_vocal": lambda row: bool(row.get("has_vocal")),
    "instrumental": lambda row: not bool(row.get("has_vocal")),
    "bpm_min": None,   # handled numerically below
    "bpm_max": None,
}


@dataclass
class CategoryResult:
    track_id: str
    score: float
    matched_dims: dict[str, float] = field(default_factory=dict)


class CategoryEngine:
    def __init__(self, space: MusicSpace):
        self.space = space

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
            if k == "has_vocal" and v and not row.get("has_vocal"):
                return False
            if k == "bpm_min" and (row.get("bpm") or 0) < v:
                return False
            if k == "bpm_max" and (row.get("bpm") or 999) > v:
                return False
        return True

    def recommend(self, category: dict, limit: int = 30) -> tuple[list[CategoryResult], dict]:
        spec = self.resolve(category)
        available = spec["available"]
        if not available:
            return [], {"ignored": spec["ignored"], "usable": 0}
        results = []
        for tid in self.space.rows:
            if not self._passes_hard(tid, spec["hard"]):
                continue
            dists, matched = [], {}
            for col, target in available.items():
                p = self._percentile(tid, col)
                if p is None:
                    continue
                dists.append(abs(p - target))
                matched[col] = round(p, 3)
            if not dists:
                continue
            score = round(1.0 - (sum(dists) / len(dists)), 4)
            results.append(CategoryResult(tid, score, matched))
        results.sort(key=lambda r: r.score, reverse=True)
        meta = {"ignored": spec["ignored"], "usable": len(available), "used_dims": list(available.keys())}
        return results[:limit], meta
