"""Music Space: rank-normalize library scalar features into [0,1] and compute
per-group similarity (spec sections 4, 60).

All sub-similarities are normalized to [0,1] before weighting, because raw units
(BPM vs MFCC vs embedding distance) are incomparable.
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict

import numpy as np

from ..database import repository
from .sequence_sim import combined_sequence_similarity

# Group -> scalar columns used for similarity. Phase 1 + Phase 2 groups.
FEATURE_GROUPS: dict[str, list[str]] = {
    "energy": ["rms_mean", "rms_std", "dynamic_range"],
    "timbre": ["spectral_centroid_mean", "spectral_bandwidth_mean", "spectral_rolloff_mean",
               "spectral_flatness_mean", "spectral_contrast_mean", "zcr_mean"],
    "rhythm": ["bpm", "beat_consistency", "onset_density", "danceability", "tempo_variance"],
    "harmony": ["chord_change_rate", "harmonic_rhythm", "dissonance_mean"],
    "melody": ["pitch_mean", "pitch_range", "pitch_variance", "high_pitch_ratio", "low_pitch_ratio"],
    "vocal": ["vocal_ratio", "vocal_intensity", "vocal_pitch_mean", "vocal_pitch_range"],
    "instrumentation": ["bass_energy_ratio", "drum_energy_ratio"],
    # handled specially: full-track contour + repeat-group pattern + section duration
    # profile. NOT `segment_type_sequence` -- those labels are uncapped heuristics until the
    # plan's stage D model lands, and letting a "chorus,verse,chorus" string dominate the
    # structure score would dress up a guess as a similarity (plan 5.1-5.2).
    "structure": [],
}

# Groups whose similarity also uses an ordered sequence column.
SEQUENCE_COLS: dict[str, str] = {
    "harmony": "relative_chord_sequence",   # key-invariant progression
    "melody": "relative_pitch_sequence",
    "structure": "group_sequence",          # A/B/C repeat pattern, canonical by first appearance
}

# Columns the category / discovery path needs but pair scoring does not: lexicon
# dimensions the pair groups never mention, plus the tag-derived text columns hard
# filters may match on. Derived from the lexicon so adding a dimension there cannot
# leave the space silently unable to read it.
from .lexicon import BY_COL as _LEXICON_COLS  # noqa: E402

_PAIR_COLUMNS = {c for cols in FEATURE_GROUPS.values() for c in cols}
EXTRA_SPACE_COLUMNS: frozenset[str] = frozenset(
    (set(_LEXICON_COLS) - _PAIR_COLUMNS) | {"language", "genre"}
)

# Everything one library read of the tracks table covers: the pair-scoring columns plus
# the wider set the category path reads. `scripts/bench_portability.py` exports only the
# group/sequence columns, so this list is a superset of the on-device payload.
# duration_ratios is a JSON list, not a lexicon column, so it is not implied by the
# group/sequence sets above -- without it the structure profile would always be empty.
SPACE_COLUMNS: tuple[str, ...] = tuple(sorted(
    {"track_id", "has_vocal", "energy_curve", "duration_ratios"}
    | {c for cols in FEATURE_GROUPS.values() for c in cols}
    | set(SEQUENCE_COLS.values())
    | EXTRA_SPACE_COLUMNS
))

# Human-readable reason labels per group (spec section 59: reasons must be real).
GROUP_REASON = {
    "energy": "响度/动态能量接近",
    "timbre": "频谱音色接近",
    "rhythm": "节奏/BPM 接近",
    "harmony": "和弦进行相似",
    "melody": "音高/旋律走向接近",
    "vocal": "人声强度/音域接近",
    "instrumentation": "配器能量分布接近",
    # names what _structure_similarity actually compared; "段落结构相同" would claim
    # agreement about verse/chorus labels that the analyser deliberately refuses to give.
    "structure": "能量轮廓/重复方式接近",
}

# Resampling target for the section-duration profile: fixed length so two songs with
# different segment counts stay comparable (a 4-part and a 9-part track cannot be
# correlated index-by-index).
_PROFILE_POINTS = 8


def _duration_profile(ratios: list[float]) -> np.ndarray | None:
    """Cumulative fraction of the track reached after each k-th segment, resampled."""
    vals = [float(r) for r in ratios or [] if isinstance(r, (int, float)) and float(r) > 0]
    if len(vals) < 2:
        return None
    total = sum(vals)
    if total <= 0:
        return None
    cum = np.cumsum(vals) / total
    x = (np.arange(len(cum)) + 1.0) / len(cum)
    return np.interp((np.arange(_PROFILE_POINTS) + 1.0) / _PROFILE_POINTS, x, cum)


class MusicSpace:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        # Only the scoring columns: SELECT * drags embeddings, segment dumps and every
        # other scalar along, and this object is rebuilt whenever the library changes.
        self.rows = {r["track_id"]: dict(r) for r in repository.rows_with_columns(conn, SPACE_COLUMNS)}
        self._build_percentiles()
        self._seqs: dict[str, dict[str, list]] = {}
        self._curves: dict[str, list[float]] = {}
        self._profiles: dict[str, np.ndarray | None] = {}
        self._vectors: dict[str, np.ndarray] | None = None
        # discovered categories are derived from this object's percentiles, so they
        # expire together with it (one instance per library version)
        self.discovery_cache: tuple[list[dict], dict] | None = None

    def _percentile_columns(self) -> list[str]:
        """Every column we can rank on: pair-similarity groups plus the lexicon."""
        cols = {c for group in FEATURE_GROUPS.values() for c in group} | set(_LEXICON_COLS)
        return sorted(c for c in cols if any(
            r.get(c) is not None and isinstance(r.get(c), (int, float)) for r in self.rows.values()
        ))

    def _build_percentiles(self) -> None:
        cols = self._percentile_columns()
        series = defaultdict(list)
        for row in self.rows.values():
            for c in cols:
                v = row.get(c)
                if isinstance(v, (int, float)):
                    series[c].append(float(v))
        self.sorted_vals = {c: np.sort(np.asarray(vals, float)) for c, vals in series.items() if len(vals)}
        # One track's percentile on one column is asked for by every pair it appears in,
        # so resolve the whole table once instead of re-running searchsorted per pair.
        self.pct: dict[str, dict[str, float]] = {}
        for tid, row in self.rows.items():
            d = self.pct[tid] = {}
            for c, arr in self.sorted_vals.items():
                v = row.get(c)
                if isinstance(v, (int, float)):
                    d[c] = float(np.searchsorted(arr, float(v), side="right") / len(arr))

    def vectors(self) -> dict[str, np.ndarray]:
        """Stored PCA embeddings, loaded once per space (category de-duplication only)."""
        if self._vectors is None:
            self._vectors = {tid: vec for _, tid, vec in repository.ordered_embeddings(self.conn)}
        return self._vectors

    def percentile(self, track_id: str, col: str) -> float | None:
        return self.pct.get(track_id, {}).get(col)

    def _curve(self, track_id: str) -> list[float]:
        cached = self._curves.get(track_id)
        if cached is not None:
            return cached
        raw = self.rows.get(track_id, {}).get("energy_curve")
        curve: list[float] = []
        if raw:
            try:
                curve = [float(x) for x in json.loads(raw)]
            except Exception:
                curve = []
        self._curves[track_id] = curve
        return curve

    def _ratios(self, track_id: str) -> list[float]:
        raw = self.rows.get(track_id, {}).get("duration_ratios")
        if not raw:
            return []
        try:
            return [float(x) for x in json.loads(raw)]
        except Exception:
            return []

    def _profile(self, track_id: str) -> np.ndarray | None:
        """Duration profile, parsed once per track -- pairs are quadratic."""
        if track_id not in self._profiles:
            self._profiles[track_id] = _duration_profile(self._ratios(track_id))
        return self._profiles[track_id]

    def _structure_similarity(self, a_id: str, b_id: str) -> float | None:
        """Blend three independent structural signals; skip whichever is missing.

        Each part is explainable on its own, which is what the player's reason line needs:
        "能量轮廓相似" and "重复方式接近" are claims we can back, "副歌结构相同" is not.
        """
        parts: list[tuple[float, float]] = []      # (weight, similarity)
        ca, cb = self._curve(a_id), self._curve(b_id)
        if len(ca) == len(cb) and len(ca) > 1 and any(ca) and any(cb):
            va, vb = np.asarray(ca), np.asarray(cb)
            if va.std() > 0 and vb.std() > 0:
                parts.append((0.45, float(np.clip((float(np.corrcoef(va, vb)[0, 1]) + 1) / 2, 0, 1))))

        ga, gb = self._seq(self.rows[a_id], "group_sequence", a_id), \
                 self._seq(self.rows[b_id], "group_sequence", b_id)
        if len(ga) >= 2 and len(gb) >= 2:
            parts.append((0.30, combined_sequence_similarity(ga, gb)))

        pa, pb = self._profile(a_id), self._profile(b_id)
        if pa is not None and pb is not None:
            va, vb = np.asarray(pa), np.asarray(pb)
            if va.std() > 0 and vb.std() > 0:
                parts.append((0.25, float(np.clip((float(np.corrcoef(va, vb)[0, 1]) + 1) / 2, 0, 1))))

        if not parts:
            return None
        wsum = sum(w for w, _ in parts)
        return float(np.clip(sum(w * s for w, s in parts) / wsum, 0.0, 1.0))

    def _seq(self, row: dict, col: str, track_id: str = "") -> list:
        """Parse a stored sequence once per track; scoring re-reads it for every pair."""
        per_track = self._seqs.setdefault(track_id, {}) if track_id else None
        if per_track is not None and col in per_track:
            return per_track[col]
        raw = row.get(col)
        if not raw:
            out: list = []
        elif col == "relative_pitch_sequence":
            try:
                out = list(json.loads(raw))
            except Exception:
                out = [int(x) for x in str(raw).split(",") if x.strip()]
        else:
            out = [x for x in str(raw).split(",") if x]
        if per_track is not None:
            per_track[col] = out
        return out

    def group_similarity(self, a_id: str, b_id: str, group: str) -> float | None:
        cols = FEATURE_GROUPS.get(group)
        if group == "structure":
            return self._structure_similarity(a_id, b_id)
        if not cols:
            return None
        # Vocal similarity is only meaningful when both tracks have vocals.
        if group == "vocal":
            if not (self.rows[a_id].get("has_vocal") and self.rows[b_id].get("has_vocal")):
                return None
        pa, pb = self.pct.get(a_id, {}), self.pct.get(b_id, {})
        diffs = []
        for c in cols:
            v1, v2 = pa.get(c), pb.get(c)
            if v1 is not None and v2 is not None:
                diffs.append(abs(v1 - v2))
        scalar_sim = min(1.0, max(0.0, 1.0 - sum(diffs) / len(diffs))) if diffs else None

        seq_col = SEQUENCE_COLS.get(group)
        if seq_col:
            sa = self._seq(self.rows[a_id], seq_col, a_id)
            sb = self._seq(self.rows[b_id], seq_col, b_id)
            if len(sa) >= 2 and len(sb) >= 2:
                seq_sim = combined_sequence_similarity(sa, sb)
                if scalar_sim is None:
                    return seq_sim
                return min(1.0, max(0.0, 0.5 * scalar_sim + 0.5 * seq_sim))
        return scalar_sim


_CACHE: dict[tuple, MusicSpace] = {}


def get_music_space(conn: sqlite3.Connection) -> MusicSpace:
    """Process-wide Music Space, rebuilt only when the library's feature data changes.

    The HTTP server constructs a Recommender per request; without this the percentile
    table and every parsed sequence are thrown away after one query. Keyed by database
    file as well as content version, so two libraries never share a cached space.
    """
    file = conn.execute("PRAGMA database_list").fetchone()[2]
    key = (file, repository.library_version(conn))
    space = _CACHE.get(key)
    if space is None:
        _CACHE.clear()  # one library at a time; the previous one is unreachable anyway
        space = MusicSpace(conn)
        _CACHE[key] = space
    return space

