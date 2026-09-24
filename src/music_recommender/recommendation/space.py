"""Music Space: rank-normalize library scalar features into [0,1] and compute
per-group similarity (spec sections 4, 60).

All sub-similarities are normalized to [0,1] before weighting, because raw units
(BPM vs MFCC vs embedding distance) are incomparable.
"""
from __future__ import annotations

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
    "structure": [],   # handled specially via energy_curve + segment_type_sequence (Phase 3)
}

# Groups whose similarity also uses an ordered sequence column.
SEQUENCE_COLS: dict[str, str] = {
    "harmony": "relative_chord_sequence",   # key-invariant progression
    "melody": "relative_pitch_sequence",
    "structure": "segment_type_sequence",
}

# Human-readable reason labels per group (spec section 59: reasons must be real).
GROUP_REASON = {
    "energy": "响度/动态能量接近",
    "timbre": "频谱音色接近",
    "rhythm": "节奏/BPM 接近",
    "harmony": "和弦进行相似",
    "melody": "音高/旋律走向接近",
    "vocal": "人声强度/音域接近",
    "instrumentation": "配器能量分布接近",
}


class MusicSpace:
    def __init__(self, conn: sqlite3.Connection):
        self.rows = {r["track_id"]: dict(r) for r in repository.all_tracks(conn)}
        self._build_percentiles()

    def _build_percentiles(self) -> None:
        cols = sorted({c for group in FEATURE_GROUPS.values() for c in group})
        series = defaultdict(list)
        for row in self.rows.values():
            for c in cols:
                v = row.get(c)
                if v is not None:
                    series[c].append(float(v))
        self.sorted_vals = {c: np.sort(np.asarray(vals, float)) for c, vals in series.items() if len(vals)}

    def percentile(self, track_id: str, col: str) -> float | None:
        arr = self.sorted_vals.get(col)
        if arr is None or track_id not in self.rows:
            return None
        v = self.rows[track_id].get(col)
        if v is None:
            return None
        return float(np.searchsorted(arr, float(v), side="right") / len(arr))

    def _curve(self, track_id: str) -> list[float]:
        raw = self.rows.get(track_id, {}).get("energy_curve")
        if not raw:
            return []
        try:
            import json
            return [float(x) for x in json.loads(raw)]
        except Exception:
            return []

    def _structure_similarity(self, a_id: str, b_id: str) -> float | None:
        ca, cb = self._curve(a_id), self._curve(b_id)
        seq_sim = None
        sa = self._seq(self.rows[a_id], "segment_type_sequence")
        sb = self._seq(self.rows[b_id], "segment_type_sequence")
        if len(sa) >= 2 and len(sb) >= 2:
            seq_sim = combined_sequence_similarity(sa, sb)
        if len(ca) == len(cb) and len(ca) > 1 and any(ca) and any(cb):
            va, vb = np.asarray(ca), np.asarray(cb)
            corr = float(np.corrcoef(va, vb)[0, 1]) if va.std() > 0 and vb.std() > 0 else 0.0
            curve_sim = float(np.clip((corr + 1) / 2, 0.0, 1.0))
            if seq_sim is None:
                return curve_sim
            return float(np.clip(0.6 * curve_sim + 0.4 * seq_sim, 0.0, 1.0))
        return seq_sim

    @staticmethod
    def _seq(row: dict, col: str) -> list:
        raw = row.get(col)
        if not raw:
            return []
        if col == "relative_pitch_sequence":
            try:
                import json
                return list(json.loads(raw))
            except Exception:
                return [int(x) for x in str(raw).split(",") if x.strip()]
        return [x for x in str(raw).split(",") if x]

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
        diffs = []
        for c in cols:
            pa, pb = self.percentile(a_id, c), self.percentile(b_id, c)
            if pa is not None and pb is not None:
                diffs.append(abs(pa - pb))
        scalar_sim = float(np.clip(1.0 - (sum(diffs) / len(diffs)), 0.0, 1.0)) if diffs else None

        seq_col = SEQUENCE_COLS.get(group)
        if seq_col:
            sa, sb = self._seq(self.rows[a_id], seq_col), self._seq(self.rows[b_id], seq_col)
            if len(sa) >= 2 and len(sb) >= 2:
                seq_sim = combined_sequence_similarity(sa, sb)
                if scalar_sim is None:
                    return seq_sim
                return float(np.clip(0.5 * scalar_sim + 0.5 * seq_sim, 0.0, 1.0))
        return scalar_sim

