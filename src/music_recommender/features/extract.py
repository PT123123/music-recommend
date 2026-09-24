"""Aggregate per-track MIR feature extraction into a structured representation.

Produces scalar stats (stored in SQLite), raw arrays (MFCC/chroma), and a
flat statistical vector used by the PCA embedding backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import acoustic, harmony, melody, rhythm, structure
from . import instruments as instr
from . import vocal as vocal_mod


@dataclass
class TrackFeatures:
    scalars: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)   # non-column data (chord arrays, confidences)
    estimated_fields: list[str] = field(default_factory=list)
    mfcc_means: Optional[np.ndarray] = None
    mfcc_stds: Optional[np.ndarray] = None
    chroma_mean: Optional[np.ndarray] = None
    stat_vector: Optional[np.ndarray] = None


def _analysis_window(y: np.ndarray, sr: int, max_seconds: float) -> tuple[np.ndarray, float, float]:
    """Return a bounded, representative slice of a long track.

    Chooses the densest-energy window (so the chorus is included) instead of the
    intro. Returns (y_window, window_start_seconds, window_end_seconds).
    """
    if not max_seconds or max_seconds <= 0:
        return y, 0.0, len(y) / sr
    win = int(max_seconds * sr)
    if len(y) <= win:
        return y, 0.0, len(y) / sr
    frame = max(1, int(0.5 * sr))
    n_frames = len(y) // frame
    blocks = (y[: n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1)
    frames_per_win = max(1, win // frame)
    kernel = np.ones(frames_per_win, dtype="float32")
    energy = np.convolve(blocks, kernel, mode="valid")
    start_frame = int(np.argmax(energy)) if energy.size else 0
    a, b = start_frame * frame, start_frame * frame + win
    return y[a:b], a / sr, min(b, len(y)) / sr


def extract(wav_path: str, sr: int, duration: float) -> TrackFeatures:
    import librosa

    from ..utils.config import get_config

    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    max_sec = float(get_config().audio.get("max_analysis_seconds", 0) or 0)
    y, win_start, win_end = _analysis_window(y, sr, max_sec)

    feats = TrackFeatures()
    feats.scalars["duration"] = float(duration)
    feats.extras["analysis_window"] = [win_start, win_end]
    feats.scalars.update(acoustic.energy_features(y))
    feats.scalars.update(acoustic.spectral_features(y))
    feats.scalars.update(rhythm.rhythm_features(y, sr))

    feats.mfcc_means, feats.mfcc_stds = acoustic.mfcc_features(y)
    feats.chroma_mean = acoustic.chroma_features(y)
    key, mode = acoustic.key_mode(y)
    feats.scalars["key"] = key
    feats.scalars["mode"] = mode

    # --- Phase 2 features (estimated, flagged) ---
    hm = harmony.estimate_chords(y, sr)
    rel = harmony.relative_sequence(hm.get("chord_roots", []), hm.get("chord_qualities", []), key, mode)
    feats.scalars["chord_sequence"] = ",".join(hm.get("chord_sequence", []))
    feats.scalars["relative_chord_sequence"] = ",".join(rel)
    feats.scalars["chord_change_rate"] = hm.get("chord_change_rate")
    feats.scalars["harmonic_rhythm"] = hm.get("harmonic_rhythm")
    feats.scalars["dissonance_mean"] = hm.get("dissonance_mean")
    feats.extras["chord_roots"] = hm.get("chord_roots", [])
    feats.extras["chord_qualities"] = hm.get("chord_qualities", [])
    feats.extras["chord_confidences"] = hm.get("chord_confidences", [])
    feats.extras["chord_sequence_list"] = hm.get("chord_sequence", [])
    feats.extras["relative_sequence_list"] = rel

    mel = melody.melody_features(y, sr)
    for k in ("pitch_min", "pitch_max", "pitch_mean", "pitch_range", "pitch_variance",
              "low_pitch_ratio", "mid_pitch_ratio", "high_pitch_ratio", "melody_contour_type"):
        feats.scalars[k] = mel.get(k)
    feats.scalars["interval_histogram"] = mel.get("interval_histogram", {})
    feats.scalars["relative_pitch_sequence"] = mel.get("relative_pitch_sequence", [])
    feats.scalars["interval_sequence"] = mel.get("interval_sequence", [])
    feats.extras["voiced_ratio"] = mel.get("voiced_ratio")
    feats.extras["melody_density"] = mel.get("melody_density")

    ins = instr.instrument_features(y, sr)
    feats.scalars["bass_energy_ratio"] = ins.get("bass_energy_ratio")
    feats.scalars["drum_energy_ratio"] = ins.get("drum_energy_ratio")
    feats.extras["low_energy_ratio"] = ins.get("low_energy_ratio")
    feats.extras["mid_energy_ratio"] = ins.get("mid_energy_ratio")
    feats.extras["high_energy_ratio"] = ins.get("high_energy_ratio")

    vc = vocal_mod.vocal_features(y, sr)
    feats.scalars["has_vocal"] = vc.get("has_vocal")
    feats.scalars["vocal_ratio"] = vc.get("vocal_ratio")
    feats.scalars["vocal_gender"] = vc.get("vocal_gender")
    feats.scalars["vocal_pitch_mean"] = vc.get("vocal_pitch_mean")
    feats.scalars["vocal_pitch_range"] = vc.get("vocal_pitch_range")
    feats.scalars["vocal_pitch_variance"] = vc.get("vocal_pitch_variance")
    feats.scalars["vocal_intensity"] = vc.get("vocal_intensity")
    feats.extras["vocal_confidence"] = vc.get("vocal_confidence")
    feats.extras["vocal_periodicity"] = vc.get("vocal_periodicity")

    # --- Phase 3 structure features ---
    st = structure.structure_features(y, sr)
    feats.scalars["segment_type_sequence"] = st.get("segment_type_sequence")
    feats.scalars["chorus_repeat_count"] = st.get("chorus_repeat_count")
    feats.scalars["chorus_energy"] = st.get("chorus_energy")
    feats.extras["segment_list"] = st.get("segment_list", [])
    feats.extras["energy_curve"] = st.get("energy_curve", [])
    # chorus-restricted chord progression (spec section 9: chorus_chord_sequence)
    if st.get("chorus_start") is not None and len(y) > sr:
        cs = int(st["chorus_start"] * sr)
        ce = int(min(st.get("chorus_end", duration) * sr, len(y)))
        if ce - cs > int(0.5 * sr):
            hm_c = harmony.estimate_chords(y[cs:ce], sr)
            rel_c = harmony.relative_sequence(hm_c.get("chord_roots", []), hm_c.get("chord_qualities", []), key, mode)
            feats.scalars["chorus_chord_sequence"] = ",".join(hm_c.get("chord_sequence", []))
            feats.extras["chorus_relative_sequence"] = rel_c

    feats.estimated_fields = ["chord", "melody_pitch", "vocal_presence", "instrument_energy", "structure"]
    feats.stat_vector = _build_stat_vector(feats)
    return feats


def _g(s: dict, key: str, default: float = 0.0) -> float:
    v = s.get(key)
    return float(v) if isinstance(v, (int, float)) and v is not None else default


def _build_stat_vector(f: TrackFeatures) -> np.ndarray:
    """Flat, order-stable vector for PCA embedding (log-scaled magnitudes)."""
    s, e = f.scalars, f.extras
    energy = [_g(s, "rms_mean"), _g(s, "rms_std"), _g(s, "rms_p10"), _g(s, "rms_p50"), _g(s, "rms_p90"), _g(s, "dynamic_range")]
    spectral = [np.log1p(_g(s, "spectral_centroid_mean")), np.log1p(_g(s, "spectral_bandwidth_mean")),
                np.log1p(_g(s, "spectral_rolloff_mean")), _g(s, "spectral_flux_mean"),
                _g(s, "spectral_contrast_mean"), _g(s, "spectral_flatness_mean"), _g(s, "zcr_mean")]
    rhythm_v = [np.log1p(_g(s, "bpm")), _g(s, "beat_consistency"), np.log1p(_g(s, "tempo_variance")),
                _g(s, "onset_density"), _g(s, "danceability")]
    harmony_v = [_g(s, "chord_change_rate"), np.log1p(_g(s, "harmonic_rhythm")), _g(s, "dissonance_mean")]
    melody_v = [_g(s, "pitch_mean"), _g(s, "pitch_range"), _g(s, "pitch_variance"),
                _g(s, "low_pitch_ratio"), _g(s, "high_pitch_ratio"),
                _g(e, "voiced_ratio"), _g(e, "melody_density")]
    vocal_v = [_g(s, "vocal_ratio"), _g(s, "vocal_intensity"), _g(s, "has_vocal"), _g(e, "vocal_periodicity")]
    instr_v = [_g(s, "bass_energy_ratio"), _g(s, "drum_energy_ratio"), _g(e, "high_energy_ratio"), _g(e, "low_energy_ratio")]
    curve = np.asarray(e.get("energy_curve") or [0.0], float)
    if curve.size > 1:
        half = curve.size // 2
        structure_v = [float(curve.mean()), float(curve.std()),
                       float(curve[half:].mean() - curve[:half].mean()),   # build-up vs decay
                       float(np.argmax(curve) / curve.size),                # peak position
                       float(_g(s, "chorus_repeat_count"))]
    else:
        structure_v = [0.0, 0.0, 0.0, 0.0, 0.0]
    parts = [np.asarray(energy, float), np.asarray(spectral, float), np.asarray(rhythm_v, float),
             np.asarray(harmony_v, float), np.asarray(melody_v, float),
             np.asarray(vocal_v, float), np.asarray(instr_v, float), np.asarray(structure_v, float),
             np.asarray(f.mfcc_means, float), np.asarray(f.mfcc_stds, float),
             np.asarray(f.chroma_mean, float)]
    return np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

