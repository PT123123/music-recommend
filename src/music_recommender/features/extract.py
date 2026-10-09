"""Aggregate per-track MIR feature extraction into a structured representation.

Produces scalar stats (stored in SQLite), raw arrays (MFCC/chroma), and a
flat statistical vector used by the PCA embedding backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import acoustic, dynamics, harmony, melody, rhythm, structure
from . import dynamic_profiles as dprof_mod
from . import description as description_mod
from . import spatial as spatial_mod
from . import texture as texture_mod
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
    """Two analysis scopes on purpose (plan stage B.1).

    Cheap frame-level DSP (RMS, bands, onset, chroma) runs on the COMPLETE track so the
    dynamics curves and section boundaries describe the song, not a slice. The expensive
    features (pyin pitch, chord CQT, vocal/instrument decomposition) stay restricted to the
    configurable loudest window, because their cost dominates and a 45 s representative
    slice is enough for the similarity scalars. `analysis_window` records which slice.
    """
    import librosa

    from ..utils.config import get_config

    cfg = get_config()
    y_full, sr = librosa.load(wav_path, sr=sr, mono=True)
    max_sec = float(cfg.audio.get("max_analysis_seconds", 0) or 0)
    y, win_start, win_end = _analysis_window(y_full, sr, max_sec)
    cfgf = cfg.features
    seq_budget = int(cfgf.get("max_sequence_tokens", 120))

    # ---- full-track scope: dynamics + structure ----
    acfg = {"analysis": cfg.analysis}
    frames = dynamics.frame_curves(y_full, sr, acfg)
    dyn = dynamics.track_dynamics(y_full, sr, acfg, frames=frames)
    st = structure.structure_features(y_full, sr, acfg, frames=frames)

    # ---- P1 扩展模块：纹理 / 空间 / 局部律动 / 轮廓 / 描述 ----
    # 纹理与律动时间线走与 dynamics 相同的桶网格约定；空间只在原声道分支
    # 解码成功时可用（单声道/多声道如实报 unavailable，绝不伪造立体声分析）。
    tex = texture_mod.texture_curves(y_full, sr, acfg)
    rhy = rhythm.rhythm_timeline(y_full, sr, acfg)
    stereo = None
    if bool(cfg.analysis.get("stereo", True)):
        try:
            stereo = librosa.load(wav_path, sr=sr, mono=False)[0]
        except Exception:
            stereo = None
    spa = spatial_mod.spatial_curves(stereo, sr, acfg) if stereo is not None \
        else {"status": "unavailable", "reason": "stereo_decode_failed", "curves": {}, "valid": [], "meta": {}}
    prof = dprof_mod.summarize_dynamic_profiles(dyn, st.get("segment_list", []),
                                                dyn.get("events", []), acfg)
    desc = description_mod.describe(dyn, tex, spa, rhy, prof, st.get("segment_list", []), acfg)

    # hpss and pyin are each computed ONCE here and injected into the modules that
    # need them. They previously ran per-module (pyin x2, hpss x3), which dominated
    # extraction cost. Pitch analysis runs on the harmonic stem, which is also the
    # cleaner melody input.
    separated = librosa.effects.hpss(y, margin=(3.0, 5.0))
    harm = separated[0]
    pitch_sec = float(cfgf.get("max_pitch_seconds", 30))
    f0, _, _ = librosa.pyin(harm[: int(min(len(harm), pitch_sec * sr))], sr=sr,
                            fmin=melody.FMIN, fmax=melody.FMAX, frame_length=2048, hop_length=256)

    feats = TrackFeatures()
    feats.scalars["duration"] = float(duration)
    feats.extras["analysis_window"] = [win_start, win_end]
    # Loudness scalars come from the full track: p10/p90 of a 45 s window cannot see the
    # quiet intro or the outro, which biased `dynamic_range` towards the loudest section.
    feats.scalars.update(acoustic.energy_features(y_full, rms=frames["rms"]))
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
    feats.scalars["chord_sequence"] = ",".join(melody.cap_sequence(hm.get("chord_sequence", []), seq_budget))
    feats.scalars["relative_chord_sequence"] = ",".join(melody.cap_sequence(rel, seq_budget))
    feats.scalars["chord_change_rate"] = hm.get("chord_change_rate")
    feats.scalars["harmonic_rhythm"] = hm.get("harmonic_rhythm")
    feats.scalars["dissonance_mean"] = hm.get("dissonance_mean")
    feats.extras["chord_roots"] = hm.get("chord_roots", [])
    feats.extras["chord_qualities"] = hm.get("chord_qualities", [])
    feats.extras["chord_confidences"] = hm.get("chord_confidences", [])
    feats.extras["chord_sequence_list"] = hm.get("chord_sequence", [])
    feats.extras["relative_sequence_list"] = rel

    mel = melody.melody_features(y, sr, f0=f0)
    for k in ("pitch_min", "pitch_max", "pitch_mean", "pitch_range", "pitch_variance",
              "low_pitch_ratio", "mid_pitch_ratio", "high_pitch_ratio", "melody_contour_type"):
        feats.scalars[k] = mel.get(k)
    feats.scalars["interval_histogram"] = mel.get("interval_histogram", {})
    feats.scalars["relative_pitch_sequence"] = mel.get("relative_pitch_sequence", [])
    feats.scalars["interval_sequence"] = mel.get("interval_sequence", [])
    feats.extras["voiced_ratio"] = mel.get("voiced_ratio")
    feats.extras["melody_density"] = mel.get("melody_density")

    ins = instr.instrument_features(y, sr, separated=separated)
    feats.scalars["bass_energy_ratio"] = ins.get("bass_energy_ratio")
    feats.scalars["drum_energy_ratio"] = ins.get("drum_energy_ratio")
    feats.extras["low_energy_ratio"] = ins.get("low_energy_ratio")
    feats.extras["mid_energy_ratio"] = ins.get("mid_energy_ratio")
    feats.extras["high_energy_ratio"] = ins.get("high_energy_ratio")

    vc = vocal_mod.vocal_features(y, sr, harm=harm, f0=f0)
    feats.scalars["has_vocal"] = vc.get("has_vocal")
    feats.scalars["vocal_ratio"] = vc.get("vocal_ratio")
    feats.scalars["vocal_gender"] = vc.get("vocal_gender")
    feats.scalars["vocal_pitch_mean"] = vc.get("vocal_pitch_mean")
    feats.scalars["vocal_pitch_range"] = vc.get("vocal_pitch_range")
    feats.scalars["vocal_pitch_variance"] = vc.get("vocal_pitch_variance")
    feats.scalars["vocal_intensity"] = vc.get("vocal_intensity")
    feats.extras["vocal_confidence"] = vc.get("vocal_confidence")
    feats.extras["vocal_periodicity"] = vc.get("vocal_periodicity")

    # --- full-track structure + dynamics (plan stage B/C) ---
    feats.scalars["segment_type_sequence"] = st.get("segment_type_sequence")
    feats.scalars["chorus_repeat_count"] = st.get("chorus_repeat_count")
    feats.scalars["chorus_energy"] = st.get("chorus_energy")
    feats.extras["segment_list"] = st.get("segment_list", [])
    feats.extras["energy_curve"] = st.get("energy_curve", [])
    feats.extras["group_sequence"] = st.get("group_sequence", "")
    feats.extras["duration_ratios"] = st.get("duration_ratios", [])
    # dynamics_json 升级为多模块载荷：dynamics 保持旧键（旧客户端读它不受影响），
    # 新模块并排放同层；repository.get_analysis 负责兼容旧的单层格式。
    deep_cfg = cfg.analysis.get("deep", {}) or {}
    deep_enabled = bool(deep_cfg.get("enabled", False))
    from ..models import source_separator, structure_labeler
    feats.extras["dynamics"] = {
        "dynamics": dyn,
        "texture": tex,
        "spatial": spa,
        "rhythm_timeline": rhy,
        "profiles": prof,
        "description": desc,
        "modules": {
            "dynamics": "ready",
            "structure": "ready",
            "texture": "ready",
            "rhythm_timeline": "ready",
            "profiles": "ready",
            "description": "ready",
            "spatial": spa.get("status", "unavailable") if spa.get("status") == "ready"
                       else f"unavailable:{spa.get('reason', 'unknown')}",
            "deep_structure_labels": structure_labeler.AllInOneStructureLabeler().status()["status"]
                                     if deep_enabled else "not_requested",
            "deep_source_separation": source_separator.DemucsSourceSeparator().status()["status"]
                                      if deep_enabled else "not_requested",
        },
    }
    # meta only -- segments/dynamics/energy_curve own dedicated columns, and repeating
    # them here would put a few thousand numbers inside features_json.
    feats.extras["analysis"] = {
        "analysis_version": int(cfg.analysis.get("version", 5)),
        "analysis_scope": "full_track",
        "duration_seconds": round(float(duration), 3),
        "feature_window_seconds": [round(win_start, 3), round(win_end, 3)],
        "alignment": st.get("alignment"),
        "tempo_bpm": st.get("tempo_bpm"),
        "segment_count": len(st.get("segments", [])),
        "group_count": st.get("group_count"),
        "group_sequence": st.get("group_sequence", ""),
        "event_count": len((dyn or {}).get("events", [])),
        "label_source": st.get("label_source"),
        "labels_are_estimates": st.get("labels_are_estimates", True),
    }
    # chorus-restricted chord progression (spec section 9: chorus_chord_sequence).
    # Slice the FULL track: chorus_start/end are now real song timestamps, not window offsets.
    if st.get("chorus_start") is not None and len(y_full) > sr:
        cs = int(st["chorus_start"] * sr)
        ce = int(min(st.get("chorus_end", duration) * sr, len(y_full)))
        if ce - cs > int(0.5 * sr):
            hm_c = harmony.estimate_chords(y_full[cs:ce], sr)
            rel_c = harmony.relative_sequence(hm_c.get("chord_roots", []), hm_c.get("chord_qualities", []), key, mode)
            feats.scalars["chorus_chord_sequence"] = ",".join(melody.cap_sequence(hm_c.get("chord_sequence", []), seq_budget))
            feats.extras["chorus_relative_sequence"] = melody.cap_sequence(rel_c, seq_budget)

    # What each proxy column actually measures, so a client can label it correctly instead of
    # calling `drum_energy_ratio` "drums" (plan stage A.3).
    feats.extras["field_meaning"] = {
        "drum_energy_ratio": "HPSS 打击性成分占比，不是鼓组轨道或鼓声音量",
        "bass_energy_ratio": "20-250 Hz 能量占比，不等于 bass 吉他/合成器",
        "vocal_ratio": "低置信人声存在启发式（voice-band 能量 × 谐波 RMS），非分离结果",
        "vocal_gender": "unknown：本引擎不臆造性别标签",
        "onset_density": "onset 包络积分，编曲密度的代理量",
        "segment_type_sequence": "重复分组启发式得到的功能标签，未校准，unknown 表示证据不足",
        "energy_curve": "全曲响度曲线（按最大值归一），绝对尺度在 dynamics.rms_db",
        "confidence": "模型分数，非概率；未经人工标注校准",
    }
    feats.estimated_fields = ["chord", "melody_pitch", "vocal_presence", "instrument_energy",
                              "structure", "dynamics_events"]
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

