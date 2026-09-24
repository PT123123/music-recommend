"""Time-structure features (spec sections 16-19): segmentation, per-segment
stats, chorus detection and a normalized energy curve.

Segment typing (verse/chorus/bridge) is genuinely ambiguous from audio alone, so
labels are energy-based HEURISTICS and marked estimated. The chorus is taken as the
loudest recurring section (spec section 17 rationale).
"""
from __future__ import annotations

import numpy as np


def _band_energy(S, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return S[m].sum(axis=0) if m.any() else np.zeros(S.shape[1])


def structure_features(y: np.ndarray, sr: int, n_bins: int = 64, max_segments: int = 8) -> dict:
    import librosa
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    S = np.abs(librosa.stft(y, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bass = _band_energy(S, freqs, 20, 250)
    high = _band_energy(S, freqs, 2000, 8000)
    harm, perc = librosa.effects.hpss(y, margin=(3.0, 5.0))
    perc_rms = librosa.feature.rms(y=perc, frame_length=2048, hop_length=hop)[0]
    harm_rms = librosa.feature.rms(y=harm, frame_length=2048, hop_length=hop)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]

    # ---- energy curve (n_bins, normalized) ----
    if len(rms):
        edges = np.linspace(0, len(rms), n_bins + 1).astype(int)
        curve = [float(rms[a:b].mean()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    else:
        curve = [0.0] * n_bins
    cmax = max(curve) or 1.0
    energy_curve = [round(c / cmax, 4) for c in curve]

    # ---- segmentation via novelty on contrast+energy ----
    n_frames = len(rms)
    bounds = _segment_bounds(contrast, rms, n_frames, max_segments, hop, sr)

    seg_list = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b <= a:
            continue
        seg_list.append({
            "index": i,
            "start": round(a * hop / sr, 2),
            "end": round(b * hop / sr, 2),
            "energy_mean": float(rms[a:b].mean()),
            "vocal_energy": float(harm_rms[a:min(b, len(harm_rms))].mean()) if len(harm_rms) else 0.0,
            "bass_energy": float(bass[a:min(b, len(bass))].mean()) if len(bass) else 0.0,
            "drum_energy": float(perc_rms[a:min(b, len(perc_rms))].mean()) if len(perc_rms) else 0.0,
            "spectral_centroid": float(centroid[a:min(b, len(centroid))].mean()) if len(centroid) else 0.0,
        })

    _label_segments(seg_list, energy_curve)
    chorus_idx = _pick_chorus(seg_list)

    type_seq = ",".join(s["type"] for s in seg_list)
    out = {
        "segment_list": seg_list,
        "segment_type_sequence": type_seq,
        "energy_curve": energy_curve,
        "chorus_repeat_count": sum(1 for s in seg_list if s["type"] == "chorus"),
        "estimated": True,
    }
    if chorus_idx is not None:
        c = seg_list[chorus_idx]
        out["chorus_energy"] = c["energy_mean"]
        out["chorus_start"] = c["start"]
        out["chorus_end"] = c["end"]
    return out


def _segment_bounds(contrast, rms, n_frames, max_segments, hop, sr) -> list[int]:
    import librosa
    if n_frames < 8:
        return [0, n_frames]
    # mean-normalized novelty from spectral contrast + energy
    c = contrast.mean(axis=0)
    if len(c) != n_frames:  # contrast may be one frame shorter
        m = min(len(c), n_frames)
        c = c[:m]; rms_ = rms[:m]
        n_eff = m
    else:
        rms_ = rms
        n_eff = n_frames
    norm_c = (c - c.mean()) / (c.std() + 1e-9)
    norm_r = (rms_ - rms_.mean()) / (rms_.std() + 1e-9)
    novelty = np.abs(np.diff(norm_c + norm_r, prepend=(norm_c + norm_r)[0]))
    try:
        b = librosa.segment.agglomerative(np.asarray([norm_c, norm_r]), k=min(max_segments, max(2, n_eff // 8)))
        bounds = [int(x) for x in b]
    except Exception:
        bounds = list(np.linspace(0, n_eff, min(max_segments, 4)).astype(int))
    bounds = sorted(set([0] + bounds + [n_frames]))
    return [x for x in bounds if 0 <= x <= n_frames]


def _label_segments(seg_list: list[dict], curve: list[float]) -> None:
    if not seg_list:
        return
    energies = [s["energy_mean"] for s in seg_list]
    emax = max(energies) or 1.0
    for i, s in enumerate(seg_list):
        s["type"] = "unknown"
    # boundaries
    seg_list[0]["type"] = "intro"
    seg_list[-1]["type"] = "outro"
    # loudest recurring level -> chorus
    threshold = 0.7 * emax
    loud = [s for s in seg_list if s["energy_mean"] >= threshold and s["type"] == "unknown"]
    for s in loud:
        s["type"] = "chorus"
    # quiet interiors -> verse
    for s in seg_list:
        if s["type"] == "unknown" and s["energy_mean"] < 0.4 * emax:
            s["type"] = "verse"
    # any remaining interior: alternate label by position
    for i, s in enumerate(seg_list):
        if s["type"] == "unknown":
            s["type"] = "chorus" if i % 2 == 1 else "verse"


def _pick_chorus(seg_list: list[dict]) -> int | None:
    idx = [i for i, s in enumerate(seg_list) if s["type"] == "chorus"]
    if not idx:
        if not seg_list:
            return None
        return int(max(range(len(seg_list)), key=lambda i: seg_list[i]["energy_mean"]))
    # prefer the loudest chorus occurrence
    return max(idx, key=lambda i: seg_list[i]["energy_mean"])
