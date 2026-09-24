"""Acoustic / global MIR features: energy, spectral, MFCC, chroma, key.

Only features that librosa can compute reliably on CPU are produced.
Fields we cannot derive are left None rather than faked (spec section 81 principle 1).
"""
from __future__ import annotations

import numpy as np

SR = 44100


def energy_features(y: np.ndarray) -> dict:
    rms = librosa_rms(y)
    n = max(1, len(rms))
    def _p(q):
        return float(np.percentile(rms, q))
    feats = {
        "rms_mean": float(rms.mean()),
        "rms_std": float(rms.std()),
        "rms_p10": _p(10), "rms_p25": _p(25), "rms_p50": _p(50),
        "rms_p75": _p(75), "rms_p90": _p(90),
        "rms_max": float(rms.max()),
    }
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    feats["crest_factor"] = peak / (feats["rms_mean"] + 1e-9)
    # dynamic range proxy: p95 - p10 of frame RMS (in dB-ish space)
    feats["dynamic_range"] = _p(95) - _p(10)
    return feats


def librosa_rms(y: np.ndarray) -> np.ndarray:
    import librosa
    return librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]


def spectral_features(y: np.ndarray) -> dict:
    import librosa
    hop, fl = 512, 2048
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=hop, n_fft=fl)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR, hop_length=hop, n_fft=fl)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR, hop_length=hop, n_fft=fl)[0]
    flux = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SR, hop_length=hop, n_fft=fl)
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop, n_fft=fl)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=fl, hop_length=hop)[0]
    return {
        "zcr_mean": float(zcr.mean()),
        "spectral_centroid_mean": float(centroid.mean()),
        "spectral_bandwidth_mean": float(bandwidth.mean()),
        "spectral_rolloff_mean": float(rolloff.mean()),
        "spectral_flux_mean": float(flux.mean()),
        "spectral_contrast_mean": float(contrast.mean()),
        "spectral_flatness_mean": float(flatness.mean()),
    }


def mfcc_features(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import librosa
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, hop_length=512, n_fft=2048)
    return mfcc.mean(axis=1), mfcc.std(axis=1)


def chroma_features(y: np.ndarray) -> np.ndarray:
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=512)
    return chroma.mean(axis=1)


def key_mode(y: np.ndarray) -> tuple[str, str]:
    """Heuristic key/mode from average chroma correlated against Krumhansl profiles."""
    import librosa
    chroma = chroma_features(y)
    major = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    best = (-2.0, "C", "major")
    for rot in range(12):
        c_major = np.corrcoef(np.roll(chroma, rot), major)[0, 1]
        c_minor = np.corrcoef(np.roll(chroma, rot), minor)[0, 1]
        if c_major > best[0]:
            best = (c_major, names[rot], "major")
        if c_minor > best[0]:
            best = (c_minor, names[rot], "minor")
    return best[1], best[2]
