"""Sound-texture analysis: timbre-neutral descriptors for synthetic and unnamed sources.

Everything here is frame-level DSP on the whole track, sharing the same
bucket conventions as `dynamics.grid` (bucket centers, interval in meta) so a
player can switch curves on one timeline. Honesty rules:

  - spectral flatness describes tonal-vs-noisy spectral shape; a high value
    does NOT mean "a white-noise synthesizer was used".
  - spectral bandwidth is power-weighted spread in Hz. It is a frequency-axis
    measure and has nothing to do with stereo width (see spatial.py).
  - per-band flux says where in the spectrum a change happened, not which
    instrument entered.
  - onsets becoming denser in a mix can come from vocals, drums, plucks or
    synths; nothing here may assert a source-level process (granulation,
    time-stretch, chops).
"""
from __future__ import annotations

import numpy as np

DEFAULTS: dict = {
    "frame_length": 2048,
    "hop_length": 512,
    "curve_interval_seconds": 0.5,
    "hf_hz": 4000.0,
    "flux_bands": {"low": (20.0, 250.0), "mid": (250.0, 2000.0), "high": (2000.0, 8000.0)},
}


def _cfg(cfg: dict | None, key: str):
    if cfg:
        v = cfg.get("analysis", {}).get(key)
        if v is not None:
            return v
    return DEFAULTS[key]


def texture_frame_curves(y: np.ndarray, sr: int, cfg: dict | None = None) -> dict:
    """One STFT, the texture frame series. Same hop/interval family as dynamics."""
    import librosa

    fl = int(_cfg(cfg, "frame_length"))
    hop = int(_cfg(cfg, "hop_length"))
    hf_hz = float(_cfg(cfg, "hf_hz"))

    stft = librosa.stft(y, n_fft=fl, hop_length=hop)
    mag = np.abs(stft)
    power = mag ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=fl)

    n = power.shape[1]
    total = power.sum(axis=0) + 1e-12

    # flatness: geometric mean / arithmetic mean per frame (librosa convention).
    # 0 = single dominant partial, 1 = noise-like. [librosa.feature.spectral_flatness]
    flatness = librosa.feature.spectral_flatness(S=mag ** 2)[0]
    flatness = _align(flatness, n)

    # bandwidth: power-weighted spread around the centroid, Hz.
    bandwidth = librosa.feature.spectral_bandwidth(S=power, sr=sr, n_fft=fl, hop_length=hop)[0]
    bandwidth = _align(bandwidth, n)

    hf = freqs >= hf_hz
    hf_ratio = power[hf].sum(axis=0) / total if hf.any() else np.zeros(n)

    # per-band positive flux (relative to that band's own mass), so a loud band is not
    # automatically a "changing" band: this says WHERE in the spectrum things moved.
    band_flux = {}
    for name, (lo, hi) in _cfg(cfg, "flux_bands").items():
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            band_flux[name] = np.zeros(n)
            continue
        band_mag = mag[m]
        diff = np.maximum(0.0, np.diff(band_mag, axis=1, prepend=band_mag[:, :1]))
        band_flux[name] = _align(diff.sum(axis=0) / (band_mag.sum(axis=0) + 1e-9), n)

    return {
        "times": librosa.times_like(flatness, sr=sr, hop_length=hop),
        "flatness": flatness,
        "bandwidth": bandwidth,
        "hf_ratio": hf_ratio,
        "flux": band_flux,
    }


def texture_curves(y: np.ndarray, sr: int, cfg: dict | None = None) -> dict:
    """Bucket-grid texture curves, same time semantics as the dynamics grid."""
    interval = float(_cfg(cfg, "curve_interval_seconds"))
    frames = texture_frame_curves(y, sr, cfg)
    times = frames["times"]
    dur = float(times[-1]) if len(times) else 0.0
    n_buckets = max(1, int(np.ceil(dur / interval))) if dur > 0 else 1
    out = {"time_seconds": [round((i + 0.5) * interval, 3) for i in range(n_buckets)],
           "interval_seconds": interval,
           "meta": {"scope": "full_track", "n_points": n_buckets,
                    "time_semantics": "bucket_center",
                    "hf_hz": float(_cfg(cfg, "hf_hz"))}}

    idx = np.clip((np.asarray(times, float) / interval).astype(int), 0, n_buckets - 1)
    counts = np.bincount(idx, minlength=n_buckets).astype(float)

    def bucket_mean(series: np.ndarray) -> list[float]:
        vals = np.bincount(idx, weights=np.asarray(series, float), minlength=n_buckets)
        vals = np.divide(vals, counts, out=np.zeros_like(vals), where=counts > 0)
        return [round(float(x), 5) for x in vals]

    out["spectral_flatness"] = bucket_mean(frames["flatness"])
    out["spectral_bandwidth"] = [round(float(x), 1) for x in bucket_mean(frames["bandwidth"])]
    out["hf_power_ratio"] = bucket_mean(frames["hf_ratio"])
    for name, series in frames["flux"].items():
        out[f"flux_{name}"] = bucket_mean(series)
    return out


def summarize_texture_by_segment(curves: dict, segments: list[dict],
                                 cfg: dict | None = None) -> list[dict]:
    """Per-segment texture means, so a card can say WHICH part got noisier/brighter."""
    times = np.asarray(curves.get("time_seconds") or [], float)
    if not times.size:
        return []
    rows = []
    for seg in segments:
        s, e = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        if e <= s:
            continue
        m = (times >= s) & (times < e)
        if not m.any():
            continue
        row = {"start": round(s, 2), "end": round(e, 2), "group": seg.get("group", "")}
        for key in ("spectral_flatness", "spectral_bandwidth", "hf_power_ratio",
                    "flux_low", "flux_mid", "flux_high"):
            vals = np.asarray(curves.get(key) or [], float)
            if vals.size == times.size:
                row[key] = round(float(vals[m].mean()), 5)
        rows.append(row)
    return rows


def _align(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, float).ravel()
    if a.size == n:
        return a
    if a.size > n:
        return a[:n]
    return np.pad(a, (0, n - a.size), constant_values=0.0)
