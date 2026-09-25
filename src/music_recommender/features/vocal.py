"""Vocal features (spec sections 13-14).

Reliable vocal *separation* needs a dedicated source-separation model we do not
ship here, so this is a low-confidence HEURISTIC:
  - voice-band energy ratio (200-4000 Hz harmonic content)
  - periodicity of the harmonic component (autocorrelation strength)
  - pitch statistics restricted to the vocal band

`harm` and `f0` are normally injected by the shared extraction pass: hpss and pyin
were previously recomputed here (and in melody/instruments/structure), which made
pitch analysis the majority of per-track cost. Both are optional so this module
still works standalone.

`vocal_gender` is intentionally left "unknown": we do NOT fabricate male/female
(spec section 13/23/81). These fields are marked estimated with low confidence and
are used for soft ranking / category filters, never as ground truth.
"""
from __future__ import annotations

import numpy as np

VOCAL_FMIN, VOCAL_FMAX = 80.0, 1100.0  # ~E2..C6


def vocal_features(y: np.ndarray, sr: int, max_seconds: float = 20.0,
                   harm: np.ndarray | None = None, f0: np.ndarray | None = None) -> dict:
    import librosa
    out = {"vocal_gender": "unknown", "estimated": True, "vocal_confidence": 0.3}

    if harm is None:
        harm, _ = librosa.effects.hpss(y, margin=(3.0, 5.0))
    win_len = int(min(len(harm), max_seconds * sr))
    seg = harm[:win_len]

    S = np.abs(librosa.stft(seg, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    vb = (freqs >= 200) & (freqs <= 4000)
    voice_band_ratio = float(S[vb].sum() / (S.sum() + 1e-9))
    rms_h = float(np.sqrt(np.mean(seg ** 2)))
    rms_full = float(np.sqrt(np.mean(y[:win_len] ** 2))) + 1e-9
    out["vocal_ratio"] = float(np.clip(rms_h / rms_full * voice_band_ratio * 2.0, 0.0, 1.0))
    out["vocal_intensity"] = float(min(1.0, rms_h / (rms_full + 1e-9)))

    # periodicity of harmonic component as a crude "voiced-ness" proxy
    mono = seg[: min(len(seg), sr * 3)]
    if len(mono) > 1024:
        corr = np.correlate(mono - mono.mean(), mono - mono.mean(), mode="full")
        corr = corr[len(corr) // 2:]
        denom = corr[0] + 1e-9
        lag_range = corr[int(sr / 1000): int(sr / 80)] if sr > 8000 else corr[:50]
        periodicity = float(np.max(lag_range) / denom) if len(lag_range) else 0.0
    else:
        periodicity = 0.0

    has_vocal = bool(out["vocal_ratio"] > 0.25 and periodicity > 0.2)
    out["has_vocal"] = int(has_vocal)
    out["vocal_periodicity"] = periodicity

    # pitch in vocal band only when we think there is a voice
    if has_vocal:
        if f0 is None:
            f0, _, _ = librosa.pyin(seg, sr=sr, fmin=VOCAL_FMIN, fmax=VOCAL_FMAX,
                                    frame_length=2048, hop_length=256)
        # restrict the shared track-level f0 to the vocal band
        band = np.where((f0 >= VOCAL_FMIN) & (f0 <= VOCAL_FMAX), f0, np.nan)
        midi = librosa.hz_to_midi(band)
        vm = np.asarray(midi[~np.isnan(midi)], float)
        if vm.size >= 3:
            out["vocal_pitch_mean"] = float(vm.mean())
            out["vocal_pitch_range"] = float(vm.max() - vm.min())
            out["vocal_pitch_variance"] = float(vm.var())
        else:
            out["vocal_pitch_mean"] = out["vocal_pitch_range"] = out["vocal_pitch_variance"] = None
    else:
        out["vocal_pitch_mean"] = out["vocal_pitch_range"] = out["vocal_pitch_variance"] = None
    return out
