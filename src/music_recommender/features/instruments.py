"""Instrument / texture energy features (spec section 15).

Energy-band ratios are computable and reliable-ish. Named instrument "presence"
(piano/guitar/strings/synth) needs a multi-label tagger we do not ship in MVP, so
those are left absent rather than fabricated (principle 1).
"""
from __future__ import annotations

import numpy as np


def instrument_features(y: np.ndarray, sr: int,
                        separated: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    import librosa
    S = np.abs(librosa.stft(y, hop_length=512)) ** 2  # power spectrogram (n_freq, n_frames)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total = S.sum() + 1e-9

    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(S[m].sum() / total)

    if separated is None:
        separated = librosa.effects.hpss(y, margin=(3.0, 5.0))
    harm, perc = separated
    rms_h = float(np.sqrt(np.mean(harm ** 2)))
    rms_p = float(np.sqrt(np.mean(perc ** 2)))
    percussive_ratio = rms_p / (rms_h + rms_p + 1e-9)

    return {
        "bass_energy_ratio": band(20, 250),
        "drum_energy_ratio": percussive_ratio,          # proxy: percussive content
        "low_energy_ratio": band(20, 200),
        "mid_energy_ratio": band(200, 2000),
        "high_energy_ratio": band(2000, 8000),
        "estimated": True,
        # piano/guitar/strings/synth presence intentionally omitted.
    }
