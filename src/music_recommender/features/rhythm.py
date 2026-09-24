"""Rhythm features: BPM, beat consistency, onset density, tempo variance.

`danceability` is a crude proxy from beat regularity + onset density in MVP;
it is flagged as an estimate, not an external truth (spec section 20 / 81).
"""
from __future__ import annotations

import numpy as np


def rhythm_features(y: np.ndarray, sr: int) -> dict:
    import librosa
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, trim=True)
    bpm = float(np.atleast_1d(tempo)[0])

    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
    intervals = np.diff(beat_times) if len(beat_times) > 1 else np.array([0.0])
    interval_std = float(intervals.std()) if len(intervals) else 0.0
    mean_interval = float(intervals.mean()) if len(intervals) else 0.0
    # beat_consistency: 1 - normalized std of inter-beat intervals
    consistency = float(np.clip(1.0 - (interval_std / (mean_interval + 1e-9)), 0.0, 1.0))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=512)
    duration = len(y) / sr if sr else 1.0
    onset_density = float(len(onsets) / duration) if duration > 0 else 0.0

    tempo_variance = 0.0
    if len(intervals) > 2:
        local_tempo = 60.0 / np.clip(intervals, 1e-3, None)
        tempo_variance = float(local_tempo.std())

    danceability = float(np.clip(consistency * 0.6 + min(onset_density / 10.0, 1.0) * 0.4, 0.0, 1.0))

    return {
        "bpm": bpm,
        "beat_consistency": consistency,
        "tempo_variance": tempo_variance,
        "onset_density": onset_density,
        "danceability": danceability,
    }
