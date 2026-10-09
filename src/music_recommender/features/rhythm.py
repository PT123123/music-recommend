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


def rhythm_timeline(y: np.ndarray, sr: int, cfg: dict | None = None) -> dict:
    """Local groove timeline: windowed tempo, beat candidates, tempo-change intervals.

    Honesty rules (expansion doc section 8):
      - onsets getting denser is NOT acceleration; the tempo curve here comes from
        inter-onset autocorrelation per window, so a constant click at 120 BPM stays
        ~120 BPM even when the surface gets busier.
      - half/double tempo is a real ambiguity (80 vs 160 BPM): candidates are SAVED,
        none is forced to be the truth.
      - meter (4/4 vs 3/4) needs bar-level downbeat knowledge this module does not
        have: meter is reported as "unknown" rather than guessed.
      - rubato is not "unstable tempo"; stability is only claimed for sustained,
        repeated local tempo agreement.
    """
    import librosa

    window_s = 8.0
    hop_s = 4.0
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    duration = len(y) / sr if sr else 1.0

    win_frames = max(1, int(window_s * sr / hop))
    step_frames = max(1, int(hop_s * sr / hop))
    times: list[float] = []
    tempos: list[float] = []
    n = onset_env.size
    for start in range(0, max(1, n - win_frames), step_frames):
        chunk = onset_env[start:start + win_frames]
        if chunk.size < win_frames // 2 or not np.any(chunk > 0):
            continue
        ac = librosa.autocorrelate(chunk, max_size=win_frames // 2)
        ac = np.asarray(ac, float)
        if ac.size < 8 or float(ac[0]) <= 0:
            continue
        ac /= ac[0]
        # tempo candidates from the first strong autocorrelation peak after 0.2 s
        # period (<= 300 BPM): lag in frames -> BPM. The SHORTEST strong period is
        # the tempo; taking the global max would read a 120 BPM click as 60 (the
        # two-period lag correlates just as well on a perfectly periodic signal).
        min_lag = max(2, int(round((60.0 / 300.0) * sr / hop)))
        peak_region = ac[min_lag:]
        if peak_region.size < 2:
            continue
        strong = np.flatnonzero(peak_region >= 0.5 * float(peak_region.max()))
        if strong.size == 0:
            continue
        lag = int(strong[0]) + min_lag
        bpm = 60.0 * sr / (hop * lag)
        times.append(round((start + win_frames / 2) * hop / sr, 2))
        tempos.append(round(float(bpm), 1))

    med = float(np.median(tempos)) if tempos else 0.0
    # tempo candidates: median plus half/double — the 80/160 ambiguity is saved,
    # not resolved silently.
    candidates = sorted({round(med / 2, 1), round(med, 1), round(med * 2, 1)}) if med > 0 else []

    # tempo-change intervals: windows whose tempo deviates >10% from the track median
    # for >= 2 consecutive windows.
    changes: list[dict] = []
    if med > 0 and len(tempos) >= 3:
        run_start = None
        for i, t in enumerate(tempos):
            deviates = abs(t - med) > 0.10 * med
            if deviates and run_start is None:
                run_start = i
            elif not deviates and run_start is not None:
                if i - run_start >= 2:
                    changes.append({
                        "start": times[run_start],
                        "end": times[i - 1],
                        "tempo_bpm": round(float(np.mean(tempos[run_start:i])), 1),
                        "evidence": [f"median:{med:.1f}bpm",
                                     f"window:{np.mean(tempos[run_start:i]):.1f}bpm",
                                     f"deviation:{abs(np.mean(tempos[run_start:i]) - med) / med:.0%}"],
                    })
                run_start = None
        if run_start is not None and len(tempos) - run_start >= 2:
            changes.append({
                "start": times[run_start],
                "end": times[-1],
                "tempo_bpm": round(float(np.mean(tempos[run_start:])), 1),
                "evidence": [f"median:{med:.1f}bpm",
                             f"window:{np.mean(tempos[run_start:]):.1f}bpm"],
            })

    stable = (len(tempos) >= 3 and med > 0
              and float(np.mean(np.abs(np.asarray(tempos) - med))) <= 0.05 * med)

    beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, trim=False)
    beat_frames = np.atleast_1d(beats[1])
    beat_times = [round(float(x), 3) for x in librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)]

    return {
        "status": "ready",
        "median_tempo_bpm": round(med, 1),
        "tempo_candidates_bpm": candidates,
        "tempo_stable": bool(stable),
        "local_tempo": {
            "time_seconds": times,
            "values": tempos,
            "window_seconds": window_s,
            "hop_seconds": hop_s,
        },
        "tempo_changes": changes,
        "beat_candidates": beat_times,
        "meter": "unknown",
        "meta": {
            "scope": "full_track",
            "notes": [
                "起音变密不等于加速：局部速度按窗口内起音包络自相关估计",
                "半速/倍速歧义保留为候选，不强行当真值",
                "拍号需要小节强拍判断，本模块给不出，一律 unknown",
                "自由 rubato 不标成拍点不稳；stable 只在持续一致时为真",
            ],
        },
    }
