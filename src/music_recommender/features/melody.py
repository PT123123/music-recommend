"""Melody / pitch features via librosa.pyin (spec sections 11-12).

Produces a key/transposition-tolerant pitch view: statistics + relative interval
sequence (absolute C-D-E -> +2,+2 deltas, spec section 11) and low/mid/high pitch
ratios used to find extremes like "极致高音女声".

pyin is CPU-heavy, so analysis is bounded to `max_seconds` windows; pitch fields
are marked estimated.
"""
from __future__ import annotations

import numpy as np

FMIN, FMAX = 65.0, 1000.0  # ~C2..C6, sensible for vocals & many instruments
# MIDI band boundaries for low/mid/high
LOW_MIDI, HIGH_MIDI = 55.0, 72.0


def melody_features(y: np.ndarray, sr: int, max_seconds: float = 30.0) -> dict:
    import librosa
    hop = 256
    win_len = int(min(len(y), max_seconds * sr))
    y_seg = y[:win_len]
    f0, voiced, _ = librosa.pyin(y_seg, sr=sr, fmin=FMIN, fmax=FMAX, frame_length=2048, hop_length=hop)
    midi = librosa.hz_to_midi(f0)
    v_mask = ~np.isnan(midi)
    vm = np.asarray(midi[v_mask], float)

    out = {
        "pitch_frame_count": int(len(midi)),
        "voiced_ratio": float(v_mask.mean()) if len(midi) else 0.0,
        "estimated": True,
    }
    if vm.size < 3:
        out.update({"pitch_mean": None, "pitch_min": None, "pitch_max": None,
                    "pitch_range": 0.0, "pitch_variance": 0.0,
                    "low_pitch_ratio": 0.0, "mid_pitch_ratio": 0.0, "high_pitch_ratio": 0.0,
                    "interval_sequence": [], "relative_pitch_sequence": [],
                    "interval_histogram": {}, "melody_contour_type": "unknown",
                    "melody_density": 0.0})
        return out

    out.update({
        "pitch_mean": float(vm.mean()), "pitch_min": float(vm.min()), "pitch_max": float(vm.max()),
        "pitch_range": float(vm.max() - vm.min()), "pitch_variance": float(vm.var()),
        "low_pitch_ratio": float((vm < LOW_MIDI).mean()),
        "mid_pitch_ratio": float(((vm >= LOW_MIDI) & (vm <= HIGH_MIDI)).mean()),
        "high_pitch_ratio": float((vm > HIGH_MIDI).mean()),
    })

    # intervals between successive voiced frames (semitones), capped to sane jumps
    steps = np.diff(vm)
    steps = steps[np.abs(steps) <= 24]
    rel = [int(round(s)) for s in steps]
    out["relative_pitch_sequence"] = rel
    out["interval_sequence"] = [abs(r) for r in rel]
    hist = Counter = {}
    for r in rel:
        Counter[abs(r)] = Counter.get(abs(r), 0) + 1
    out["interval_histogram"] = Counter

    # contour type from the coarse trajectory of the melody
    smooth = vm[np.linspace(0, len(vm) - 1, min(8, len(vm))).astype(int)]
    d = np.diff(smooth)
    if np.all(d >= -0.5):
        contour = "ascending"
    elif np.all(d <= 0.5):
        contour = "descending"
    elif d[0] > 0 and d[-1] < 0:
        contour = "arch"
    elif d[0] < 0 and d[-1] > 0:
        contour = "valley"
    else:
        contour = "mixed"
    out["melody_contour_type"] = contour
    out["melody_density"] = float((np.abs(steps) > 1).mean()) if steps.size else 0.0
    return out
