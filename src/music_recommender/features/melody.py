"""Melody / pitch features via librosa.pyin (spec sections 11-12).

Produces a key/transposition-tolerant pitch view: statistics + relative interval
sequence (absolute C-D-E -> +2,+2 deltas, spec section 11) and low/mid/high pitch
ratios used to find extremes like "极致高音女声".

pyin is CPU-heavy, so analysis is bounded to `max_seconds` windows; pitch fields
are marked estimated.

The interval sequence is built from *note events*, not from raw frames. Raw
per-frame deltas run to ~2700 tokens on a 45s window and make Levenshtein (O(n*m))
the dominant cost of song->song. Note events also carry more signal: pyin's
frame-to-frame error (~0.3-0.5 semitone) is the size of one semitone, so 1-semitone
buckets read rounding flip-flop as melody motion (83-90% of measured steps).
Quantise width and median smoothing are both config-driven (`features.*`).
"""
from __future__ import annotations

import numpy as np

FMIN, FMAX = 65.0, 1100.0  # ~C2..C6, sensible for vocals & many instruments
# MIDI band boundaries for low/mid/high
LOW_MIDI, HIGH_MIDI = 55.0, 72.0


def cap_sequence(seq: list, max_tokens: int) -> list:
    """Evenly subsample a sequence to at most `max_tokens` items.

    Keeps the first and last element so overall shape/direction survives; this is
    a budget guard, not a musical transform.
    """
    n = len(seq)
    if not max_tokens or max_tokens <= 0 or n <= max_tokens:
        return seq
    idx = np.unique(np.linspace(0, n - 1, max_tokens).round().astype(int))
    return [seq[int(i)] for i in idx]


def _note_events(midi: np.ndarray, quantise: float = 2.0, smooth: int = 5) -> list[float]:
    """Collapse frames that fall in the same pitch bucket into one note event.

    Bucket width and median smoothing exist because pyin's per-frame error is itself
    about a semitone: at 1-semitone buckets, 83-90% of the measured "note changes" on
    real tracks were rounding flip-flop, not melody motion.
    """
    if midi.size == 0:
        return []
    v = midi
    if smooth and smooth >= 3:
        from scipy.ndimage import median_filter
        v = median_filter(midi, size=int(smooth), mode="nearest")
    q = np.round(v / quantise).astype(np.int64)
    keep = np.concatenate(([True], q[1:] != q[:-1]))
    return [float(x) for x in (q[keep] * quantise)]


def melody_features(y: np.ndarray, sr: int, max_seconds: float = 30.0,
                    f0: np.ndarray | None = None, hop: int = 256) -> dict:
    """Args:
        f0: precomputed pyin result for `y` (from the shared extraction pass).
            When omitted, pyin runs here. Passing it in keeps pitch analysis
            single-computation across melody/vocal.
    """
    import librosa

    if f0 is None:
        win_len = int(min(len(y), max_seconds * sr))
        f0, _, _ = librosa.pyin(y[:win_len], sr=sr, fmin=FMIN, fmax=FMAX,
                                frame_length=2048, hop_length=hop)
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
                    "melody_density": 0.0, "note_event_count": 0})
        return out

    out.update({
        "pitch_mean": float(vm.mean()), "pitch_min": float(vm.min()), "pitch_max": float(vm.max()),
        "pitch_range": float(vm.max() - vm.min()), "pitch_variance": float(vm.var()),
        "low_pitch_ratio": float((vm < LOW_MIDI).mean()),
        "mid_pitch_ratio": float(((vm >= LOW_MIDI) & (vm <= HIGH_MIDI)).mean()),
        "high_pitch_ratio": float((vm > HIGH_MIDI).mean()),
    })

    # intervals between successive *note events* (semitones), capped to sane jumps
    from ..utils.config import get_config
    cfgf = get_config().features
    notes = np.asarray(_note_events(
        vm,
        float(cfgf.get("melody_note_quantise_semitones", 2)),
        int(cfgf.get("melody_pitch_smooth_frames", 5)),
    ), float)
    out["note_event_count"] = int(notes.size)

    steps = np.diff(notes)
    steps = steps[np.abs(steps) <= 24]
    rel_raw = [int(round(s)) for s in steps]
    budget = int(cfgf.get("max_sequence_tokens", 120))
    rel = cap_sequence(rel_raw, budget)
    out["relative_pitch_sequence"] = rel
    out["interval_sequence"] = cap_sequence([abs(r) for r in rel_raw], budget)

    # Histogram is a full-population statistic: keep it over *all* note events so the
    # distribution is not biased by the sequence length budget.
    hist: dict[str, int] = {}
    for r in rel_raw:
        k = str(abs(r))
        hist[k] = hist.get(k, 0) + 1
    out["interval_histogram"] = hist

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
