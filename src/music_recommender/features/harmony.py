"""Harmony features: chord estimation from chroma via template matching.

This is an ESTIMATE, not ground truth (spec principle 1). Every chord carries a
confidence; results are recorded with `estimated` markers. Supports key-invariant
comparison by emitting an absolute and a relative (Roman-numeral) sequence
(spec section 10).

No external chord model is required; a robust later replacement can implement the
same output contract.
"""
from __future__ import annotations

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], float)   # root, 3rd, 5th
MINOR = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], float)

# Roman numerals per scale degree for major / natural-minor keys.
ROMAN_MAJOR = ["I", "ii", "iii", "IV", "V", "vi", "iio", "IV", "V", "vi", "VII", "I"]


def _pc(name: str) -> int:
    return NOTE_NAMES.index(name)


def _templates():
    """Return [(chord_label, roman_key, profile)] over 24 triads."""
    out = []
    for root_pc, root in enumerate(NOTE_NAMES):
        for quality, prof in (("maj", MAJOR), ("min", MINOR)):
            label = root if quality == "maj" else f"{root}m"
            out.append((label, root_pc, quality, prof))
    return out


def estimate_chords(y: np.ndarray, sr: int, hop: float = 0.5) -> dict:
    import librosa
    frames = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)  # (12, n)
    n = frames.shape[1]
    if n == 0:
        return {}
    fps = sr / 512.0
    frame_step = max(1, int(hop * fps))
    tmpl = _templates()
    norms = np.stack([t[3] / np.linalg.norm(t[3]) for t in tmpl])

    labels, confs, roots, quals = [], [], [], []
    for start in range(0, n, frame_step):
        block = frames[:, start:start + frame_step].mean(axis=1)
        norm = np.linalg.norm(block)
        if norm == 0:
            continue
        block = block / norm
        scores = norms @ block  # cosine to each triad template
        best = int(np.argmax(scores))
        label, root_pc, quality, _ = tmpl[best]
        margin = float(scores[best] - np.sort(scores)[-2])
        labels.append(label)
        confs.append(float(scores[best]))
        roots.append(root_pc)
        quals.append(quality)

    # collapse consecutive identical chords
    seq, seq_conf, seq_roots, seq_quals = [], [], [], []
    for i, lab in enumerate(labels):
        if not seq or seq[-1] != lab:
            seq.append(lab); seq_conf.append(confs[i]); seq_roots.append(roots[i]); seq_quals.append(quals[i])
        else:
            seq_conf[-1] = max(seq_conf[-1], confs[i])

    dur = len(y) / sr
    blocks = max(1, len(seq))
    chord_change_rate = blocks / dur if dur > 0 else 0.0
    harmonic_rhythm = dur / blocks if dur > 0 else 0.0
    mean_conf = float(np.mean(confs)) if confs else 0.0
    # dissonance proxy: 1 - (how strongly the chroma concentrates on one triad)
    dissonance = float(np.clip(1.0 - mean_conf, 0.0, 1.0))

    return {
        "chord_sequence": seq,
        "chord_confidences": [round(c, 3) for c in seq_conf],
        "chord_roots": seq_roots,
        "chord_qualities": seq_quals,
        "chord_change_rate": round(chord_change_rate, 4),
        "harmonic_rhythm": round(harmonic_rhythm, 4),
        "dissonance_mean": round(dissonance, 4),
        "estimated": True,
    }


def relative_sequence(roots: list[int], qualities: list[str], key: str, mode: str) -> list[str]:
    """Key-invariant view of the chord progression (spec section 10).

    Encodes each chord as <semitone-offset-from-tonic>:<quality>, so
    C-G-Am-F and D-A-Bm-G both become ``0:maj, 7:maj, 9:min, 5:maj`` and compare
    as equal regardless of absolute key. A canonical Roman-numeral string is also
    derivable downstream if desired.
    """
    tonic = _pc(key) if key in NOTE_NAMES else 0
    return [f"{(r - tonic) % 12}:{q}" for r, q in zip(roots, qualities)]

