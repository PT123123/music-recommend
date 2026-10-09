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
# Semitone intervals above the root. The old constants were named MAJOR/MINOR but
# held *scale* profiles ([1,0,1,0,1,1,...]) that were never transposed per root, so
# all 12 roots of a quality shared one identical template and only 2 of the 24
# vectors were distinct -- root comparison was meaningless (docs/MUSIC_ANALYSIS_REFACTOR_PLAN.md 2.4).
MAJOR_THIRD, MINOR_THIRD, FIFTH = 4, 3, 7


def triad_profile(root_pc: int, third: int) -> np.ndarray:
    """Pitch-class set of a major/minor triad, transposed to `root_pc`."""
    v = np.zeros(12, float)
    v[root_pc % 12] = 1.0
    v[(root_pc + third) % 12] = 1.0
    v[(root_pc + FIFTH) % 12] = 1.0
    return v


# Roman numerals per scale degree for major / natural-minor keys.
ROMAN_MAJOR = ["I", "ii", "iii", "IV", "V", "vi", "iio", "IV", "V", "vi", "VII", "I"]


def _pc(name: str) -> int:
    return NOTE_NAMES.index(name)


def templates() -> list[tuple[str, int, str, np.ndarray]]:
    """Return [(chord_label, root_pc, quality, profile)] over 24 triads."""
    out = []
    for root_pc, root in enumerate(NOTE_NAMES):
        for quality, third in (("maj", MAJOR_THIRD), ("min", MINOR_THIRD)):
            label = root if quality == "maj" else f"{root}m"
            out.append((label, root_pc, quality, triad_profile(root_pc, third)))
    return out


def template_check() -> dict:
    """Self-audit of the 24 templates: distinctness is what makes root comparison real.

    `max_pair_cosine` is how close the two most similar different triads get (a major and
    its parallel minor share root and fifth, so 2/3), so a caller can tell 'templates
    collapsed again' (max -> 1.0) from 'the audio is genuinely ambiguous' (max stays at 2/3).
    """
    tmpl = templates()
    norms = [t[3] / np.linalg.norm(t[3]) for t in tmpl]
    distinct = {tuple(np.round(v, 6)) for v in norms}
    min_cos, min_pair, max_cos, max_pair = 1.0, None, -1.0, None
    for i in range(len(norms)):
        for j in range(i + 1, len(norms)):
            c = float(norms[i] @ norms[j])
            if c < min_cos:
                min_cos, min_pair = c, (tmpl[i][0], tmpl[j][0])
            if c > max_cos:
                max_cos, max_pair = c, (tmpl[i][0], tmpl[j][0])
    return {"n_templates": len(tmpl), "n_distinct": len(distinct),
            "min_pair_cosine": round(min_cos, 4), "min_pair": list(min_pair or []),
            "max_pair_cosine": round(max_cos, 4), "max_pair": list(max_pair or [])}


def estimate_chords(y: np.ndarray, sr: int, hop: float = 0.5) -> dict:
    import librosa
    frames = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)  # (12, n)
    n = frames.shape[1]
    if n == 0:
        return {}
    fps = sr / 512.0
    frame_step = max(1, int(hop * fps))
    tmpl = templates()
    norms = np.stack([t[3] / np.linalg.norm(t[3]) for t in tmpl])

    labels, confs, roots, quals, margins = [], [], [], [], []
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
        margins.append(margin)

    # collapse consecutive identical chords
    seq, seq_conf, seq_roots, seq_quals, seq_margins = [], [], [], [], []
    for i, lab in enumerate(labels):
        if not seq or seq[-1] != lab:
            seq.append(lab); seq_conf.append(confs[i]); seq_roots.append(roots[i])
            seq_quals.append(quals[i]); seq_margins.append(margins[i])
        else:
            seq_conf[-1] = max(seq_conf[-1], confs[i])
            seq_margins[-1] = max(seq_margins[-1], margins[i])

    dur = len(y) / sr
    blocks = max(1, len(seq))
    chord_change_rate = blocks / dur if dur > 0 else 0.0
    harmonic_rhythm = dur / blocks if dur > 0 else 0.0
    mean_conf = float(np.mean(confs)) if confs else 0.0
    # dissonance proxy: 1 - (how strongly the chroma concentrates on one triad)
    dissonance = float(np.clip(1.0 - mean_conf, 0.0, 1.0))
    # How far the winning triad stood above the runner-up. Near zero means the chroma
    # fit several triads equally well, i.e. the root is not actually determined.
    mean_margin = float(np.mean(seq_margins)) if seq_margins else 0.0

    return {
        "chord_sequence": seq,
        "chord_confidences": [round(c, 3) for c in seq_conf],
        "chord_margins": [round(m, 3) for m in seq_margins],
        "chord_margin_mean": round(mean_margin, 4),
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

