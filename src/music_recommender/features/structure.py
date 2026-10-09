"""Time-structure analysis on the FULL track: boundaries, repeating-section groups, honest labels.

Three results that used to be conflated (docs/MUSIC_ANALYSIS_REFACTOR_PLAN.md 1, 2.1) are now
separate fields, because each has its own confidence:

  1. `boundary_confidence`  -- something changed here (spectral/harmonic/rhythmic novelty).
  2. `group`                -- these sections resemble each other, i.e. a repeat (A/B/C).
  3. `label`                -- the section's *song function* (verse/chorus/...).

A detected boundary is not a chorus, and two similar sections are not proof of the same
chorus. The old implementation labelled segment[0] intro and segment[-1] outro unconditionally
and filled the rest by alternating on index parity, which produced confident, wrong,
self-consistent-looking sequences on songs with no intro, on quiet choruses and on two-section
songs. That logic is gone. Without a supervised model (plan stage D) interior sections stay
`unknown` and the A/B/C repeat pattern carries the information.
"""
from __future__ import annotations

import numpy as np

from . import dynamics

DEFAULTS: dict = {
    "min_segment_seconds": 8.0,
    "max_segment_seconds": 45.0,
    "max_segments": 12,
    "min_beats_for_beat_sync": 16,
    "fallback_window_seconds": 1.0,
    # 段间相似度是「减去本曲各段均值后的家族余弦」，不是原始余弦。0.20 是在真实曲库 19 首上
    # 数出来的：0.05-0.25 之间每首平均 3-4 组、19/19 首都有 size>=3 的重复组，而 +0.20 这一步
    # 只有 1 首的答案发生变化（最稳的一格）；再往上涨到 0.375 就开始把同一段副歌拆散。
    "group_similarity_threshold": 0.20,
    "group_feature_weights": {"chroma": 0.6, "timbre": 0.25, "density": 0.15},
    "chorus_min_repeat": 2,
    "chorus_min_energy_pct": 0.50,
    "chorus_min_onset_pct": 0.40,
    "label_max_confidence": 0.60,
    "edge_max_energy_pct": 0.35,
    "edge_max_onset_pct": 0.40,
    "novelty_smoothing_blocks": 3,
    "curve_bins": 64,
}

# Per-block descriptor layout: 12 chroma bins, then band share / brightness / density / change.
_CHROMA, _EXTRA = 12, 4


def _cfg_value(cfg: dict | None, key: str):
    if cfg:
        v = cfg.get("analysis", {}).get(key)
        if v is not None:
            return v
    return DEFAULTS[key]


def structure_features(y: np.ndarray, sr: int, cfg: dict | None = None,
                       frames: dict | None = None,
                       separated: tuple | None = None) -> dict:
    """Segment the complete track, group repeats, and label only what the evidence supports.

    `separated` is accepted and ignored: HPSS stems came from the 45 s analysis window, so
    stamping their energies onto full-track segments attached the wrong numbers to the wrong
    times. Sections report `onset_activity` (a percussive-activity proxy) instead of
    `drum_energy`, and no vocal energy at all -- a harmonic/percussive split is not source
    separation (plan 2.3).
    """
    frames = frames if frames is not None else dynamics.frame_curves(y, sr, cfg)
    duration = float(len(y) / sr)
    if frames["n_frames"] < 8 or duration <= 0:
        return {"segments": [], "segment_list": [], "segment_type_sequence": "",
                "group_sequence": "", "duration_ratios": [], "energy_curve": [0.0] * 64,
                "chorus_repeat_count": 0, "estimated": True, "alignment": "none"}

    blocks = _blocks(frames, cfg, int(sr), float(duration))
    novelty = _novelty(blocks["group_matrix"], cfg)
    edges = _boundaries(novelty, blocks, cfg)
    segs = _segments(blocks, edges, novelty, cfg)
    _assign_groups(segs, cfg)
    _label_segments(segs, cfg)
    for s in segs:
        s.pop("descriptor", None)          # internal clustering input, never serialized

    curve = _energy_curve(frames, _cfg_value(cfg, "curve_bins"))
    chorus = _chorus(segs)

    out = {
        "segments": segs,
        "segment_list": segs,                     # legacy key, same payload
        "segment_type_sequence": ",".join(s["label"] for s in segs),
        "group_sequence": ",".join(s["group"] for s in segs),
        "duration_ratios": [round((s["end"] - s["start"]) / duration, 4) for s in segs],
        "energy_curve": curve,
        "alignment": blocks["alignment"],
        "tempo_bpm": blocks["tempo_bpm"],
        "group_count": len({s["group"] for s in segs}),
        "n_segments": len(segs),
        "label_source": "heuristic_repeat",
        "labels_are_estimates": True,
        "estimated": True,
    }
    out.update(chorus)
    return out


# ---- block features (beat-aligned when the beat track is trustworthy) ----

def _blocks(frames: dict, cfg: dict | None, sr: int, duration: float) -> dict:
    import librosa

    hop = frames["hop_length"]
    n = frames["n_frames"]
    chroma = frames["chroma"][:, :n]
    cn = np.linalg.norm(chroma, axis=0)
    chroma = chroma / np.where(cn > 0, cn, 1.0)

    matrix = np.vstack([chroma,
                        frames["low"][:n],
                        np.log1p(frames["bright"][:n]) / 8.0,
                        np.log1p(frames["onset"][:n]),
                        frames["change"][:n]]).T          # (n_frames, _CHROMA + _EXTRA)
    # Loudness is deliberately NOT in `matrix`: a repeated chorus that got arranged thicker is
    # the same section, and letting dB dominate the cosine similarity split it into its own
    # group (plan stage C.3-4 -- repeats tolerate time stretch and arrangement changes).
    # It stays a separate per-block series for the energy/arrangement comparison.
    loud = frames["rms_db"][:n]

    beat_frames: np.ndarray | None = None
    tempo = 0.0
    try:
        tempo_arr, beat_frames = librosa.beat.beat_track(onset_envelope=frames["onset"],
                                                         sr=sr, hop_length=hop, trim=False)
        tempo = float(np.atleast_1d(tempo_arr)[0])
    except Exception:
        beat_frames = None
    if beat_frames is None or len(beat_frames) < int(_cfg_value(cfg, "min_beats_for_beat_sync")):
        # Too few beats to align on (rubato, sparse percussion, tracker failure). Fall back to
        # a fixed time grid rather than trusting a garbage beat track (plan stage C.1).
        win = max(1, int(round(float(_cfg_value(cfg, "fallback_window_seconds")) * sr / hop)))
        starts = list(range(0, n, win))
        alignment = "time"
        block_len = win * hop / sr
    else:
        b = sorted({int(x) for x in beat_frames if 0 <= int(x) < n})
        starts = [0] + b
        alignment = "beat"
        block_len = float(np.mean(np.diff(b))) * hop / sr if len(b) > 1 else 0.0

    rows, loud_rows, row_blocks = [], [], []

    def _consume(s: int, e: int) -> None:
        rows.append(matrix[s:e].mean(axis=0))
        loud_rows.append(float(loud[s:e].mean()))
        row_blocks.append((s, e))

    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else n
        if e <= s:
            continue
        _consume(s, e)
    if len(rows) < 3:
        # a beat track that produced almost no usable blocks (heavily clipped tracker output)
        rows, loud_rows, row_blocks = [], [], []
        win = max(1, n // 24)
        for s in range(0, n - 1, win):
            _consume(s, min(n, s + win))
        alignment, block_len = "time", win * hop / sr

    mat = np.asarray(rows, float)
    # `group_matrix` is the clustering/novelty view: chroma keeps its real cosine meaning,
    # but each non-harmonic column is stretched to this track's own [0,1] range. Raw units
    # cannot be cosine'd together -- brightness (thousands of Hz) swamped the low-band share
    # (0-1) so every pair scored ~1.0 on timbre and two clearly different sections looked
    # identical. Loudness stays out of the matrix entirely (see above).
    ex = mat[:, _CHROMA:]
    lo_ex = ex.min(axis=0)
    rng = ex.max(axis=0) - lo_ex
    scaled = np.divide(ex - lo_ex, np.where(rng > 1e-9, rng, 1.0))
    group = np.hstack([mat[:, :_CHROMA], np.where(rng > 1e-9, scaled, 0.0)])

    return {"matrix": mat, "group_matrix": group, "loud": np.asarray(loud_rows, float),
            "block_frames": row_blocks,
            "alignment": alignment, "tempo_bpm": round(tempo, 2),
            "block_seconds": float(block_len), "hop": hop, "sr": sr, "duration": duration}


def _novelty(m: np.ndarray, cfg: dict | None) -> np.ndarray:
    """Change between consecutive blocks: harmonic (chroma) distance plus feature movement."""
    if m.shape[0] < 2:
        return np.zeros(m.shape[0])
    prev = m[:-1]
    cur = m[1:]
    chroma_d = 1.0 - np.sum(prev[:, :_CHROMA] * cur[:, :_CHROMA], axis=1)
    other = np.abs(prev[:, _CHROMA:] - cur[:, _CHROMA:]).mean(axis=1)
    nov = np.concatenate([[0.0], np.maximum(chroma_d, 0.0) + other])
    w = int(_cfg_value(cfg, "novelty_smoothing_blocks"))
    if nov.size >= w >= 3:
        pad = w // 2
        nov = np.convolve(np.pad(nov, (pad, pad), mode="edge"),
                          np.ones(w) / w, mode="valid")[: nov.size]
    return nov


def _boundaries(novelty: np.ndarray, blocks: dict, cfg: dict | None) -> list[tuple[int, float]]:
    """Every change that stands out, constrained to min/max section length.

    Returns [(block index, confidence)] starting at 0 and ending at len(novelty).

    Peak-picking, not greedy argmax-with-lookahead: the greedy version always jumped to the
    strongest change inside its 8-45 s window, so one loud late transition swallowed the
    earlier ones and a six-section song came back as two. `find_peaks` takes all of them,
    `distance=min_segment` is what keeps metering noise from producing dozens of 2-second
    'sections' (plan C.2), and the max-length pass only splits where the music really moved.
    """
    from scipy.signal import find_peaks

    m = int(novelty.size)
    if m < 3:
        return [(0, 1.0), (m, 1.0)]
    block_s = max(blocks["block_seconds"], 1e-3)
    lo_blocks = max(1, int(round(float(_cfg_value(cfg, "min_segment_seconds")) / block_s)))
    hi_blocks = max(lo_blocks + 1, int(round(float(_cfg_value(cfg, "max_segment_seconds")) / block_s)))
    max_segments = int(_cfg_value(cfg, "max_segments"))

    med = float(np.median(novelty))
    spread = max(float(np.percentile(novelty, 95)) - med, 1e-6)
    peaks, _ = find_peaks(novelty, distance=lo_blocks, prominence=0.20 * spread)
    edges = sorted({0, m} | {int(x) for x in peaks if 0 < int(x) < m})

    # A span longer than max_segment_seconds is split only where there is evidence to put the
    # split: a genuinely unchanged passage stays one segment rather than gaining a boundary
    # invented to fit a display limit.
    while True:
        added = None
        for a, b in zip(edges, edges[1:]):
            if b - a > hi_blocks:
                lo_i, hi_i = a + lo_blocks, b - lo_blocks
                if hi_i > lo_i:
                    cand = lo_i + int(np.argmax(novelty[lo_i:hi_i]))
                    if float(novelty[cand]) > med:
                        added = cand
                        break
        if added is None:
            break
        edges.append(added)
        edges = sorted(set(edges))

    # a boundary closer than min_segment_seconds to an edge leaves a stub section
    while len(edges) > 2 and edges[1] < lo_blocks:
        del edges[1]
    # the tail gets no vote: a final segment shorter than min_segment_seconds is the tracker
    # running out of track, not a real section
    while len(edges) > 2 and m - edges[-2] < lo_blocks:
        del edges[-2]

    if len(edges) - 1 > max_segments:
        interior = edges[1:-1]
        drop = sorted(interior, key=lambda x: float(novelty[min(x, m - 1)]))[:len(interior) - (max_segments - 1)]
        edges = [x for x in edges if x not in set(drop)]

    p98 = float(np.percentile(novelty, 98)) if novelty.size else 0.0
    p25 = float(np.percentile(novelty, 25)) if novelty.size else 0.0
    spread = max(p98 - p25, 1e-6)
    out: list[tuple[int, float]] = []
    for k, x in enumerate(edges):
        if k == 0:
            out.append((0, 1.0))
        elif k == len(edges) - 1:
            out.append((m, 1.0))
        else:
            conf = float(np.clip((float(novelty[min(x, m - 1)]) - p25) / spread, 0.05, 0.99))
            out.append((x, round(conf, 3)))
    return out


def _segments(blocks: dict, edges: list[tuple[int, float]], novelty: np.ndarray,
              cfg: dict | None) -> list[dict]:
    m, gm, bf = blocks["matrix"], blocks["group_matrix"], blocks["block_frames"]
    sr = blocks["sr"]
    segs: list[dict] = []
    for i in range(len(edges) - 1):
        (a, conf_a), (b, _conf_b) = edges[i], edges[i + 1]
        if b <= a:
            continue
        rows = m[a:b]
        start = bf[a][0] * blocks["hop"] / sr
        end = min(bf[b - 1][1] * blocks["hop"] / sr, blocks["duration"])
        evidence = []
        if conf_a >= 0.5 and i > 0:
            evidence.append("novelty_boundary")
        if blocks["alignment"] == "beat":
            evidence.append("beat_aligned")
        else:
            evidence.append("time_aligned")
        segs.append({
            "index": i,
            "start": round(float(start), 2),
            "end": round(float(end), 2),
            "boundary_confidence": conf_a,
            "evidence": evidence,
            "descriptor": gm[a:b].mean(axis=0),   # range-scaled view, for grouping only
            "energy_mean": round(float(np.mean(blocks["loud"][a:b])), 4),
            "onset_activity": round(float(np.mean(np.expm1(rows[:, _CHROMA + 2]))), 4),
            "bass_energy": round(float(np.mean(rows[:, _CHROMA])), 4),
            "spectral_centroid": round(float(np.mean(np.expm1(rows[:, _CHROMA + 1]))), 2),
            "arrangement_change": round(float(np.mean(rows[:, _CHROMA + 3])), 4),
            # No drum_energy / vocal_energy: those were HPSS stem energies of the 45 s window
            # stamped onto full-track timestamps. `onset_activity` is the honest proxy name.
        })
    if not segs:
        return segs
    for s in segs:
        s["duration"] = round(max(0.0, s["end"] - s["start"]), 2)
    return segs


# ---- repeat grouping ----

FAMILY_SLICES = {"chroma": slice(0, _CHROMA), "timbre": slice(_CHROMA, _CHROMA + 2),
                 "density": slice(_CHROMA + 2, _CHROMA + _EXTRA)}


def family_similarity(x: np.ndarray, weights: dict | None = None,
                      center: bool = True) -> tuple[np.ndarray, float]:
    """Weighted cosine between sections, one family at a time. Returns (sim, weight used).

    `center=False` reproduces the old raw cosine, and that is the control: every descriptor
    dimension is non-negative (normalised chroma energies, band shares scaled to [0,1]), so any
    two sections of one song sit at ~0.93-0.99 no matter whether they repeat. Removing each
    family's mean over this track's own sections keeps the family's scale (a per-column z-score
    would let the noisiest dim veto everything) and leaves only the direction each section
    deviates in -- which is what "this is the same part again" actually claims.
    """
    sim = np.zeros((x.shape[0], x.shape[0]))
    used = 0.0
    for name, sl in FAMILY_SLICES.items():
        w = float((weights or {}).get(name, DEFAULTS["group_feature_weights"].get(name, 0.0)))
        if w <= 0:
            continue
        u = x[:, sl]
        if center:
            u = u - u.mean(axis=0, keepdims=True)
            if float(np.abs(u).max()) < 1e-9:
                continue                          # this family says nothing about this track
        u = u / np.linalg.norm(u, axis=1, keepdims=True).clip(1e-9)
        sim += w * (u @ u.T)
        used += w
    return (sim / used if used > 0 else sim), used


def _assign_groups(segs: list[dict], cfg: dict | None) -> None:
    """Cluster sections by content, not by position: same `group` means 'this repeats'.

    A family that does not vary across this track's sections is dropped and its weight
    redistributed (plan stage C.3-4).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    x = np.stack([s["descriptor"] for s in segs])
    if x.shape[0] == 1:
        segs[0]["group"] = "A"
        segs[0]["group_size"] = 1
        return

    sim, used = family_similarity(x, _cfg_value(cfg, "group_feature_weights"))
    if used <= 0:
        for i, s in enumerate(segs):              # nothing varies: every section stands alone
            s["group"] = chr(ord("A") + i) if i < 26 else f"G{i}"
            s["group_size"] = 1
        return

    dist = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    thr = float(_cfg_value(cfg, "group_similarity_threshold"))
    linked = linkage(squareform(dist, checks=False), method="average")
    # 'distance' criterion on average linkage: a cluster only merges while the *average*
    # pairwise distance stays under the threshold, which blocks the chaining effect that
    # single linkage would use to swallow the whole song into one group.
    lab = fcluster(linked, t=1.0 - thr, criterion="distance")

    first: dict[int, str] = {}
    for s, l in zip(segs, lab):
        k = int(l)
        if k not in first:
            n = len(first)
            first[k] = chr(ord("A") + n) if n < 26 else f"G{n}"
        s["group"] = first[k]
    sizes = {}
    for s in segs:
        sizes[s["group"]] = sizes.get(s["group"], 0) + 1
    for s in segs:
        s["group_size"] = sizes[s["group"]]


# ---- song-function labels: only where the evidence carries it ----

def _pct_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return order / max(1, values.size - 1)


def _label_segments(segs: list[dict], cfg: dict | None) -> None:
    cap = float(_cfg_value(cfg, "label_max_confidence"))
    energy = _pct_rank(np.array([s["energy_mean"] for s in segs], float))
    onset = _pct_rank(np.array([s["onset_activity"] for s in segs], float))

    groups: dict[str, list[int]] = {}
    for i, s in enumerate(segs):
        groups.setdefault(s["group"], []).append(i)
    reps = {g: len(v) for g, v in groups.items()}
    min_rep = int(_cfg_value(cfg, "chorus_min_repeat"))

    def strength(idxs: list[int]) -> tuple[float, float]:
        return (float(np.mean([energy[i] for i in idxs])), float(np.mean([onset[i] for i in idxs])))

    candidates = [(g, idxs) for g, idxs in groups.items() if reps[g] >= min_rep]
    scores = {g: strength(idxs) for g, idxs in candidates}
    chorus_g = None
    e_min = float(_cfg_value(cfg, "chorus_min_energy_pct"))
    o_min = float(_cfg_value(cfg, "chorus_min_onset_pct"))
    qualified = [(g, e, o) for g, (e, o) in scores.items() if e >= e_min and o >= o_min]
    if qualified:
        chorus_g = max(qualified, key=lambda t: (t[1] * t[2] * (reps[t[0]] ** 0.5)))[0]

    for i, s in enumerate(segs):
        g = s["group"]
        e, o = energy[i], onset[i]
        s["label"] = "unknown"
        s["label_confidence"] = 0.0
        ev = list(s["evidence"])
        # Group membership picks the *candidate*, the segment's own loudness/density decides:
        # a quiet section that clustered with the choruses (same chords, thinner arrangement)
        # must not inherit "chorus" from its group's average. That averaged-away defect labelled
        # a silent intro as the hook on the first hand-checked fixture.
        if chorus_g is not None and g == chorus_g and e >= e_min and o >= o_min:
            conf = 0.20 + 0.14 * (reps[g] - 1) + 0.5 * (e - e_min) + 0.3 * (o - o_min)
            s["label"] = "chorus"
            s["label_confidence"] = round(min(cap, max(0.25, conf)), 3)
            ev += ["repeat_group", "above_median_loudness", "above_median_density"]
        elif chorus_g is not None and g != chorus_g and reps[g] >= min_rep and e < scores[chorus_g][0]:
            conf = 0.15 + 0.14 * (reps[g] - 1) + 0.4 * (scores[chorus_g][0] - e)
            s["label"] = "verse"
            s["label_confidence"] = round(min(cap * 0.9, max(0.2, conf)), 3)
            ev += ["repeat_group", "distinct_from_chorus_group"]
        elif i == 0 and e < float(_cfg_value(cfg, "edge_max_energy_pct")) and o < float(_cfg_value(cfg, "edge_max_onset_pct")):
            s["label"] = "intro"
            s["label_confidence"] = round(min(0.5, 0.3 + 0.4 * (0.35 - e)), 3)
            ev += ["first_section", "quiet_and_sparse"]
        elif i == len(segs) - 1 and len(segs) > 2 and e < float(_cfg_value(cfg, "edge_max_energy_pct")):
            s["label"] = "outro"
            s["label_confidence"] = round(min(0.5, 0.3 + 0.4 * (0.35 - e)), 3)
            ev += ["last_section", "below_median_loudness"]
        if s["label"] == "unknown" and g != "":
            ev.append(f"repeat_group_of_{reps[g]}")
        s["evidence"] = sorted(set(ev))
    # pre_chorus / bridge stay un-emitted: a repeated-section rule cannot tell a bridge from
    # an odd verse, and `unknown` is the honest answer until a model covers it (plan stage D).


def _chorus(segs: list[dict]) -> dict:
    idxs = [i for i, s in enumerate(segs) if s["label"] == "chorus"]
    if not idxs:
        return {}
    loudest = max(idxs, key=lambda i: segs[i]["energy_mean"])
    c = segs[loudest]
    return {
        "chorus_repeat_count": len(idxs),
        "chorus_energy": c["energy_mean"],
        "chorus_start": c["start"],
        "chorus_end": c["end"],
    }


def _energy_curve(frames: dict, n_bins: int) -> list[float]:
    """Legacy 64-bucket, max-normalised curve -- now over the WHOLE track."""
    rms = frames["rms"][: frames["n_frames"]]
    if rms.size == 0:
        return [0.0] * int(n_bins)
    edges = np.linspace(0, rms.size, int(n_bins) + 1).astype(int)
    curve = [float(rms[a:b].mean()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    cmax = max(curve) or 1.0
    return [round(c / cmax, 4) for c in curve]
