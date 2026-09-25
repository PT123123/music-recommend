"""Query-path cost refactors must not change results (docs/PERFORMANCE.md, ADR-15).

Each fast path is paired with a deliberately naive reference of the same formula:
textbook Levenshtein DP, per-pair searchsorted percentiles, per-pair numpy cosines.
The reference is the behaviour spec -- if the two disagree, the fast path is wrong.
Writing these references is what caught three real defects while vectorizing the MMR
loop: candidate positions used to index the embedding matrix (twice), and a running
max initialised at 0 that silently clamped diversity to <= 1.0 -- cosine can be
negative, so 1 - cos legitimately exceeds 1 for an anti-correlated pick.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "src"))
sys.path.insert(0, str(TESTS))

GROUPS = ("energy", "timbre", "rhythm", "harmony", "melody", "vocal", "instrumentation", "structure")


# ---- reference implementations (slow on purpose) ----

def _dp_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def _ref_lev(a: list, b: list) -> float:
    if not a and not b:
        return 1.0
    return 1.0 - _dp_distance(a, b) / max(len(a), len(b))


def _ref_combined(a: list, b: list) -> float:
    from music_recommender.recommendation.sequence_sim import (
        jaccard_similarity, ngram_similarity, transition_similarity)
    if not a or not b:
        return 0.0
    return float(0.35 * _ref_lev(a, b) + 0.25 * ngram_similarity(a, b)
                 + 0.2 * jaccard_similarity(a, b) + 0.2 * transition_similarity(a, b))


def _ref_group_similarity(rows: dict, sorted_vals: dict, a_id: str, b_id: str, group: str):
    """Music Space before the precompute: searchsorted per pair, DP edit distance."""
    from music_recommender.recommendation.space import FEATURE_GROUPS, SEQUENCE_COLS
    if group == "structure":
        ra, rb = rows[a_id], rows[b_id]
        sa = [x for x in str(ra["segment_type_sequence"] or "").split(",") if x]
        sb = [x for x in str(rb["segment_type_sequence"] or "").split(",") if x]
        seq_sim = _ref_combined(sa, sb) if len(sa) >= 2 and len(sb) >= 2 else None
        ca = json.loads(ra["energy_curve"]) if ra["energy_curve"] else []
        cb = json.loads(rb["energy_curve"]) if rb["energy_curve"] else []
        if len(ca) == len(cb) and len(ca) > 1 and any(ca) and any(cb):
            va, vb = np.asarray(ca, float), np.asarray(cb, float)
            corr = float(np.corrcoef(va, vb)[0, 1]) if va.std() > 0 and vb.std() > 0 else 0.0
            curve = float(np.clip((corr + 1) / 2, 0.0, 1.0))
            return curve if seq_sim is None else float(np.clip(0.6 * curve + 0.4 * seq_sim, 0.0, 1.0))
        return seq_sim

    cols = FEATURE_GROUPS.get(group) or []
    if group == "vocal" and not (rows[a_id]["has_vocal"] and rows[b_id]["has_vocal"]):
        return None

    def pct(tid, col):
        v, arr = rows[tid][col], sorted_vals.get(col)
        if v is None or arr is None:
            return None
        return float(np.searchsorted(arr, float(v), side="right") / len(arr))

    diffs = []
    for c in cols:
        pa, pb = pct(a_id, c), pct(b_id, c)
        if pa is not None and pb is not None:
            diffs.append(abs(pa - pb))
    scalar = float(np.clip(1.0 - (sum(diffs) / len(diffs)), 0.0, 1.0)) if diffs else None
    seq_col = SEQUENCE_COLS.get(group)
    if seq_col:
        def seq(tid):
            raw = rows[tid][seq_col]
            if not raw:
                return []
            if seq_col == "relative_pitch_sequence":
                return list(json.loads(raw))
            return [x for x in str(raw).split(",") if x]
        sa, sb = seq(a_id), seq(b_id)
        if len(sa) >= 2 and len(sb) >= 2:
            seq_sim = _ref_combined(sa, sb)
            return seq_sim if scalar is None else float(np.clip(0.5 * scalar + 0.5 * seq_sim, 0.0, 1.0))
    return scalar


# ---- tests ----

def test_levenshtein_matches_textbook_dp():
    from music_recommender.recommendation.sequence_sim import levenshtein_similarity

    rng = np.random.default_rng(11)
    vocab = ["0:maj", "5:maj", "9:min", "chorus", "verse"]
    cases = [([], []), ([], [1, 2, 3]), (["0:maj"], ["0:maj"])]
    for _ in range(150):
        la, lb = int(rng.integers(0, 130)), int(rng.integers(0, 130))
        kind = int(rng.integers(0, 3))
        if kind == 0:      # int tokens including negatives (relative pitch steps)
            a = [int(x) for x in rng.integers(-12, 13, la)]
            b = [int(x) for x in rng.integers(-12, 13, lb)]
        elif kind == 1:    # multi-character tokens (chord labels, segment types)
            a = [vocab[int(i)] for i in rng.integers(0, 5, la)]
            b = [vocab[int(i)] for i in rng.integers(0, 5, lb)]
        else:              # near-duplicate sequences, the common real case
            a = [int(x) for x in rng.integers(0, 4, max(la, 2))]
            b = a[:max(1, len(a) // 2)] + [7]
        cases.append((a, b))
    for a, b in cases:
        assert levenshtein_similarity(a, b) == _ref_lev(a, b), (a, b)


def test_group_similarity_matches_naive_reference(tmp_path):
    from music_recommender.database import repository
    from music_recommender.recommendation.space import FEATURE_GROUPS
    from test_feed import _build

    conn, store, space, _ = _build(tmp_path)
    rows = {r["track_id"]: dict(r) for r in repository.all_tracks(conn)}
    sorted_vals = {}
    for c in sorted({col for group in FEATURE_GROUPS.values() for col in group}):
        vals = [float(r[c]) for r in rows.values() if r.get(c) is not None]
        if vals:
            sorted_vals[c] = np.sort(np.asarray(vals, float))

    ids = sorted(rows)
    comparisons = 0
    for a in ids:
        for b in ids:
            for g in GROUPS:
                fast, slow = space.group_similarity(a, b, g), _ref_group_similarity(rows, sorted_vals, a, b, g)
                assert (fast is None) == (slow is None), (a, b, g)
                if fast is not None:
                    assert abs(fast - slow) < 1e-12, (a, b, g, fast, slow)
                comparisons += 1
    assert comparisons == len(ids) ** 2 * len(GROUPS)
    assert comparisons > 1000, "the fixture is too small to be evidence"


def test_scoring_load_pulls_nothing_but_scoring_columns(tmp_path):
    from music_recommender.database import repository
    from music_recommender.recommendation.space import SPACE_COLUMNS
    from test_feed import _build

    conn, store, space, _ = _build(tmp_path)
    rows = repository.rows_with_columns(conn, list(SPACE_COLUMNS) + ["does_not_exist"])
    assert len(rows) == 12
    assert set(rows[0].keys()) <= set(SPACE_COLUMNS)
    for heavy in ("stat_vector", "mfcc_means", "segment_list", "features_json", "file_path"):
        assert heavy not in rows[0].keys(), f"{heavy} is not needed to score a pair"
    assert repository.rows_with_columns(conn, ["does_not_exist"]) == []


def test_music_space_cache_reuse_and_invalidation(tmp_path):
    from music_recommender.database import repository
    from music_recommender.features import extract
    from music_recommender.recommendation.space import get_music_space
    from test_feed import _build, _tone

    conn, store, space, _ = _build(tmp_path)
    first = get_music_space(conn)
    assert get_music_space(conn) is first, "an unchanged library must be reused"

    wav = tmp_path / "extra.wav"
    _tone(wav, 300.0)
    repository.upsert_track(conn, "extra", str(wav), str(wav), extract.extract(str(wav), 44100, 3.0), "unit")
    second = get_music_space(conn)
    assert second is not first, "a new analysis must not be served from the stale cache"
    assert "extra" in second.rows and "low0" in second.rows


def test_music_space_cache_never_leaks_across_databases(tmp_path):
    from music_recommender.database import repository
    from music_recommender.features import extract
    from music_recommender.recommendation.space import get_music_space
    from test_feed import _build, _tone

    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    conn_a, *_ = _build(tmp_path / "a")
    conn_b, *_ = _build(tmp_path / "b")
    wav = tmp_path / "only_b.wav"
    _tone(wav, 320.0)
    repository.upsert_track(conn_b, "only_b", str(wav), str(wav), extract.extract(str(wav), 44100, 3.0), "unit")

    assert "only_b" not in get_music_space(conn_a).rows
    assert "only_b" in get_music_space(conn_b).rows
    # only one library is held at a time; switching back must rebuild, not reuse b's rows
    again = get_music_space(conn_a)
    assert "only_b" not in again.rows


def test_feed_mmr_matches_naive_pairwise_reference(tmp_path, monkeypatch):
    """Batched cosine must reproduce the per-pair loop, including diversity > 1.0."""
    from music_recommender.database import behavior, repository
    from music_recommender.interest.feed import FeedEngine
    from test_feed import _build

    FIXED = 1_800_000_000.0
    conn, store, space, _ = _build(tmp_path)
    ids = sorted(r["track_id"] for r in repository.all_tracks(conn))
    # arrange embeddings on a circle so some pairs are anti-correlated (cos < 0)
    vecs = {tid: np.array([np.cos(2 * np.pi * i / len(ids)), np.sin(2 * np.pi * i / len(ids))]
                          + [0.0] * (store.dim - 2), dtype="float32") for i, tid in enumerate(ids)}
    for i, tid in enumerate(ids):
        repository.upsert_embedding(conn, tid, vecs[tid], i, "unit")
    store.build([(i, tid, vecs[tid]) for i, tid in enumerate(ids)])
    for k in (0, 1, 2):
        behavior.log_interaction(conn, "u", ids[k], "like", timestamp=FIXED - 60)

    monkeypatch.setattr(time, "time", lambda: FIXED)
    eng = FeedEngine(conn, store, space)
    fast = eng.next("u", limit=8)

    # reference: one np.dot and two np.linalg.norm calls per (candidate, selected) pair
    rw = eng.feed["ranking_weights"]
    interest = eng.model.current("u", now=FIXED)
    recent = behavior.recent_track_ids(conn, "u", int(eng.feed["recent_history_window"]))
    pool = {t: (c + 1) / 2 for t, c in eng.faiss.search(interest["embedding"], 200)
            if not t.startswith("__missing")}
    emb = eng.embeddings

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    rows = []
    for tid, pe in pool.items():
        if tid in set(recent):
            continue
        ma = eng._metadata_affinity(tid, interest["metadata"])
        pref = pe if ma is None else 0.6 * pe + 0.4 * ma
        sim = ma if ma is not None else pe
        nov = 1.0 - max([cos(emb[tid], emb[r]) for r in recent if r in emb], default=0.0)
        rows.append([tid, pref, sim, nov])

    slow, chosen, best_div = [], [], 0.0
    while len(slow) < 8 and rows:
        best, bs, bdiv = None, -1e9, 0.0
        for row in rows:
            tid, p_, s_, n_ = row
            div = 1.0 - max([cos(emb[tid], emb[x]) for x in chosen], default=0.0)
            sc = rw["preference"] * p_ + rw["similarity"] * s_ + rw["novelty"] * n_ + rw["diversity"] * div
            if sc > bs:
                best, bs, bdiv = tid, sc, div
        chosen.append(best)
        best_div = max(best_div, bdiv)
        slow.append({"track_id": best, "score": round(bs, 4)})
        rows = [r for r in rows if r[0] != best]

    assert best_div > 1.0, "fixture must pick an anti-correlated track, else it cannot catch a 0-clamped max"
    assert [(x["track_id"], x["score"]) for x in fast] == [(x["track_id"], x["score"]) for x in slow]


def test_cold_start_farthest_point_matches_naive_reference(tmp_path):
    from music_recommender.interest.feed import FeedEngine
    from test_feed import _build

    conn, store, space, _ = _build(tmp_path)
    eng = FeedEngine(conn, store, space)
    k = 6
    fast = eng._cold_start(k)
    assert len(fast) == k and len(set(fast)) == k

    emb = eng.embeddings

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    ids = list(emb)
    M = np.stack([emb[t] for t in ids])
    centroid = M.mean(axis=0)
    start = ids[int(np.argmin([np.linalg.norm(M[i] - centroid) for i in range(len(ids))]))]
    slow, chosen = [start], [start]
    for _ in range(k - 1):
        farthest = max((t for t in ids if t not in chosen),
                       key=lambda t: -min(cos(emb[t], emb[c]) for c in chosen))
        chosen.append(farthest)
        slow.append(farthest)
    assert fast == slow
