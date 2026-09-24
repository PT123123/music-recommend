"""Phase 5 tests: time-decayed interest model + dynamic feed.

Builds an isolated temp library (no shared state), then checks:
  * cold-start feed returns distinct items without history
  * decay maths: a RECENT like of a different family shifts the SHORT-term
    interest away from the LONG-term (dominant) preference (spec 38)
  * feed_next biases toward a liked family
  * recent_history exclusion prevents immediate repeats (spec 65)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

SR = 44100
FAMILIES = {"low": 150.0, "mid": 440.0, "high": 880.0}


def _tone(path, freq, rms=0.12, seconds=3.0):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    sig = 0.6 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    sig = sig / (np.sqrt(np.mean(sig ** 2)) + 1e-9) * rms
    sf.write(str(path), sig.astype("float32"), SR)


def _build(tmp_path):
    from music_recommender.database import repository
    from music_recommender.database.schema import connect, init_db
    from music_recommender.embedding.pca_embedder import PcaStatsEmbedder
    from music_recommender.features import extract
    from music_recommender.recommendation.space import MusicSpace
    from music_recommender.vector_store.faiss_store import FaissStore

    conn = connect(tmp_path / "db.sqlite"); init_db(conn)
    feats, tid_map, ordered = {}, {}, []
    i = 0
    for fam, f in FAMILIES.items():
        for k in range(4):
            wav = tmp_path / f"{fam}{k}.wav"
            _tone(wav, f * (1 + 0.01 * k))
            tf = extract.extract(str(wav), SR, 3.0)
            tid = f"{fam}{k}"
            repository.upsert_track(conn, tid, str(wav), str(wav), tf, "unit")
            feats[tid] = tf.stat_vector
            tid_map[tid] = fam
            ordered.append((i, tid, None))
            i += 1
    keys = list(feats.keys())
    matrix = np.stack([feats[k] for k in keys])
    emb = PcaStatsEmbedder(n_components=4).fit(matrix)
    embs = emb.embed_many(matrix)
    ordered = []
    for idx, tid in enumerate(keys):
        repository.upsert_embedding(conn, tid, embs[idx], idx, "unit")
        ordered.append((idx, tid, embs[idx]))
    store = FaissStore(tmp_path / "faiss", dim=embs.shape[1], model_name="unit")
    store.build(ordered)
    return conn, store, MusicSpace(conn), tid_map


def test_cold_start_diverse(tmp_path):
    from music_recommender.interest.feed import FeedEngine
    conn, store, space, _ = _build(tmp_path)
    picks = FeedEngine(conn, store, space).next("u", limit=5)
    assert len(picks) == 5
    assert len({p["track_id"] for p in picks}) == 5  # distinct -> diverse, no repeats


def test_recent_beats_long_term(tmp_path):
    from music_recommender.database import behavior
    from music_recommender.interest.model import InterestModel
    conn, store, space, _ = _build(tmp_path)
    now = time.time()
    # dominant long history: likes on the HIGH family, 2 days ago
    for k in range(3):
        behavior.log_interaction(conn, "u", f"high{k}", "like", timestamp=now - 2 * 86400)
    # recent shift: a fresh like on the LOW family
    behavior.log_interaction(conn, "u", "low0", "like", timestamp=now - 10)

    summary = InterestModel(conn, space).state_summary("u")
    assert summary["has_history"]
    # high family has higher pitch percentile -> long_term pitch > short_term pitch,
    # i.e. the most-recent low-family like drags short-term toward lower pitch.
    assert summary["long_term"]["pitch"] > summary["short_term"]["pitch"], \
        f"recent preference should move short-term away from dominant: {summary}"


def test_feed_biases_to_liked_family(tmp_path):
    """Spec 38: the feed must move toward the liked REGION, not re-serve the liked
    tracks themselves (spec 65 excludes anything just touched, likes included).

    Differential check: liking family X must make X the top pick, and liking a
    different family must change that top pick. Comparing two users on the same
    library isolates *bias direction* from the MMR diversity term, which otherwise
    (correctly) suppresses the fixture's near-identical same-frequency clones.
    """
    from music_recommender.database import behavior
    from music_recommender.interest.feed import FeedEngine
    conn, store, space, tid_map = _build(tmp_path)
    now = time.time()
    fed = FeedEngine(conn, store, space)

    for user, liked in (("u_high", "high"), ("u_low", "low")):
        for k in (0, 1):
            behavior.log_interaction(conn, user, f"{liked}{k}", "like", timestamp=now - 60)

    top = {}
    for user, liked in (("u_high", "high"), ("u_low", "low")):
        items = fed.next(user, limit=6)
        served = {i["track_id"] for i in items}
        assert not ({"%s0" % liked, "%s1" % liked} & served), "liked tracks are recent history"
        fams = [tid_map.get(t) for t in served]
        assert fams.count(liked) >= 1, f"feed should still reach the liked family: {fams}"
        top[user] = tid_map.get(items[0]["track_id"])

    assert top["u_high"] == "high", f"top pick should follow the liked family, got {top}"
    assert top["u_low"] == "low", f"top pick should follow the liked family, got {top}"


def test_recent_exclusion(tmp_path):
    from music_recommender.database import behavior
    from music_recommender.interest.feed import FeedEngine
    conn, store, space, _ = _build(tmp_path)
    now = time.time()
    for k in range(3):
        behavior.log_interaction(conn, "u", f"mid{k}", "play", timestamp=now - 30)
    fed = FeedEngine(conn, store, space)
    batch = fed.next("u", limit=6)
    ids = {i["track_id"] for i in batch}
    assert not (ids & {"mid0", "mid1", "mid2"}), "just-played tracks must be excluded (spec 65)"
