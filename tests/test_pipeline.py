"""Pytest suite for Phase 1 MVP. Uses isolated temp SQLite + temp FAISS dir so it
never touches the real data/ directory.

Run:  .venv/Scripts/python.exe -m pytest tests/test_pipeline.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

SR = 44100


def _write_tone(path: Path, freq: float, rms: float, seconds: float, noise: float = 0.0) -> Path:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    sig = np.sin(2 * np.pi * freq * t)
    if noise > 0:
        sig += noise * np.random.default_rng(0).normal(0, 1, len(t))
    sig = sig / (np.sqrt(np.mean(sig ** 2)) + 1e-9) * rms
    sf.write(str(path), sig.astype("float32"), SR)
    return path


def test_feature_extraction_shapes(tmp_path):
    from music_recommender.features import extract
    wav = _write_tone(tmp_path / "a.wav", 220.0, 0.1, 3.0)
    feats = extract.extract(str(wav), SR, 3.0)
    assert feats.stat_vector is not None and feats.stat_vector.ndim == 1
    assert feats.mfcc_means.shape[0] == 13
    assert feats.chroma_mean.shape[0] == 12
    for key in ("rms_mean", "spectral_centroid_mean", "bpm", "onset_density", "key", "mode"):
        assert key in feats.scalars and feats.scalars[key] is not None


def test_no_fabricated_fields(tmp_path):
    """MVP must not fake unavailable fields (spec principle 1)."""
    from music_recommender.features import extract
    wav = _write_tone(tmp_path / "b.wav", 330.0, 0.1, 2.0)
    feats = extract.extract(str(wav), SR, 2.0)
    # language / genre probabilities / named-instrument presence are NOT derivable -> absent
    assert "language" not in feats.scalars
    assert "fine_genre_tags" not in feats.scalars
    # gender must stay 'unknown' rather than a fabricated male/female
    assert feats.scalars.get("vocal_gender") == "unknown"
    # estimated feature groups must be flagged
    assert feats.estimated_fields, "estimated fields must be marked"


def test_pca_embedder(tmp_path):
    from music_recommender.embedding.pca_embedder import PcaStatsEmbedder
    X = np.random.default_rng(1).normal(size=(10, 30)).astype("float32")
    emb = PcaStatsEmbedder(n_components=5).fit(X)
    out = emb.embed_many(X)
    assert out.shape[0] == 10 and out.shape[1] <= 5
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)  # L2 normalized for cosine/inner-product
    emb.save(tmp_path / "e.joblib")
    loaded = PcaStatsEmbedder.load(tmp_path / "e.joblib")
    assert np.allclose(loaded.embed(X[0]), out[0], atol=1e-5)


def test_faiss_roundtrip(tmp_path):
    from music_recommender.vector_store.faiss_store import FaissStore
    dim = 6
    ordered = [(i, f"t{i}", np.eye(dim, dtype="float32")[i]) for i in range(dim)]
    store = FaissStore(tmp_path, dim=dim, model_name="unit")
    store.build(ordered)
    store.save()
    loaded = FaissStore.load(tmp_path)
    hits = loaded.search(ordered[0][2], 3)
    assert hits and hits[0][0] == "t0"  # exact self-match first (cosine=1)
    assert (tmp_path / "track_ids.json").exists()
    assert (tmp_path / "embedding_dimension.txt").read_text() == str(dim)


def test_end_to_end_recommendation(tmp_path):
    """Same-frequency tones should be nearest neighbors (control assertion)."""
    from music_recommender.database import repository
    from music_recommender.database.schema import connect, init_db
    from music_recommender.embedding.pca_embedder import PcaStatsEmbedder
    from music_recommender.features import extract
    from music_recommender.recommendation.core import Recommender
    from music_recommender.vector_store.faiss_store import FaissStore

    conn = connect(tmp_path / "db.sqlite"); init_db(conn)
    families = {"low": 150.0, "mid": 440.0, "high": 880.0}
    feats_by_tid, tids_by_family = {}, {}
    for fam, f in families.items():
        for k in range(3):
            wav = _write_tone(tmp_path / f"{fam}{k}.wav", f * (1 + 0.01 * k), 0.12, 3.0)
            tid = f"{fam}{k}"
            tf = extract.extract(str(wav), SR, 3.0)
            repository.upsert_track(conn, tid, str(wav), str(wav), tf, "unit")
            feats_by_tid[tid] = tf.stat_vector
            tids_by_family.setdefault(fam, []).append(tid)

    matrix = np.stack(list(feats_by_tid.values()))
    emb = PcaStatsEmbedder(n_components=4).fit(matrix)
    embs = emb.embed_many(matrix)
    ordered = []
    for i, tid in enumerate(feats_by_tid.keys()):
        repository.upsert_embedding(conn, tid, embs[i], i, "unit")
        ordered.append((i, tid, embs[i]))
    store = FaissStore(tmp_path / "faiss", dim=embs.shape[1], model_name="unit")
    store.build(ordered)

    rec = Recommender(conn, store)
    correct = 0
    total = 0
    for fam, tids in tids_by_family.items():
        for seed in tids:
            top = rec.similar(seed, limit=2)
            assert top, f"empty recs for {seed}"
            total += 1
            if top[0].track_id.startswith(fam):
                correct += 1
    assert total == 9
    assert correct / total >= 0.8, f"only {correct}/{total} top-1 matched own family"


def test_sequence_similarity_key_invariant():
    from music_recommender.features.harmony import relative_sequence
    from music_recommender.recommendation.sequence_sim import combined_sequence_similarity
    # C-G-Am-F and D-A-Bm-G are the same progression transposed -> identical relative view.
    cgamf_roots = [0, 7, 9, 5]      # C G A F pitch classes
    dabmg_roots = [2, 9, 11, 7]     # D A B G pitch classes
    rel1 = relative_sequence(cgamf_roots, ["maj", "maj", "min", "maj"], "C", "major")
    rel2 = relative_sequence(dabmg_roots, ["maj", "maj", "min", "maj"], "D", "major")
    assert rel1 == rel2
    assert combined_sequence_similarity(rel1, rel2) > 0.99
    # a different progression must score lower
    other = ["0:maj", "2:maj", "4:maj", "5:maj"]
    assert combined_sequence_similarity(rel1, other) < 0.9


def test_categories_from_config():
    from music_recommender.recommender import Recommender
    cats = Recommender().categories()
    assert any(c["id"] == "high_female_vocal" for c in cats)


def test_evaluation_record_and_export(tmp_path):
    import csv
    from music_recommender.database import behavior
    from music_recommender.database.schema import connect, init_db
    conn = connect(tmp_path / "db.sqlite"); init_db(conn)
    assert behavior.record_evaluation(conn, "seedA", "recB", 1) >= 1
    behavior.record_evaluation(conn, "seedA", "recC", 0, note="not similar")
    out = tmp_path / "eval.csv"
    n = behavior.export_evaluations_csv(conn, out)
    assert n == 2
    rows = list(csv.reader(out.open(encoding="utf-8")))
    assert rows[0] == ["seed", "recommendation", "label"]
    assert {"seedA", "recB", "1"} <= set(rows[1])
    stats = behavior.evaluation_stats(conn)
    assert stats["count"] == 2 and stats["positive"] == 1
