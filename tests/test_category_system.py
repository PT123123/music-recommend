"""Category system tests: lexicon, presets, discovery, small-library honesty, free text.

No audio decoding: the library is written straight into SQLite with synthetic feature
rows in three well-separated families, one per vocal/instrumental shape, so every
assertion is about scoring/discovery/parsing rather than about librosa. Values for
*every* lexicon dimension are populated, which is what makes "a preset category must
be answerable" a real assertion. Scale choices are indicative, not measured; only the
bpm range is load-bearing because a preset hard filter bounds it.

Control assertions (an intentionally broken case that must be caught) are included per
project convention.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

# column -> (low, high) plausible range, so the synthetic library looks like audio
RANGES = {
    "rms_mean": (0.02, 0.6), "rms_std": (0.0, 0.3), "dynamic_range": (2.0, 30.0),
    "crest_factor": (2.0, 20.0),
    "spectral_centroid_mean": (400.0, 7000.0), "spectral_bandwidth_mean": (300.0, 5000.0),
    "spectral_flatness_mean": (0.0, 0.4), "spectral_contrast_mean": (10.0, 40.0),
    "spectral_flux_mean": (0.01, 2.0), "zcr_mean": (0.01, 0.3),
    "bpm": (55.0, 180.0), "onset_density": (0.2, 4.0), "danceability": (0.0, 1.0),
    "beat_consistency": (0.0, 1.0), "tempo_variance": (0.0, 40.0),
    "chord_change_rate": (0.0, 4.0), "harmonic_rhythm": (0.0, 4.0),
    "dissonance_mean": (0.0, 0.6),
    "pitch_mean": (100.0, 700.0), "pitch_range": (5.0, 40.0), "pitch_variance": (10.0, 600.0),
    "high_pitch_ratio": (0.0, 0.6), "low_pitch_ratio": (0.0, 0.7), "mid_pitch_ratio": (0.0, 0.8),
    "vocal_ratio": (0.0, 1.0), "vocal_intensity": (0.0, 1.0),
    "vocal_pitch_mean": (45.0, 80.0), "vocal_pitch_range": (1.0, 24.0),
    "vocal_pitch_variance": (0.0, 16.0),
    "bass_energy_ratio": (0.0, 0.8), "drum_energy_ratio": (0.0, 0.8),
    "chorus_energy": (0.0, 1.0), "chorus_repeat_count": (0.0, 12.0),
}

# family -> (percentile level, vocal presence, extra overrides)
FAMILIES = {
    "loud_instrumental": (0.90, 0),   # loud, bright, fast, dense
    "quiet_deep_vocal": (0.12, 1),    # soft, dark, slow, low voice
    "soaring_vocal": (0.55, 1),       # middle energy, very high and wide voice
}
VOCAL_DIMS = ("vocal_pitch_mean", "vocal_pitch_range", "vocal_pitch_variance")
PER_FAMILY = 10
DISCOVERY_SETTINGS = {"min_cluster_size": 5, "k_min": 3, "k_max": 6, "min_cluster_share": 0.0}


def _scalars(fam: str, level: int, has_vocal: int, rng) -> dict:
    from music_recommender.recommendation.lexicon import LEXICON
    scalars: dict[str, float] = {}
    for spec in LEXICON:
        lo, hi = RANGES.get(spec.col, (0.0, 1.0))
        if spec.col in VOCAL_DIMS and not has_vocal:
            continue  # a real library leaves these NULL for instrumentals
        if spec.col in ("vocal_ratio", "vocal_intensity") and not has_vocal:
            scalars[spec.col] = lo + (hi - lo) * 0.02
            continue
        jitter = 0.05 * rng.uniform(-1, 1)
        value = lo + (hi - lo) * float(np.clip(level + jitter, 0.0, 1.0))
        scalars[spec.col] = round(value) if spec.col == "chorus_repeat_count" else value
    scalars["has_vocal"] = has_vocal
    scalars["duration"] = 180.0 + level * 20
    return scalars


@pytest.fixture()
def library(tmp_path):
    from music_recommender.database import repository
    from music_recommender.database.schema import connect, init_db
    from music_recommender.features.extract import TrackFeatures

    conn = connect(tmp_path / "db.sqlite")
    init_db(conn)
    rng = np.random.default_rng(7)
    tid_of: dict[str, list[str]] = {}
    embeddings: dict[str, np.ndarray] = {}
    for fam, (level, has_vocal) in FAMILIES.items():
        tid_of[fam] = []
        for k in range(PER_FAMILY):
            tid = f"{fam}-{k}"
            feats = TrackFeatures(scalars=_scalars(fam, level, has_vocal, rng))
            feats.estimated_fields = ["chord", "vocal_presence"]
            repository.upsert_track(conn, tid, f"/music/{tid}.wav", f"/norm/{tid}.wav", feats, "unit")
            tid_of[fam].append(tid)
            center = np.eye(4, dtype="float32")[list(FAMILIES).index(fam)]
            vec = (center + 0.004 * rng.normal(size=4)).astype("float32")
            embeddings[tid] = vec / np.linalg.norm(vec)
    for i, (tid, vec) in enumerate(embeddings.items()):
        repository.upsert_embedding(conn, tid, vec, i, "unit")
    conn.commit()
    yield conn, tid_of, tmp_path
    conn.close()


def _cfg(tmp_path, category_system=None):
    from music_recommender.utils.config import Config, load_categories
    root = Path(str(tmp_path or "data"))
    return Config(audio={"normalized_dir": str(root / "norm")},
                  paths={"db_file": str(root / "db.sqlite"), "faiss_dir": str(root / "faiss")},
                  embedding={"model_store": str(root / "e.joblib")},
                  categories=load_categories(),
                  category_system=category_system if category_system is not None else
                  {"discovery": DISCOVERY_SETTINGS})


def _recommender(conn, tmp_path=None, category_system=None):
    from music_recommender.recommender import Recommender
    return Recommender(cfg=_cfg(tmp_path, category_system), conn=conn)


def _family(tid: str) -> str:
    return tid.rsplit("-", 1)[0]


# ---- lexicon / config integrity ----
def test_every_preset_category_field_is_known(library):
    """A preset category must be answerable; a typo must be impossible to ship."""
    from music_recommender.database import repository
    from music_recommender.recommendation.lexicon import DIM_MAP, LEXICON
    from music_recommender.recommendation.space import SPACE_COLUMNS
    conn, _, _ = library
    rec = _recommender(conn)
    columns = set(repository.track_columns(conn))
    for spec in LEXICON:
        assert spec.col in columns, f"lexicon names a column that is not in the schema: {spec.col}"
        assert spec.col in SPACE_COLUMNS, f"the space never reads {spec.col}, so it cannot rank it"
        assert spec.high and spec.low, f"{spec.col} needs both poles"
    for cat in rec.cfg.categories:
        unknown = [f for f in (cat.get("query") or {}) if f not in DIM_MAP]
        assert not unknown, f"{cat['id']} uses fields with no measured dimension: {unknown}"
        unresolved = [f for f in (cat.get("query") or {})
                      if DIM_MAP[f] not in rec._core.space.sorted_vals]
        assert not unresolved, f"{cat['id']} targets dimensions the library cannot rank: {unresolved}"
    # control: a bogus field must be reported, never silently dropped
    items, meta = rec.category_query({"definitely_not_a_dimension": 0.5})
    assert items == [] and meta["ignored"] == ["definitely_not_a_dimension"]


def test_lexicon_words_are_unambiguous():
    from music_recommender.recommendation.lexicon import LEXICON, TEXT_TERMS
    all_words = [w for d in LEXICON for w in (*d.high, *d.low)]
    assert len(TEXT_TERMS) == len(all_words), "one word mapped to two dimensions: text parsing is ambiguous"


# ---- hard filters ----
def test_instrumental_hard_filter_really_filters(library):
    conn, tid_of, _ = library
    rec = _recommender(conn)
    items, meta = rec.category("instrumental", limit=50)
    got = {i["track_id"] for i in items}
    vocal_tids = {t for t, r in rec._core.space.rows.items() if r.get("has_vocal")}
    assert vocal_tids, "fixture must contain vocal tracks"
    assert got == set(tid_of["loud_instrumental"]), got
    assert not (got & vocal_tids)
    assert meta["unknown_filters"] == []
    assert meta["filters_applied"] == ["instrumental"]
    assert meta["filter_estimated"] == ["instrumental"], \
        "has_vocal is a thresholded guess, so the answer must not look like a tag"
    # control: asking for voices must give the complement
    voiced, _ = rec.category("vocal_power", limit=50)
    assert {i["track_id"] for i in voiced} & set(tid_of["loud_instrumental"]) == set()


def test_unknown_hard_filter_is_reported_not_applied(library):
    conn, _, _ = library
    rec = _recommender(conn)
    items, meta = rec.category_query({"energy": 0.5}, hard_filter={"made_up_col_min": 3})
    assert meta["unknown_filters"] == ["made_up_col_min"]
    assert len(items) == len(rec._core.space.rows), "an unknown filter must not remove anything"


def test_numeric_bound_filters(library):
    conn, tid_of, _ = library
    rec = _recommender(conn)
    items, _ = rec.category("slow_ballad", limit=50)
    rows = rec._core.space.rows
    assert items and max(rows[i["track_id"]]["bpm"] for i in items) <= 85.0001
    assert _family(items[0]["track_id"]) == "quiet_deep_vocal"
    # control: raising the floor must switch families
    fast, m_fast = rec.category_query({"energy": 0.5}, hard_filter={"bpm_min": 150})
    assert fast and all(rows[i["track_id"]]["bpm"] >= 150 for i in fast)
    assert {_family(i["track_id"]) for i in fast} == {"loud_instrumental"}
    assert m_fast["filters_applied"] == ["bpm_min"]
    assert m_fast["filter_estimated"] == [], "a measured bound is not a guess and must not be caveated"


def test_tag_filter_with_no_tagged_tracks_returns_nothing(library):
    conn, _, _ = library
    rec = _recommender(conn)
    items, meta = rec.category_query({"energy": 0.5}, hard_filter={"language_is": "en"}, limit=50)
    assert items == []
    assert meta["tag_missing"] == {"language": 3 * PER_FAMILY}, "un-tagged tracks must be counted"


# ---- support / small-library honesty ----
def test_support_flag_tracks_evidence_density(library):
    from music_recommender.recommendation.category import CategoryConfig, CategoryEngine
    conn, _, _ = library
    space = _recommender(conn)._core.space
    wide = CategoryEngine(space, CategoryConfig(support_max_distance=0.4, support_min_count=5,
                                                support_min_share=0.0))
    _, meta_wide = wide.recommend({"query": {"energy": 0.5}}, limit=5)
    tight = CategoryEngine(space, CategoryConfig(support_max_distance=0.02, support_min_count=5,
                                                 support_min_share=0.0))
    _, meta_tight = tight.recommend({"query": {"energy": 0.5}}, limit=5)
    assert meta_wide["support"] > meta_tight["support"]
    assert meta_tight["low_support"] is True and meta_wide["low_support"] is False
    assert meta_wide["candidate_pool"] == len(space.rows) == 3 * PER_FAMILY


def test_dedupe_breaks_family_repetition(library):
    from music_recommender.recommendation.category import CategoryConfig, CategoryEngine
    conn, _, _ = library
    space = _recommender(conn)._core.space
    plain = CategoryEngine(space, CategoryConfig(dedupe=False))
    plain_items, _ = plain.recommend({"query": {"energy": 0.9}}, limit=10)
    spread = CategoryEngine(space, CategoryConfig(dedupe=True, dedupe_lambda=0.2,
                                                  dedupe_max_similarity=0.9))
    spread_items, meta = spread.recommend({"query": {"energy": 0.9}}, limit=10)
    fams_plain = [_family(r.track_id) for r in plain_items]
    fams_spread = [_family(r.track_id) for r in spread_items]
    assert len(set(fams_plain)) < len(set(fams_spread)), \
        f"dedupe changed nothing: {fams_plain} vs {fams_spread}"
    assert meta["deduped"] is True


def test_dedupe_similarity_is_an_angle_not_a_magnitude(library):
    """Stored embeddings are not unit length, so a raw dot product would let a *quiet*
    duplicate slip through the cap."""
    from music_recommender.recommendation.category import CategoryConfig, CategoryEngine
    conn, tid_of, _ = library
    space = _recommender(conn)._core.space
    unit = {"loud_instrumental": [1.0, 0.0], "quiet_deep_vocal": [0.0, 1.0],
            "soaring_vocal": [0.7, 0.7]}
    # every track of a family gets the SAME direction at 0.4 length: identical songs,
    # but the inner product is 0.16, far below any cosine cap
    space._vectors = {t: np.asarray(unit[_family(t)]) * 0.4 for t in space.rows}
    # arm identity: these vectors must NOT be unit length, otherwise the case below is
    # indistinguishable from a raw dot product and the test would prove nothing
    assert max(float(np.linalg.norm(v)) for v in space._vectors.values()) < 0.5
    eng = CategoryEngine(space, CategoryConfig(dedupe=True, dedupe_lambda=0.0,
                                               dedupe_max_similarity=0.97))
    items, meta = eng.recommend({"query": {"energy": 0.9}}, limit=3)
    fams = [_family(r.track_id) for r in items]
    assert len(set(fams)) == 3, f"one per family expected, got {fams}"
    assert meta["duplicates_suppressed"] == len(space.rows) - 3, meta
    # control: a raw dot product between these vectors would be 0.16 -- under every cap
    # -- and would have returned three copies of one family. Tightening the cap must
    # therefore keep, not weaken, the spread.
    eng2 = CategoryEngine(space, CategoryConfig(dedupe=True, dedupe_lambda=0.0,
                                                dedupe_max_similarity=0.999))
    items2, _ = eng2.recommend({"query": {"energy": 0.9}}, limit=3)
    assert len({_family(r.track_id) for r in items2}) == 3, items2


def test_dedupe_is_skipped_without_embeddings(tmp_path, library):
    from music_recommender.recommendation.category import CategoryConfig, CategoryEngine
    from music_recommender.recommendation.space import MusicSpace
    conn, _, _ = library
    space = MusicSpace(conn)
    space._vectors = {}  # a library analysed before embeddings existed
    items, meta = CategoryEngine(space, CategoryConfig(dedupe=True)).recommend(
        {"query": {"energy": 0.9}}, limit=10)
    assert len(items) == 10
    assert meta["deduped"] is False, "reporting dedupe that never ran would be a lie"


# ---- discovery ----
def test_discovery_recovers_the_three_families(library):
    from music_recommender.recommendation.discovery import DiscoveryConfig, discover
    from music_recommender.recommendation.space import MusicSpace
    conn, _, _ = library
    space = MusicSpace(conn)
    cfg = DiscoveryConfig(**DISCOVERY_SETTINGS)
    specs, meta = discover(space, cfg)
    assert meta["status"] == "ok", meta
    assert len(specs) >= 3, f"expected at least one category per family, got {[s['name'] for s in specs]}"
    for s in specs:
        members = s["member_track_ids"]
        assert len(members) >= cfg.min_cluster_size
        dominant = max(FAMILIES, key=lambda f: sum(m.startswith(f + "-") for m in members))
        purity = sum(m.startswith(dominant + "-") for m in members) / len(members)
        assert purity >= 0.8, f"{s['name']} mixes families: {members}"
        assert s["name"] and s["query"], s
    # determinism: the same library must yield the same clusters and the same name
    specs2, meta2 = discover(space, cfg)
    assert [s["member_track_ids"] for s in specs] == [s["member_track_ids"] for s in specs2]
    assert [s["name"] for s in specs] == [s["name"] for s in specs2]
    assert meta["silhouette"] == meta2["silhouette"]


def test_discovery_refuses_to_invent_categories_on_a_tiny_library(tmp_path):
    from music_recommender.database import repository
    from music_recommender.database.schema import connect, init_db
    from music_recommender.features.extract import TrackFeatures
    from music_recommender.recommendation.discovery import DiscoveryConfig, discover
    from music_recommender.recommendation.space import MusicSpace
    conn = connect(tmp_path / "tiny.sqlite")
    init_db(conn)
    for i in range(6):
        repository.upsert_track(conn, f"t{i}", f"/m/{i}.wav", f"/n/{i}.wav",
                               TrackFeatures(scalars={"rms_mean": 0.1 * i, "bpm": 60 + 10 * i,
                                                      "spectral_centroid_mean": 1000 + 500 * i}),
                               "unit")
    specs, meta = discover(MusicSpace(conn), DiscoveryConfig(min_cluster_size=4))
    assert specs == []
    assert meta["status"] in {"library-too-small", "no-k-satisfied-min-cluster-size"}, meta
    conn.close()


def test_discovered_category_is_queryable_end_to_end(library):
    conn, _, tmp = library
    rec = _recommender(conn, tmp)
    auto = [c for c in rec.categories() if c["source"] == "discovered"]
    assert auto, "no discovered categories in the listing"
    specs, _ = rec.discovered()
    for target in auto:
        spec = next(s for s in specs if s["id"] == target["id"])
        items, meta = rec.category(target["id"], limit=50)
        assert items, f"{target['name']} returned nothing"
        assert {i["track_id"] for i in items} <= set(spec["member_track_ids"]), \
            "a discovered category must only return its own cluster members"
        assert meta["candidate_pool"] == len(spec["member_track_ids"])
    # control: an id from the other library must 404
    with pytest.raises(KeyError):
        rec.category("auto-99")


def test_instrumental_only_cluster_is_named_and_filtered(library):
    conn, tid_of, tmp = library
    rec = _recommender(conn, tmp)
    specs, _ = rec.discovered()
    instrumental = [s for s in specs if s["hard_filter"].get("instrumental")]
    assert instrumental, "the all-instrumental family should yield a 无人声 cluster"
    for s in instrumental:
        assert "无人声" in s["name"]
        assert set(s["member_track_ids"]) <= set(tid_of["loud_instrumental"])


# ---- tag-derived dimensions (language / genre) ----
def test_language_without_tags_is_reported_missing_not_guessed(library):
    conn, _, tmp = library
    items, meta = _recommender(conn, tmp).category_text("来点英文歌", limit=50)
    assert items == []
    assert {u["reason"] for u in meta["unmatched"]} == {"no-language-tag-in-library"}
    assert meta["matched"] == [], "nothing in this request is answerable from audio"


def test_language_tag_backfill_invalidates_the_space_and_filters(library):
    from music_recommender.database import repository
    conn, tid_of, tmp = library
    rec = _recommender(conn, tmp)
    version_before = repository.library_version(conn)
    for tid in tid_of["quiet_deep_vocal"]:
        repository.update_tags(conn, f"/music/{tid}.wav",
                               {"title": tid, "language": "en", "meta_source": "tag"})
    assert repository.library_version(conn) != version_before, \
        "a tag write must invalidate the cached Music Space"
    rec2 = _recommender(conn, tmp)
    items, meta = rec2.category_text("来点英文歌", limit=50)
    assert {i["track_id"] for i in items} == set(tid_of["quiet_deep_vocal"])
    assert meta["used_dims"] == [], "language is a filter, not a scored dimension"
    assert meta["filter_only"] is True
    assert meta["tag_missing"] == {"language": 2 * PER_FAMILY}, "the rest must be reported as un-tagged"
    assert rec2.category_text("来点中文歌", limit=50)[0] == []


def test_rescanning_unchanged_tags_does_not_invalidate_the_space(library):
    from music_recommender.database import repository
    conn, tid_of, _ = library
    tags = {"title": tid_of["quiet_deep_vocal"][0], "language": "en", "meta_source": "tag"}
    repository.update_tags(conn, "/music/quiet_deep_vocal-0.wav", tags)
    before = repository.library_version(conn)
    repository.update_tags(conn, "/music/quiet_deep_vocal-0.wav", tags)
    assert repository.library_version(conn) == before, "writing identical tags must not churn the cache"


def test_genre_tag_wins_over_acoustic_proxy(library):
    from music_recommender.database import repository
    conn, tid_of, tmp = library
    for tid in tid_of["soaring_vocal"][:6]:
        repository.update_tags(conn, f"/music/{tid}.wav", {"title": tid, "genre": "Pop",
                                                           "meta_source": "tag"})
    items, meta = _recommender(conn, tmp).category_text("来点流行", limit=50)
    assert meta["genre_proxies"] == [], "a real tag must be used instead of a proxy"
    assert {i["track_id"] for i in items} == set(tid_of["soaring_vocal"][:6])
    assert any(m["side"] == "tag" for m in meta["matched"]), meta["matched"]


def test_genre_without_tags_falls_back_to_a_flagged_acoustic_proxy(library):
    conn, _, tmp = library
    items, meta = _recommender(conn, tmp).category_text("想说唱", limit=50)
    assert meta["genre_proxies"], "说唱 has no tag here, so an explicitly-flagged proxy is expected"
    assert meta["genre_proxies"][0]["genre"] == "hip"
    assert "vocal_pitch_range" in meta["genre_proxies"][0]["dims"]
    assert items, "the proxy must still be able to rank something"
    assert all(m["side"] == "proxy" for m in meta["matched"]), meta["matched"]


# ---- free-text parsing ----
def test_text_query_maps_vocabulary_and_negation(library):
    conn, _, tmp = library
    rec = _recommender(conn, tmp)
    items, meta = rec.category_text("来点安静一点的", limit=50)
    assert [m["dimension"] for m in meta["matched"]] == ["rms_mean"]
    assert meta["matched"][0]["side"] == "low"
    assert items and _family(items[0]["track_id"]) == "quiet_deep_vocal"
    _, meta_neg = rec.category_text("不要安静的", limit=50)
    assert meta_neg["matched"][0]["side"] == "high", "negation must flip the target percentile"


def test_text_query_reports_unanswerable_words(library):
    conn, _, tmp = library
    _, meta = _recommender(conn, tmp).category_text("来点很治愈的高级感", limit=50)
    terms = {u["term"] for u in meta["unmatched"]}
    assert {"治愈", "高级感"} <= terms, meta["unmatched"]
    assert meta["query"] == {}, "unanswerable words must not be mapped onto unrelated dimensions"
    # a genre word the lexicon does know still gets its documented acoustic proxy
    _, m2 = _recommender(conn, tmp).category_text("来点很治愈的说唱", limit=50)
    assert {"治愈"} <= {u["term"] for u in m2["unmatched"]}, m2["unmatched"]
    assert m2["genre_proxies"], "说唱 must be answered by proxy when no genre tag exists"


def test_text_query_handles_the_users_own_example_words(library):
    """The words the user actually asks with must each land on a measured dimension."""
    conn, _, tmp = library
    rec = _recommender(conn, tmp)
    _, meta = rec.category_text("高音、女声、低沉声、纯音乐、节奏感强", limit=50)
    matched = {m["term"] for m in meta["matched"]}
    assert "纯音乐" in matched, "纯音乐 must set the instrumental filter"
    assert meta["hard_filter"] == {"instrumental": True}
    assert "高音" in matched, matched
    _, m2 = rec.category_text("节奏密集的")
    assert m2["matched"][0]["dimension"] == "onset_density"
    _, m5 = rec.category_text("节奏感强")
    assert m5["matched"][0]["dimension"] == "onset_density", "the user's own wording must be understood"
    assert m5["unmatched"] == [], m5["unmatched"]
    _, m6 = rec.category_text("女声")
    assert m6["vocal_proxies"], "女声 has no measurable gender, so it must be an explicit proxy"
    assert m6["matched"][0]["side"] == "proxy" and "不识别性别" in m6["matched"][0]["word"]
    _, m3 = rec.category_text("低音重一点的")
    assert m3["matched"][0]["dimension"] == "low_pitch_ratio"
    _, m4 = rec.category_text("重低音")
    assert m4["matched"][0]["dimension"] == "bass_energy_ratio"


def test_a_request_can_ask_for_dimensions_the_pool_does_not_have(library):
    """女声 + 纯音乐 cannot both be answered: say so instead of ranking nothing at 1.0."""
    conn, _, tmp = library
    rec = _recommender(conn, tmp)
    items, meta = rec.category_text("女生纯音乐", limit=50)
    assert items == [], "the instrumental pool has no voice-pitch data to rank on"
    assert meta["filter_estimated"] == ["instrumental"]
    assert set(meta["unscored_dims"]) == {"vocal_pitch_mean", "vocal_pitch_range"}, meta
    assert meta["filter_only"] is True, "nothing scored, so this is not a ranking"
    # control: a request whose dimensions do exist in the pool scores normally
    inst, m2 = rec.category("instrumental", limit=50)
    assert inst and m2["unscored_dims"] == [] and m2["filter_only"] is False
    # control: the same proxy without the contradiction does rank
    voiced, m3 = rec.category_text("女声", limit=50)
    assert m3["unscored_dims"] == [] and m3["filter_only"] is False
    assert len({i["score"] for i in voiced}) > 1
