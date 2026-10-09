"""Full-track analysis (docs/MUSIC_ANALYSIS_REFACTOR_PLAN.md stages A/B/C).

Every test here asserts against *hand-built* audio whose structure is known, because the
point of the refactor is that a claim must be backed by evidence:
  * chord templates must actually be 24 different shapes (plan 2.4),
  * the dynamics curve must span the whole file, not the loudest 45 s (plan 2.2),
  * repeating sections must be grouped, while an unsupported song function must stay
    `unknown` instead of being invented from the section's position (plan 2.1).

Run:  .venv/Scripts/python.exe -m pytest tests/test_analysis_v2.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_scan_identity import isolated_norm_dir  # noqa: E402,F401  (pytest fixture, re-used)

SR = 22050


def _section(freqs: list[float], rms: float, seconds: float, onsets: float = 0.0,
             noise: float = 0.0, rng=None) -> np.ndarray:
    """One section: a chord at a given loudness, optionally pulsed and noised.

    `onsets` (pulses per second) is what makes a section rhythmically denser, and `noise`
    makes it spectrally brighter -- the two axes the grouping and labelling heuristics read.
    """
    n = int(SR * seconds)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for f in freqs:
        sig += np.sin(2 * np.pi * f * t)
    if onsets > 0:
        period = int(SR / onsets)
        env = np.ones(n)
        env[::period][: max(1, period // 3)] = 0.2   # gated, so each pulse is an onset
        sig *= env
    if noise > 0:
        sig += (noise / 0.3) * (rng or np.random.default_rng(7)).normal(0, 0.3, n)
    return sig / (np.sqrt(np.mean(sig ** 2)) + 1e-9) * rms


def _write(path: Path, parts: list[np.ndarray]) -> Path:
    y = np.concatenate(parts)
    sf.write(str(path), y.astype("float32"), SR)
    return path


CHORD_C = [261.63, 329.63, 392.0]        # C major triad
CHORD_F = [349.23, 440.0, 523.25]        # F major triad
CHORD_G = [246.94, 293.66, 392.0]        # G major triad


def test_chord_templates_are_twentyfour_distinct_shapes():
    """Plan 2.4: the old profiles were two vectors duplicated 12 times."""
    from music_recommender.features.harmony import template_check, templates

    t = templates()
    assert len(t) == 24, f"expected 24 major/minor triads, got {len(t)}"
    check = template_check()
    assert check["n_distinct"] == 24, check
    # a major and its own minor third differ, so no two templates may be the same shape
    assert check["min_pair_cosine"] < 0.7, check
    # 最相像的一对也只能共用根音与五度(C 与 Cm = 2/3)。旧实现 12 个根音共用一份音阶轮廓,
    # 这个值会是 1.0,所以它才是"模板又塌回去了"的门。
    assert abs(check["max_pair_cosine"] - 2 / 3) < 1e-3, check
    assert check["max_pair"] in (["C", "Cm"], ["Cm", "C"]), check


def test_chord_estimation_recovers_synthetic_triads():
    """Control: a track built from C / F / G major must be read back as those roots."""
    from music_recommender.features.harmony import estimate_chords

    y = np.concatenate([
        _section(CHORD_C, 0.20, 2.0),
        _section(CHORD_F, 0.20, 2.0),
        _section(CHORD_G, 0.20, 2.0),
        _section(CHORD_C, 0.20, 2.0),
    ])
    hm = estimate_chords(y, SR)
    seq = hm["chord_sequence"]
    assert seq.count("C") >= 2, seq
    assert "F" in seq and "G" in seq, seq
    assert all(q == "maj" for q in hm["chord_qualities"]), hm["chord_qualities"]
    # the margin is the honesty field: a triad that beat its runner-up clearly is reportable
    assert hm["chord_margin_mean"] > 0.05, hm["chord_margin_mean"]
    assert len(hm["chord_margins"]) == len(seq)


def test_dynamics_curve_spans_the_whole_track(tmp_path):
    """Plan 2.2: the curve used to describe only the loudest 45 s slice."""
    from music_recommender.features import dynamics

    quiet = _section(CHORD_C, 0.04, 20.0)
    loud = _section(CHORD_C, 0.30, 20.0, onsets=4.0)
    y = np.concatenate([quiet, loud])
    dyn = dynamics.track_dynamics(y, SR, {"analysis": {}})

    times = np.asarray(dyn["time_seconds"], float)
    interval = dyn["meta"]["interval_seconds"]
    # points are bucket centres (meta says so), so the axis starts half a bucket in
    # and never runs past the end of the file
    assert times[0] == round(interval / 2, 3), times[:3]
    assert times[-1] <= 40.0 and times[-1] >= 40.0 - 2 * interval, times[-1]
    assert np.allclose(np.diff(times), interval), "an evenly spaced axis is what a plot needs"
    assert dyn["meta"]["time_semantics"] == "bucket_center"
    assert dyn["meta"]["rms_db_unit"] == "dBFS"
    # every curve is the same length as the time axis, so a client can plot by index
    for key in ("rms_db", "rms_norm", "onset_density", "low_band_energy",
                "spectral_brightness", "arrangement_change"):
        assert len(dyn[key]) == len(times), key
        assert all(np.isfinite(v) for v in dyn[key]), key

    half = len(times) // 2
    second = np.mean(dyn["rms_db"][half:])
    first = np.mean(dyn["rms_db"][:half])
    assert second > first + 6.0, (first, second)
    # the floor keeps digital silence out of the plot; nothing may claim to be +6 dBFS
    assert max(dyn["rms_db"]) <= 0.0
    assert min(dyn["rms_db"]) >= -80.0


def test_dynamics_events_report_the_build_and_the_peak():
    from music_recommender.features import dynamics

    y = np.concatenate([
        _section(CHORD_C, 0.05, 12.0),
        _section(CHORD_C, 0.12, 12.0, onsets=2.0),
        _section(CHORD_C, 0.30, 16.0, onsets=4.0, noise=0.3),
        _section(CHORD_C, 0.06, 10.0),
    ])
    dyn = dynamics.track_dynamics(y, SR, {"analysis": {}})
    kinds = {e["type"] for e in dyn["events"]}
    assert "peak" in kinds, dyn["events"]
    assert kinds & {"build", "drop", "release", "pull-back"}, dyn["events"]
    for e in dyn["events"]:
        assert 0.0 <= e["time"] <= 50.0
        assert 0.0 <= e["confidence"] <= 0.9, e
        assert e["evidence"], e
        assert all(v.startswith(("rms_", "from_", "dynamic", "density", "onset",
                                 "low_band", "spectral", "arrangement", "axis",
                                 "change", "first_quarter", "last_quarter", "drift",
                                 "fall_within"))
                   for v in e["evidence"]), e
    assert dyn["meta"]["calibrated"] is False
    assert dyn["meta"]["confidence_kind"] == "model_score"


def test_flat_track_produces_no_events():
    """A recording with no dynamic range must say so by returning nothing."""
    from music_recommender.features import dynamics

    y = _section(CHORD_C, 0.15, 30.0)
    dyn = dynamics.track_dynamics(y, SR, {"analysis": {}})
    assert dyn["events"] == []


def test_structure_groups_repeats_without_inventing_labels():
    """Hand-labelled A/B/A/B/C/B: sections must cluster by content, not by index parity."""
    from music_recommender.features import structure

    verse = lambda: _section(CHORD_C, 0.08, 8.0, onsets=1.0)          # A
    chorus = lambda: _section(CHORD_C, 0.28, 8.0, onsets=4.0, noise=0.3)   # B
    bridge = lambda: _section(CHORD_G, 0.14, 8.0, onsets=2.0)        # C
    y = np.concatenate([verse(), chorus(), verse(), chorus(), bridge(), chorus()])
    st = structure.structure_features(y, SR, {"analysis": {}})
    segs = st["segments"]
    duration = len(y) / SR

    assert len(segs) >= 4, segs
    assert segs[0]["start"] <= 1.0
    assert abs(segs[-1]["end"] - duration) <= 2.0, (segs[-1]["end"], duration)
    for a, b in zip(segs, segs[1:]):
        assert abs(a["end"] - b["start"]) < 0.01, (a, b)   # contiguous, no gaps

    groups = st["group_sequence"].split(",")
    assert len(groups) == len(segs)
    sizes = {g: groups.count(g) for g in set(groups)}
    assert max(sizes.values()) >= 3, f"the three choruses did not group together: {sizes}"
    assert len(sizes) >= 2, sizes

    # the engine may only ever name these five; pre_chorus/bridge need a model (plan stage D)
    allowed = {"intro", "verse", "chorus", "outro", "unknown"}
    for s in segs:
        assert s["label"] in allowed, s
        assert 0.0 <= s["boundary_confidence"] <= 1.0, s
        assert 0.0 <= s["label_confidence"] <= 0.6, s      # label_max_confidence
        assert "group" in s and s["evidence"], s
        assert "drum_energy" not in s and "vocal_energy" not in s, \
            "window HPSS stems must not be stamped onto full-track segments"
    # the loudest repeated group is the only thing that may be called a chorus
    chorus_segs = [s for s in segs if s["label"] == "chorus"]
    assert len(chorus_segs) >= 3, st["segment_type_sequence"]
    assert {s["group"] for s in chorus_segs} == {chorus_segs[0]["group"]}
    assert st["chorus_repeat_count"] == len(chorus_segs)
    # group membership alone must not hand out the label: the quiet opening shares the
    # choruses' chords, so it clusters with them, and it is still not the hook
    quietest = min(segs, key=lambda s: s["energy_mean"])
    assert quietest["label"] != "chorus", quietest
    assert not any(s["label"] == "bridge" for s in segs), "no rule may claim a bridge"

    ratios = st["duration_ratios"]
    assert abs(sum(ratios) - 1.0) < 0.02, ratios
    assert len(st["energy_curve"]) == 64


def test_repeat_grouping_needs_the_common_mode_removed():
    """Control for the real-audio bug: one key + one instrument set defeats a raw cosine.

    Every descriptor dimension is non-negative, so in one song *any* two sections score ~1.0
    against each other whether they repeat or not. On the user's library a 12-section track
    measured min 0.934 / median 0.982, so `group_similarity_threshold: 0.90` merged the whole
    song into one group and the player said "重复归为 1 组" for songs with three choruses.
    Subtracting each family's mean over the track's own sections keeps the family's scale (a
    per-column z-score would hand a veto to the noisiest dim) and leaves only the direction a
    section deviates in, which is the actual repeat claim. The centred distribution then spans
    -1..+1 and 0.20 was picked where the library's answer is stable (19 tracks: 2-6 groups,
    mean 3.6, every track left with a repeated group of size >= 3).
    """
    from music_recommender.features import structure

    key = np.array([3.0, 2.0, 1.5, 2.5, 1.0, 2.0, 1.2, 2.6, 1.4, 1.0, 1.8, 2.2,
                    0.60, 0.35, 0.50, 0.20])                 # chroma | timbre | density
    # four sections lean one way relative to the song's own average, four the opposite way
    dev = np.array([0.9, -0.7, 0.5, -0.4, 0.8, -0.6, 0.3, -0.9,
                    0.7, -0.5, 0.6, -0.8, 0.4, -0.9, 0.5, -0.3])
    descs = [key + 0.08 * (dev if i % 2 == 0 else -dev) for i in range(8)]
    x = np.stack(descs)
    weights = structure.DEFAULTS["group_feature_weights"]

    raw, _ = structure.family_similarity(x, weights, center=False)
    centered, used = structure.family_similarity(x, weights)
    same = [raw[i, j] for i in range(8) for j in range(i + 1, 8) if i % 2 == j % 2]
    cross = [raw[i, j] for i in range(8) for j in range(i + 1, 8) if i % 2 != j % 2]
    assert min(same) > 0.95 and max(cross) < min(same), (min(same), max(cross))
    assert max(same + cross) - min(same + cross) < 0.05, \
        "the raw arm is supposed to be indiscriminate: every pair sits in one narrow band"
    cs = [centered[i, j] for i in range(8) for j in range(i + 1, 8) if i % 2 == j % 2]
    cc = [centered[i, j] for i in range(8) for j in range(i + 1, 8) if i % 2 != j % 2]
    assert min(cs) - max(cc) > 0.5, (min(cs), max(cc))
    assert abs(used - 1.0) < 1e-9, used

    def grouped(thr: float) -> list[str]:
        segs = [{"descriptor": d} for d in descs]
        structure._assign_groups(segs, {"analysis": {"group_similarity_threshold": thr}})
        return [s["group"] for s in segs]

    def raw_groups(thr: float) -> str:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        dist = np.clip(1.0 - raw, 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        lab = fcluster(linkage(squareform(dist, checks=False), method="average"),
                       t=1.0 - thr, criterion="distance")
        first, out = {}, []
        for l in lab:
            if int(l) not in first:
                first[int(l)] = chr(ord("A") + len(first))
            out.append(first[int(l)])
        return "".join(out)

    assert grouped(structure.DEFAULTS["group_similarity_threshold"]) == list("ABABABAB"), \
        grouped(structure.DEFAULTS["group_similarity_threshold"])
    # control: the old raw cosine, cut anywhere at all, never gives the right answer. At the
    # shipped 0.90 it swallows the whole song (one group, what the player showed); every other
    # cut only fragments it by accident, so no threshold separates A from B.
    def abab(g: list[str] | str) -> bool:
        g = "".join(g)
        return g[0] == g[2] == g[4] == g[6] and g[1] == g[3] == g[5] == g[7] and g[0] != g[1]

    assert raw_groups(0.90) == "AAAAAAAA", raw_groups(0.90)
    grid = [round(t, 3) for t in np.arange(0.0, 1.0, 0.02)]
    assert not any(abab(raw_groups(t)) for t in grid), [raw_groups(t) for t in grid]
    assert abab(grouped(0.20))


def test_beat_alignment_only_when_the_beat_track_is_usable():
    """Beat-sync needs enough beats to place blocks; a short clip uses the time grid."""
    from music_recommender.features import structure

    pulsed = np.concatenate([_section(CHORD_C, 0.2, 8.0, onsets=4.0) for _ in range(6)])
    beat = structure.structure_features(pulsed, SR, {"analysis": {}})
    assert beat["alignment"] == "beat" and beat["tempo_bpm"] > 0, beat["alignment"]

    # 6 s cannot hold the 16 beats the beat-sync path requires (min_beats_for_beat_sync)
    short = _section(CHORD_C, 0.15, 6.0)
    assert structure.structure_features(short, SR, {"analysis": {}})["alignment"] == "time"


def test_extract_stores_the_full_track_payload(tmp_path, isolated_norm_dir):
    from music_recommender.database import repository
    from music_recommender.database.schema import connect, init_db
    from music_recommender.features import extract
    from music_recommender.utils.config import get_config

    parts = [
        _section(CHORD_C, 0.05, 10.0),                             # quiet opening
        _section(CHORD_C, 0.10, 10.0, onsets=2.0),
        _section(CHORD_C, 0.30, 10.0, onsets=4.0, noise=0.3),      # loud, dense middle
        _section(CHORD_G, 0.06, 10.0),                             # quiet ending
    ]
    wav = _write(tmp_path / "song.wav", parts)
    feats = extract.extract(str(wav), SR, 40.0)

    s = feats.scalars
    # full-track loudness sees both tails: a 45 s window percentile could not span 6x
    assert s["rms_p90"] > s["rms_p10"] * 2.0, (s["rms_p10"], s["rms_p90"])
    st = feats.extras["segment_list"]
    assert st and abs(st[-1]["end"] - 40.0) <= 2.5, st[-1]
    # v5 起 extras["dynamics"] 是多模块载荷，dynamics 块在同名键下
    assert feats.extras["dynamics"]["dynamics"]["meta"]["scope"] == "full_track"
    assert feats.extras["dynamics"]["modules"]["texture"] == "ready"
    assert feats.extras["dynamics"]["description"]["meta"]["deterministic"] is True
    assert feats.extras["analysis"]["analysis_version"] == get_config().analysis_version
    assert feats.extras["analysis"]["feature_window_seconds"][1] <= 40.0
    assert "segments" not in feats.extras["analysis"], "meta block must not duplicate columns"
    assert feats.extras["field_meaning"]["drum_energy_ratio"], "proxy fields must say what they are"

    conn = connect(tmp_path / "db.sqlite"); init_db(conn)
    repository.upsert_track(conn, "t1", str(wav), str(wav), feats, "unit")
    repository.insert_segments(conn, "t1", feats.extras["segment_list"])

    payload = repository.get_analysis(conn, "t1", get_config().analysis_version)
    assert payload["stale"] is False
    assert payload["analysis_scope"] == "full_track"
    assert payload["analysis_version"] == get_config().analysis_version
    assert len(payload["segments"]) == len(st)
    assert payload["segments"][0]["group"] == st[0]["group"]
    assert payload["dynamics"]["time_seconds"]
    assert payload["field_meaning"], "the player captions fields from this map"

    # a row produced by an older algorithm is reported, never silently served as current
    old = repository.get_analysis(conn, "t1", get_config().analysis_version + 1)
    assert old["stale"] is True
    assert old["segments"], "stale data must still be readable so the UI can degrade"
    assert repository.get_analysis(conn, "missing", 2) is None

    rows = conn.execute("SELECT segment_type, group_label, label_confidence, boundary_confidence "
                        "FROM track_segments WHERE track_id='t1' ORDER BY segment_index").fetchall()
    assert [r["group_label"] for r in rows] == [s["group"] for s in st]
    assert rows[0]["segment_type"] == st[0]["label"]


def test_rescan_reanalyses_only_when_the_version_changed(tmp_path, isolated_norm_dir, monkeypatch):
    """analysis.version is the knob that invalidates a library after an algorithm change."""
    from music_recommender.database import repository
    from music_recommender.database.schema import connect, init_db
    from music_recommender.library import MusicLibrary

    music = tmp_path / "music"
    music.mkdir()
    _write(music / "s1.wav", [_section(CHORD_C, 0.1, 5.0), _section(CHORD_G, 0.2, 5.0)])
    _write(music / "s2.wav", [_section(CHORD_C, 0.2, 5.0), _section(CHORD_G, 0.1, 5.0)])

    conn = connect(tmp_path / "db.sqlite"); init_db(conn)
    lib = MusicLibrary(conn=conn)

    first = lib.index(str(music), workers=1)
    assert first["indexed"] == 2 and first["reanalysed_stale"] == 0, first

    again = lib.index(str(music), workers=1)
    assert (again["indexed"], again["skipped"]) == (0, 2), again

    # get_config() is an lru_cache singleton, so the bump is undone by monkeypatch
    monkeypatch.setitem(lib.cfg.analysis, "version", 99)
    third = lib.index(str(music), workers=1)
    assert (third["indexed"], third["reanalysed_stale"]) == (2, 2), third
    assert (third["skipped"], third["failed"]) == (0, 0)
    rows = {r["track_id"]: r["analysis_version"] for r in repository.all_tracks(conn)}
    assert set(rows.values()) == {"99"}, rows
    assert lib.index(str(music), workers=1)["skipped"] == 2, "the new version must be stable"


def test_analysis_endpoints_serve_the_timeline(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from music_recommender.api import server
    from music_recommender.database import repository
    from music_recommender.database.schema import connect, init_db
    from music_recommender.features import extract
    from music_recommender.utils.config import get_config

    wav = _write(tmp_path / "ep.wav", [
        _section(CHORD_C, 0.06, 10.0), _section(CHORD_C, 0.28, 10.0, onsets=4.0, noise=0.3),
        _section(CHORD_C, 0.06, 10.0)])
    conn = connect(tmp_path / "db.sqlite"); init_db(conn)
    feats = extract.extract(str(wav), SR, 30.0)
    repository.upsert_track(conn, "t9", str(wav), str(wav), feats, "unit")
    conn.close()

    cfg = SimpleNamespace(db_path=tmp_path / "db.sqlite",
                          analysis_version=get_config().analysis_version)
    monkeypatch.setattr(server, "get_config", lambda: cfg)
    client = TestClient(server.create_app())

    by_id = client.get("/v1/tracks/t9/analysis")
    assert by_id.status_code == 200
    body = by_id.json()
    assert body["stale"] is False and body["segments"] and body["dynamics"]["rms_db"]

    # 播放器（w-music 的 RecommendService::ParseTimeline）就是按这些名字读的。
    # 改了任何一个而不同步改客户端，界面上不会是报错，只会是一张空白时间轴。
    for key in ("track_id", "duration_seconds", "stale", "segments", "dynamics", "group_sequence"):
        assert key in body, key
    assert body["dynamics"]["time_seconds"] and body["dynamics"]["rms_norm"]
    assert len(body["dynamics"]["time_seconds"]) == len(body["dynamics"]["rms_norm"])
    assert body["dynamics"]["meta"]["time_semantics"] == "bucket_center"
    for key in ("start", "end", "label", "group", "boundary_confidence", "label_confidence"):
        assert key in body["segments"][0], key
    assert max(body["dynamics"]["rms_norm"]) <= 1.0 and min(body["dynamics"]["rms_norm"]) >= 0.0

    # x 轴是 duration_seconds：曲线最后一格和任何段落末尾都必须落在里面，
    # 不然播放器会把响度柱画到界外（客户端有兜底，但根源得由引擎保证）。
    times = body["dynamics"]["time_seconds"]
    half = (times[1] - times[0]) * 0.5 if len(times) > 1 else 0.0
    assert times[-1] + half <= body["duration_seconds"] + 1e-6
    assert max(seg["end"] for seg in body["segments"]) <= body["duration_seconds"] + 1e-6

    by_path = client.get("/v1/analysis", params={"file_path": str(wav)})
    assert by_path.status_code == 200
    assert by_path.json()["track_id"] == "t9"

    assert client.get("/v1/tracks/nope/analysis").status_code == 404
    missing = client.get("/v1/analysis", params={"file_path": str(tmp_path / "gone.wav")})
    assert missing.status_code == 404
