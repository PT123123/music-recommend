"""Tests for scan identity, embedded tags, incremental scan and the DSP dedup.

These guard the four real-library fixes: content-stable track_id, mutagen tag
reading, skip-unchanged re-scan, and single-computation pyin/hpss (plus the
sequence budget that removed the 266-second song->song query).

Run:  .venv/Scripts/python.exe -m pytest tests/test_scan_identity.py -q
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

SR = 44100


def _tone(path: Path, freq: float = 220.0, seconds: float = 3.0, amp: float = 0.1) -> Path:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    sig = np.sin(2 * np.pi * freq * t) * amp
    sf.write(str(path), sig.astype("float32"), SR)
    return path


@pytest.fixture
def isolated_norm_dir(tmp_path, monkeypatch):
    """Point the normalized-WAV cache at tmp_path so tests never touch data/."""
    from music_recommender.utils.config import get_config
    cfg = get_config()
    monkeypatch.setitem(cfg.audio, "normalized_dir", str(tmp_path / "norm"))
    return cfg


# ---- track identity ----

def test_track_id_survives_rename_and_copy(tmp_path):
    from music_recommender.preprocess.audio import track_id_for

    a = _tone(tmp_path / "original.wav")
    b = tmp_path / "sub" / "完全换个名字.wav"
    b.parent.mkdir()
    b.write_bytes(a.read_bytes())
    assert track_id_for(a) == track_id_for(b), "moving a file must not change its id"

    c = _tone(tmp_path / "different.wav", freq=440.0)
    assert track_id_for(a) != track_id_for(c), "different content must get a different id"


def test_track_id_stable_when_path_unavailable(tmp_path):
    from music_recommender.preprocess.audio import track_id_for

    missing = tmp_path / "gone.wav"
    assert track_id_for(missing) == track_id_for(missing), "fallback must still be deterministic"


# ---- embedded tags ----

def _tagged_mp3(path: Path) -> Path | None:
    """Write a tagged mp3 at `path`, generated from a scratch wav kept outside its folder."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    src = path.parent.parent / (path.stem + "_src.wav")
    _tone(src)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-i", str(src), "-metadata", "title=Ordinary Song",
                    "-metadata", "artist=Test Artist", "-metadata", "album=Test Album",
                    "-metadata", "date=2021", str(path)], check=True, capture_output=True)
    return path


def test_read_tags_from_real_mp3(tmp_path):
    mp3 = _tagged_mp3(tmp_path / "tagged.mp3")
    if mp3 is None:
        pytest.skip("ffmpeg not available to create a tagged mp3")
    from music_recommender.preprocess.metadata import read_tags

    t = read_tags(mp3)
    assert t["title"] == "Ordinary Song"
    assert t["artist"] == "Test Artist"
    assert t["album"] == "Test Album"
    assert t["year"] == 2021
    assert t["meta_source"] == "tag"


def test_read_tags_absent_is_none_not_guessed(tmp_path):
    from music_recommender.preprocess.metadata import read_tags

    t = read_tags(_tone(tmp_path / "untagged.wav"))
    # No tags -> every field stays None with meta_source='none'. A filename is never
    # parsed into a title: tag data must be the file's own claim, not our inference.
    assert t == {"title": None, "artist": None, "album": None, "year": None,
                 "language": None, "meta_source": "none"}


def test_tag_fields_are_stored_and_marked(tmp_path, isolated_norm_dir):
    from music_recommender.database import repository
    from music_recommender.database.schema import connect, init_db
    from music_recommender.library import MusicLibrary

    music = tmp_path / "music"
    mp3 = _tagged_mp3(music / "lib.mp3")
    if mp3 is None:
        pytest.skip("ffmpeg not available to create a tagged mp3")
    conn = connect(tmp_path / "db.sqlite")
    init_db(conn)
    lib = MusicLibrary(conn=conn)
    assert lib.index(str(music), workers=1)["indexed"] == 1

    row = conn.execute("SELECT * FROM tracks").fetchone()
    assert row["title"] == "Ordinary Song" and row["artist"] == "Test Artist"
    assert row["meta_source"] == "tag", "tag provenance must be queryable"
    assert row["file_size"] == mp3.stat().st_size
    assert row["file_mtime"] == pytest.approx(mp3.stat().st_mtime)
    # language is NOT inferred from audio or tags here; absent stays absent
    assert row["language"] is None

    names = repository.display_map(conn, [row["track_id"]])
    assert names[row["track_id"]]["title"] == "Ordinary Song"
    assert names[row["track_id"]]["meta_source"] == "tag"


def test_update_tags_only_touches_tag_columns(tmp_path):
    from music_recommender.database import repository
    from music_recommender.database.schema import connect, init_db
    from music_recommender.features import extract
    from music_recommender.preprocess.metadata import read_tags

    conn = connect(tmp_path / "db.sqlite")
    init_db(conn)
    wav = _tone(tmp_path / "a.wav")
    feats = extract.extract(str(wav), SR, 3.0)
    repository.upsert_track(conn, "t1", str(wav), str(wav), feats, "unit")
    assert conn.execute("SELECT meta_source FROM tracks").fetchone()[0] == "none"

    before = conn.execute("SELECT stat_vector, analyzed_at FROM tracks").fetchone()
    repository.update_tags(conn, str(wav), {**read_tags(wav), "title": "Later Tag",
                                            "meta_source": "tag"})
    after = conn.execute("SELECT stat_vector, analyzed_at, title, meta_source FROM tracks").fetchone()
    assert after["stat_vector"] == before["stat_vector"]
    assert after["analyzed_at"] == before["analyzed_at"], "tag backfill must not re-analyse"
    assert after["title"] == "Later Tag" and after["meta_source"] == "tag"


# ---- incremental scan ----

def test_unchanged_files_are_skipped_on_rescan(tmp_path, isolated_norm_dir):
    from music_recommender.database.schema import connect, init_db
    from music_recommender.library import MusicLibrary

    music = tmp_path / "music"
    music.mkdir()
    for i, f in enumerate((300.0, 500.0)):
        _tone(music / f"t{i}.wav", freq=f, seconds=2.0)

    conn = connect(tmp_path / "db.sqlite")
    init_db(conn)
    lib = MusicLibrary(conn=conn)

    first = lib.index(str(music), workers=1)
    assert (first["indexed"], first["skipped"]) == (2, 0)

    second = lib.index(str(music), workers=1)
    assert (second["indexed"], second["skipped"]) == (0, 2), "re-scan must not re-decode"

    # a new file is picked up without disturbing the untouched one
    _tone(music / "t2.wav", freq=700.0, seconds=2.0)
    third = lib.index(str(music), workers=1)
    assert (third["indexed"], third["skipped"]) == (1, 2)

    # touching a file marks it changed
    import os
    p = music / "t0.wav"
    os.utime(p, (p.stat().st_atime + 100, p.stat().st_mtime + 100))
    fourth = lib.index(str(music), workers=1)
    assert (fourth["indexed"], fourth["skipped"]) == (1, 2)

    forced = lib.index(str(music), workers=1, force=True)
    assert (forced["indexed"], forced.get("skipped")) == (3, 0)


def test_scan_state_only_counts_completed_rows(tmp_path):
    from music_recommender.database import repository
    from music_recommender.database.schema import connect, init_db

    conn = connect(tmp_path / "db.sqlite")
    init_db(conn)
    conn.execute("INSERT INTO tracks (track_id, file_path, file_size, file_mtime) VALUES ('a','/a',10,1.0)")
    conn.execute("INSERT INTO tracks (track_id, file_path, file_size, file_mtime, stat_vector) "
                 "VALUES ('b','/b',20,2.0, x'0000')")
    conn.commit()
    # a row without features (crashed mid-pipeline) must be re-analysed, not skipped
    assert repository.scan_state(conn) == {"/b": (20, 2.0)}


# ---- extraction cost / dedup ----

def test_pyin_and_hpss_run_once_per_track(tmp_path, monkeypatch):
    import librosa

    calls = {"pyin": 0, "hpss": 0}
    real_pyin, real_hpss = librosa.pyin, librosa.effects.hpss

    def spy_pyin(*a, **k):
        calls["pyin"] += 1
        return real_pyin(*a, **k)

    def spy_hpss(*a, **k):
        calls["hpss"] += 1
        return real_hpss(*a, **k)

    monkeypatch.setattr(librosa, "pyin", spy_pyin)
    monkeypatch.setattr(librosa.effects, "hpss", spy_hpss)

    from music_recommender.features import extract
    extract.extract(str(_tone(tmp_path / "a.wav")), SR, 3.0)

    assert calls["pyin"] == 1, f"pyin must run once, ran {calls['pyin']}"
    assert calls["hpss"] == 1, f"hpss must run once, ran {calls['hpss']}"


def test_melody_sequence_respects_budget():
    from music_recommender.features.melody import cap_sequence, melody_features
    from music_recommender.utils.config import get_config

    assert cap_sequence(list(range(50)), 120) == list(range(50))
    capped = cap_sequence(list(range(1000)), 120)
    assert len(capped) == 120
    assert capped[0] == 0 and capped[-1] == 999, "shape endpoints must survive subsampling"

    # glissando: every frame changes pitch, so an uncapped sequence would be thousands long
    t = np.linspace(0, 4.0, int(SR * 4), endpoint=False)
    freq = 200.0 * (2 ** (3.0 * t / 4.0))  # sweep up 3 octaves
    sig = np.sin(2 * np.pi * np.cumsum(freq) / SR).astype("float32") * 0.2
    out = melody_features(sig, SR, max_seconds=4.0)
    budget = int(get_config().features.get("max_sequence_tokens", 120))
    assert len(out["relative_pitch_sequence"]) <= budget
    assert len(out["interval_sequence"]) <= budget
    # the histogram is the full population, the sequence only the capped view
    assert sum(out["interval_histogram"].values()) >= len(out["relative_pitch_sequence"])


def test_note_events_ignore_pitch_jitter():
    from music_recommender.features.melody import _note_events

    # pyin's reported error on a held note is under a semitone. Held inside one
    # 2-semitone bucket that noise must produce no note changes at all, while the
    # same signal quantised to 1 semitone manufactures dozens of them.
    for seed in range(4):
        rng = np.random.default_rng(seed)
        held = 60.0 + rng.uniform(-0.9, 0.9, 400)
        fine = _note_events(held, quantise=1, smooth=5)
        coarse = _note_events(held, quantise=2, smooth=5)
        assert len(fine) > 20, "1-semitone buckets turn tracking noise into fake note changes"
        assert coarse == [60.0], "the configured bucket width must absorb the same noise"

    # a real leap must survive it
    motion = np.concatenate([np.full(60, 60.0), np.full(60, 66.0)])
    assert _note_events(motion, quantise=2, smooth=5) == [60.0, 66.0]
    assert len(_note_events(np.full(400, 60.0))) == 1
