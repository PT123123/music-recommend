"""SQLite schema (spec section 54). SQLite is the source of truth (principle 4).

The MVP populates a subset of columns; the rest exist so later phases do not
need a table rewrite. Arrays are stored as JSON text or BLOB.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    track_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    normalized_path TEXT,
    title TEXT,
    artist TEXT,
    album TEXT,
    year INTEGER,
    duration REAL,

    language TEXT,
    genre TEXT,

    rms_mean REAL, rms_std REAL, rms_p10 REAL, rms_p25 REAL,
    rms_p50 REAL, rms_p75 REAL, rms_p90 REAL, rms_max REAL,
    dynamic_range REAL, crest_factor REAL,

    zcr_mean REAL,
    spectral_centroid_mean REAL, spectral_bandwidth_mean REAL,
    spectral_rolloff_mean REAL, spectral_flux_mean REAL,
    spectral_contrast_mean REAL, spectral_flatness_mean REAL,

    mfcc_means BLOB, mfcc_stds BLOB, chroma_mean BLOB, stat_vector BLOB,

    bpm REAL, beat_consistency REAL, tempo_variance REAL,
    danceability REAL, onset_density REAL,
    rhythmic_complexity REAL, syncopation REAL,

    key TEXT, mode TEXT,
    dissonance_mean REAL, chord_sequence TEXT, relative_chord_sequence TEXT,
    chorus_chord_sequence TEXT, chord_change_rate REAL, harmonic_rhythm REAL,

    pitch_min REAL, pitch_max REAL, pitch_mean REAL, pitch_range REAL,
    pitch_variance REAL, low_pitch_ratio REAL, mid_pitch_ratio REAL,
    high_pitch_ratio REAL, interval_histogram TEXT, melody_contour_type TEXT,
    relative_pitch_sequence TEXT, interval_sequence TEXT,

    segment_list TEXT, segment_type_sequence TEXT, chorus_repeat_count INTEGER,
    chorus_energy REAL, energy_curve TEXT,

    has_vocal INTEGER, vocal_ratio REAL, vocal_gender TEXT,
    vocal_pitch_mean REAL, vocal_pitch_range REAL, vocal_pitch_variance REAL,
    vocal_intensity REAL,

    instruments TEXT, bass_energy_ratio REAL, drum_energy_ratio REAL,

    valence REAL, arousal REAL, scene_tags TEXT, fine_genre_tags TEXT,

    features_json TEXT,          -- non-column extras: chord arrays, confidences, energy-band ratios
    estimate_flags TEXT,         -- comma list of which feature groups are estimated (not ground truth)

    embedding_id INTEGER,
    feature_version TEXT,
    embedding_model TEXT,
    analyzed_at TEXT,

    -- File-level facts used for incremental re-scan (skip unchanged files).
    file_size INTEGER,
    file_mtime REAL,

    -- Where title/artist/album/year came from: 'tag' (read from the file's own
    -- embedded metadata), 'none', or null for a legacy row. Kept separate from
    -- estimate_flags because tag data is *not* audio analysis: it is reported
    -- verbatim from the file and must never be presented as MIR-derived.
    meta_source TEXT,

    -- Bumped only by tag writes. `language` / `genre` are read by category hard
    -- filters and by discovered-cluster naming, so a tag-only update must still
    -- invalidate the cached Music Space; `analyzed_at` deliberately does not move
    -- when no audio was re-analysed.
    meta_updated_at TEXT

);

-- PCA/nn embeddings live here so FAISS can be deleted and rebuilt without
-- losing music data (spec principle 4). FAISS is only a cache/index.
CREATE TABLE IF NOT EXISTS track_embeddings (
    track_id TEXT PRIMARY KEY,
    embedding_id INTEGER,
    vector BLOB NOT NULL,
    embedding_model TEXT,
    FOREIGN KEY(track_id) REFERENCES tracks(track_id)
);

-- Per-segment stats so future logic can retrieve/compare just the chorus without
-- touching the main table (spec section 55).
CREATE TABLE IF NOT EXISTS track_segments (
    segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id TEXT NOT NULL,
    segment_index INTEGER NOT NULL,
    segment_type TEXT,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    energy_mean REAL,
    vocal_energy REAL,
    bass_energy REAL,
    drum_energy REAL,
    spectral_centroid REAL,
    FOREIGN KEY(track_id) REFERENCES tracks(track_id)
);
CREATE INDEX IF NOT EXISTS idx_segments_track ON track_segments(track_id);

-- User behaviour events feeding the time-decayed interest model (spec 56).
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    track_id TEXT NOT NULL,
    event TEXT NOT NULL,
    timestamp REAL NOT NULL,
    play_seconds REAL,
    completion_ratio REAL,
    FOREIGN KEY(track_id) REFERENCES tracks(track_id)
);
CREATE INDEX IF NOT EXISTS idx_interactions_user_ts ON interactions(user_id, timestamp);

-- Materialised short/medium/long interest snapshots for explainability (spec 57).
CREATE TABLE IF NOT EXISTS interest_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    scope TEXT NOT NULL,          -- short | medium | long
    vector BLOB,                  -- embedding-space interest vector
    metadata_json TEXT            -- readable dim summary
);

-- Human evaluation labels for future weight/model tuning (spec 80).
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seed_track_id TEXT NOT NULL,
    recommended_track_id TEXT NOT NULL,
    label INTEGER NOT NULL,       -- 1 = good/recommended, 0 = not
    note TEXT,
    created_at TEXT
);
"""


# Columns added after the initial release. CREATE TABLE IF NOT EXISTS will not
# alter an existing table, so pre-existing databases get patched here.
_MIGRATIONS: dict[str, str] = {
    "file_size": "INTEGER",
    "file_mtime": "REAL",
    "year": "INTEGER",
    "meta_source": "TEXT",
    "genre": "TEXT",
    "meta_updated_at": "TEXT",
}


def _migrate(conn: sqlite3.Connection) -> None:
    have = {r["name"] for r in conn.execute("PRAGMA table_info(tracks)")}
    for col, decl in _MIGRATIONS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE tracks ADD COLUMN {col} {decl}")
    conn.commit()


def connect(db_file: Path) -> sqlite3.Connection:
    Path(db_file).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)  # idempotent: ensures all tables exist for any consumer
    _migrate(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
