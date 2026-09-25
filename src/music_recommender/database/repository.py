"""Repository: read/write tracks and embeddings in SQLite (source of truth)."""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from ..features.extract import TrackFeatures


def _file_stat(file_path: str) -> tuple[Optional[int], Optional[float]]:
    """(size, mtime) for incremental-scan bookkeeping; None if the file is gone."""
    try:
        st = Path(file_path).stat()
        return int(st.st_size), float(st.st_mtime)
    except OSError:
        return None, None


def _blob(a: Optional[np.ndarray]) -> Optional[bytes]:
    return None if a is None else np.asarray(a, dtype="float32").tobytes()


def _json(obj) -> Optional[str]:
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False)


def upsert_track(conn: sqlite3.Connection, track_id: str, file_path: str,
                 normalized_path: str, feats: TrackFeatures, embedding_model: str,
                 tags: Optional[dict] = None) -> None:
    s = feats.scalars
    tags = tags or {}
    size, mtime = _file_stat(file_path)
    row = {
        "track_id": track_id,
        "file_path": str(file_path),
        "normalized_path": str(normalized_path),
        "title": tags.get("title"),
        "artist": tags.get("artist"),
        "album": tags.get("album"),
        "year": tags.get("year"),
        "language": tags.get("language"),
        # provenance of the four fields above: 'tag' = the file's own claim, never MIR output
        "meta_source": tags.get("meta_source", "none"),
        "file_size": size,
        "file_mtime": mtime,
        "duration": s.get("duration"),
        "rms_mean": s.get("rms_mean"), "rms_std": s.get("rms_std"),
        "rms_p10": s.get("rms_p10"), "rms_p25": s.get("rms_p25"),
        "rms_p50": s.get("rms_p50"), "rms_p75": s.get("rms_p75"),
        "rms_p90": s.get("rms_p90"), "rms_max": s.get("rms_max"),
        "dynamic_range": s.get("dynamic_range"), "crest_factor": s.get("crest_factor"),
        "zcr_mean": s.get("zcr_mean"),
        "spectral_centroid_mean": s.get("spectral_centroid_mean"),
        "spectral_bandwidth_mean": s.get("spectral_bandwidth_mean"),
        "spectral_rolloff_mean": s.get("spectral_rolloff_mean"),
        "spectral_flux_mean": s.get("spectral_flux_mean"),
        "spectral_contrast_mean": s.get("spectral_contrast_mean"),
        "spectral_flatness_mean": s.get("spectral_flatness_mean"),
        "mfcc_means": _blob(feats.mfcc_means),
        "mfcc_stds": _blob(feats.mfcc_stds),
        "chroma_mean": _blob(feats.chroma_mean),
        "stat_vector": _blob(feats.stat_vector),
        "bpm": s.get("bpm"), "beat_consistency": s.get("beat_consistency"),
        "tempo_variance": s.get("tempo_variance"), "danceability": s.get("danceability"),
        "onset_density": s.get("onset_density"),
        "key": s.get("key"), "mode": s.get("mode"),
        # --- Phase 2 fields ---
        "chord_sequence": s.get("chord_sequence"),
        "relative_chord_sequence": s.get("relative_chord_sequence"),
        "chord_change_rate": s.get("chord_change_rate"),
        "harmonic_rhythm": s.get("harmonic_rhythm"),
        "dissonance_mean": s.get("dissonance_mean"),
        "pitch_min": s.get("pitch_min"), "pitch_max": s.get("pitch_max"),
        "pitch_mean": s.get("pitch_mean"), "pitch_range": s.get("pitch_range"),
        "pitch_variance": s.get("pitch_variance"),
        "low_pitch_ratio": s.get("low_pitch_ratio"), "mid_pitch_ratio": s.get("mid_pitch_ratio"),
        "high_pitch_ratio": s.get("high_pitch_ratio"),
        "melody_contour_type": s.get("melody_contour_type"),
        "interval_histogram": _json(s.get("interval_histogram")),
        "relative_pitch_sequence": _json(s.get("relative_pitch_sequence")),
        "interval_sequence": _json(s.get("interval_sequence")),
        "bass_energy_ratio": s.get("bass_energy_ratio"), "drum_energy_ratio": s.get("drum_energy_ratio"),
        "has_vocal": s.get("has_vocal"), "vocal_ratio": s.get("vocal_ratio"),
        "vocal_gender": s.get("vocal_gender"),
        "vocal_pitch_mean": s.get("vocal_pitch_mean"), "vocal_pitch_range": s.get("vocal_pitch_range"),
        "vocal_pitch_variance": s.get("vocal_pitch_variance"), "vocal_intensity": s.get("vocal_intensity"),
        "features_json": _json(feats.extras),
        # --- Phase 3 structure fields ---
        "segment_list": _json(feats.extras.get("segment_list")),
        "energy_curve": _json(feats.extras.get("energy_curve")),
        "segment_type_sequence": s.get("segment_type_sequence"),
        "chorus_repeat_count": s.get("chorus_repeat_count"),
        "chorus_energy": s.get("chorus_energy"),
        "chorus_chord_sequence": s.get("chorus_chord_sequence"),
        "estimate_flags": ",".join(feats.estimated_fields) if feats.estimated_fields else None,
        "embedding_model": embedding_model,
        "analyzed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    cols = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row.keys())
    updates = ", ".join(f"{k}=excluded.{k}" for k in row.keys() if k != "track_id")
    conn.execute(
        f"INSERT INTO tracks ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT(track_id) DO UPDATE SET {updates}",
        row,
    )
    conn.commit()


def get_track(conn: sqlite3.Connection, track_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM tracks WHERE track_id=?", (track_id,)).fetchone()


def insert_segments(conn: sqlite3.Connection, track_id: str, segments: list[dict]) -> None:
    """Persist per-segment stats into track_segments (replacing prior rows for idempotency)."""
    conn.execute("DELETE FROM track_segments WHERE track_id=?", (track_id,))
    for seg in segments or []:
        conn.execute(
            "INSERT INTO track_segments (track_id, segment_index, segment_type, start_time, end_time, "
            "energy_mean, vocal_energy, bass_energy, drum_energy, spectral_centroid) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (track_id, seg.get("index"), seg.get("type"), seg.get("start", 0.0), seg.get("end", 0.0),
             seg.get("energy_mean"), seg.get("vocal_energy"), seg.get("bass_energy"),
             seg.get("drum_energy"), seg.get("spectral_centroid")),
        )
    conn.commit()


def get_track_by_path(conn: sqlite3.Connection, file_path: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM tracks WHERE file_path=?", (str(file_path),)).fetchone()


def scan_state(conn: sqlite3.Connection) -> dict[str, tuple[int, float]]:
    """{file_path: (size, mtime)} for rows that hold a complete analysis.

    Only rows with a stored stat_vector count as analysed: a track written by an
    earlier crash mid-pipeline must be re-done, not skipped forever.
    """
    rows = conn.execute(
        "SELECT file_path, file_size, file_mtime FROM tracks "
        "WHERE stat_vector IS NOT NULL AND file_size IS NOT NULL AND file_mtime IS NOT NULL"
    ).fetchall()
    return {r["file_path"]: (int(r["file_size"]), float(r["file_mtime"])) for r in rows}


def paths_without_tags(conn: sqlite3.Connection) -> set[str]:
    """Files whose tag columns were never filled from the file itself."""
    rows = conn.execute(
        "SELECT file_path FROM tracks WHERE stat_vector IS NOT NULL "
        "AND (meta_source IS NULL OR meta_source='none')"
    ).fetchall()
    return {r["file_path"] for r in rows}


def update_tags(conn: sqlite3.Connection, file_path: str, tags: dict) -> None:
    """Write only the tag columns for an already-analysed file (no re-analysis)."""
    conn.execute(
        "UPDATE tracks SET title=:title, artist=:artist, album=:album, year=:year, "
        "language=:language, meta_source=:meta_source WHERE file_path=:file_path",
        {"file_path": str(file_path),
         "title": tags.get("title"), "artist": tags.get("artist"),
         "album": tags.get("album"), "year": tags.get("year"),
         "language": tags.get("language"), "meta_source": tags.get("meta_source", "none")},
    )
    conn.commit()


def display_map(conn: sqlite3.Connection, track_ids) -> dict[str, dict]:
    """{track_id: {title, artist, album, file_path, meta_source}} in one query.

    A player renders a recommendation list without N+1 requests, and `meta_source`
    travels with every row so the UI can never present a tag as analysed audio.
    """
    ids = [t for t in dict.fromkeys(track_ids) if t]
    if not ids:
        return {}
    q = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT track_id, title, artist, album, file_path, meta_source FROM tracks WHERE track_id IN ({q})",
        ids,
    ).fetchall()
    return {r["track_id"]: dict(r) for r in rows}


def all_tracks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM tracks ORDER BY track_id").fetchall()


def count_tracks(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS c FROM tracks").fetchone()["c"]


# ---- embeddings ----

def upsert_embedding(conn: sqlite3.Connection, track_id: str, vector: np.ndarray,
                     embedding_id: int, embedding_model: str) -> None:
    conn.execute(
        "INSERT INTO track_embeddings (track_id, embedding_id, vector, embedding_model) "
        "VALUES (?, ?, ?, ?) ON CONFLICT(track_id) DO UPDATE SET "
        "embedding_id=excluded.embedding_id, vector=excluded.vector, embedding_model=excluded.embedding_model",
        (track_id, embedding_id, _blob(vector), embedding_model),
    )
    conn.execute("UPDATE tracks SET embedding_id=? WHERE track_id=?", (embedding_id, track_id))
    conn.commit()


def ordered_embeddings(conn: sqlite3.Connection) -> list[tuple[int, str, np.ndarray]]:
    """Return [(embedding_id, track_id, vector)] ordered by embedding_id for FAISS build."""
    rows = conn.execute(
        "SELECT embedding_id, track_id, vector FROM track_embeddings ORDER BY embedding_id"
    ).fetchall()
    out = []
    dim = None
    for r in rows:
        v = np.frombuffer(r["vector"], dtype="float32")
        if dim is None:
            dim = v.shape[0]
        elif v.shape[0] != dim:
            raise ValueError("embedding dimension mismatch in track_embeddings table")
        out.append((int(r["embedding_id"]), r["track_id"], v))
    return out
