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


# extras that already own a dedicated column: duplicating them inside features_json doubled
# the row size (the v2 dynamics curves are a few thousand numbers) for no reader's benefit.
# `analysis` is NOT on this list -- extract keeps it to a small meta block so the honesty
# fields (scope, alignment, label_source, window) travel with the row and stay readable
# through the existing /v1/tracks/{id} response.
_COLUMN_HELD_EXTRAS = ("segment_list", "energy_curve", "dynamics")


def _features_json(extras: dict) -> Optional[str]:
    return _json({k: v for k, v in (extras or {}).items() if k not in _COLUMN_HELD_EXTRAS})


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
        "genre": tags.get("genre"),
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
        "features_json": _features_json(feats.extras),
        # --- Phase 3 structure fields ---
        "segment_list": _json(feats.extras.get("segment_list")),
        "energy_curve": _json(feats.extras.get("energy_curve")),
        "segment_type_sequence": s.get("segment_type_sequence"),
        "chorus_repeat_count": s.get("chorus_repeat_count"),
        "chorus_energy": s.get("chorus_energy"),
        "chorus_chord_sequence": s.get("chorus_chord_sequence"),
        # --- full-track analysis (v2) ---
        "analysis_version": str(feats.extras.get("analysis", {}).get("analysis_version") or ""),
        "group_sequence": feats.extras.get("group_sequence"),
        "duration_ratios": _json(feats.extras.get("duration_ratios")),
        "dynamics_json": _json(feats.extras.get("dynamics")),
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
            "energy_mean, vocal_energy, bass_energy, drum_energy, spectral_centroid, "
            "group_label, label_confidence, boundary_confidence) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (track_id, seg.get("index"), seg.get("label") or seg.get("type"),
             seg.get("start", 0.0), seg.get("end", 0.0),
             seg.get("energy_mean"), seg.get("vocal_energy"), seg.get("bass_energy"),
             seg.get("drum_energy"), seg.get("spectral_centroid"),
             seg.get("group"), seg.get("label_confidence"), seg.get("boundary_confidence")),
        )
    conn.commit()


def get_analysis(conn: sqlite3.Connection, track_id: str,
                 current_version: int | str | None = None) -> Optional[dict]:
    """Reassemble the v2 analysis payload from its columns, and say whether it is current.

    `stale` is computed here rather than trusted from the row: a v1 database simply has no
    `analysis_version`, and a player must be able to tell "not analysed yet" apart from
    "analysed by an older algorithm" apart from "current".
    """
    row = conn.execute(
        "SELECT track_id, duration, segment_list, energy_curve, dynamics_json, features_json, "
        "analysis_version, group_sequence, duration_ratios, segment_type_sequence "
        "FROM tracks WHERE track_id=?", (track_id,)).fetchone()
    if row is None:
        return None
    stored = (row["analysis_version"] or "").strip()
    extra = _loads(row["features_json"], {})
    dynamics_raw = _loads(row["dynamics_json"], {})
    # dynamics_json 有两种格式：v4 及更早是单层 dynamics dict；v5 起是
    # {"dynamics": ..., "texture": ..., "spatial": ..., "rhythm_timeline": ...,
    #  "profiles": ..., "description": ..., "modules": {...}}。旧格式照常服务。
    if isinstance(dynamics_raw.get("dynamics"), dict):
        dyn = dynamics_raw["dynamics"]
        texture = dynamics_raw.get("texture", {})
        spatial = dynamics_raw.get("spatial", {})
        rhythm_tl = dynamics_raw.get("rhythm_timeline", {})
        profiles = dynamics_raw.get("profiles", {})
        desc = dynamics_raw.get("description", {})
        modules = dynamics_raw.get("modules", {})
    else:
        dyn = dynamics_raw
        texture, spatial, rhythm_tl, profiles, desc, modules = {}, {}, {}, {}, {}, {}
    version = int(stored) if stored.isdigit() else None
    return {
        "track_id": row["track_id"],
        "duration_seconds": row["duration"],
        "analysis_version": version,
        "current_version": int(current_version) if current_version is not None else None,
        "stale": bool(current_version is not None and stored != str(current_version)),
        "analysis_scope": "full_track" if stored.isdigit() else "window",
        "segments": _loads(row["segment_list"], []),
        "energy_curve": _loads(row["energy_curve"], []),
        "dynamics": dyn,
        # 播放器用的通用曲线契约：每条带 id / 单位 / 含义 / 有效掩码，缺失的模块
        # 直接不在列表里（缺失不是零），旧客户端继续读 dynamics 旧字段。
        "curves": _curves_from_modules(dyn, {"texture": texture, "spatial": spatial,
                                             "rhythm_timeline": rhythm_tl}, current_version),
        "modules": modules,
        "texture": texture,
        "spatial": spatial,
        "rhythm_timeline": rhythm_tl,
        "profiles": profiles,
        "description": desc,
        "group_sequence": row["group_sequence"],
        "duration_ratios": _loads(row["duration_ratios"], []),
        "segment_type_sequence": row["segment_type_sequence"],
        # what each proxy field really measures, so a player can caption the curve honestly
        "analysis_meta": extra.get("analysis", {}),
        "field_meaning": extra.get("field_meaning", {}),
    }


# (曲线 id, 单位, 含义, dynamics 里的键)。一条曲线缺失 = 引擎没算过这一项，
# 不能默认成零：零有可能是真实测量结果。
_ANALYSIS_CURVES: list[tuple[str, str, str, str]] = [
    ("level_db", "dBFS", "短时电平 RMS；不是感知响度 LUFS，也不是声压级", "rms_db"),
    ("level_norm", "0..1（全曲内归一化）", "归一化电平，用于同曲内比较", "rms_norm"),
    ("onset_activity", "activity_proxy", "onset 包络积分代理量（旧约定包络），不是起音次数", "onset_density"),
    ("onset_rate_hz", "events/s", "实际检测到的起音事件数 / 桶真实时长", "onset_rate_hz"),
    ("low_band_ratio", "ratio 0..1", "20-250 Hz 功率占比；不等于 bass 乐器", "low_band_ratio"),
    ("spectral_brightness", "Hz", "功率加权频率质心", "spectral_brightness"),
    ("spectral_flux", "normalized 0..1", "谱流：帧间正变化占比；只说明音色在变", "spectral_flux"),
]

# P1 模块曲线：(模块, 曲线键, 曲线 id, 单位, 含义)。
_MODULE_CURVES: list[tuple[str, str, str, str, str]] = [
    ("texture", "spectral_flatness", "spectral_flatness", "0..1",
     "频谱平坦度：偏噪声性↔偏音调性；高不等于用了白噪声合成器"),
    ("texture", "spectral_bandwidth", "spectral_bandwidth", "Hz",
     "功率加权谱带宽；频率维度指标，与立体声宽度无关"),
    ("texture", "hf_power_ratio", "hf_power_ratio", "ratio 0..1", "4kHz 以上功率占比"),
    ("texture", "flux_low", "flux_low", "normalized 0..1", "低频段（20-250Hz）谱流"),
    ("texture", "flux_mid", "flux_mid", "normalized 0..1", "中频段（250-2kHz）谱流"),
    ("texture", "flux_high", "flux_high", "normalized 0..1", "高频段（2-8kHz）谱流"),
    ("spatial", "lr_balance_db", "lr_balance_db", "dB", "左右电平差；偏一边不等于更宽"),
    ("spatial", "lr_correlation", "lr_correlation", "-1..1", "左右相关度；接近 +1 常是双单声道内容"),
    ("spatial", "side_energy_ratio", "side_energy_ratio", "0..1", "Side 能量占比；必须与相关度一起读"),
    ("rhythm_timeline", "local_tempo", "local_tempo", "BPM", "窗口局部速度；起音变密不等于加速"),
]


def _curves_from_modules(dyn: dict, modules: dict, current_version: int | str | None) -> list[dict]:
    curves: list[dict] = []
    times = dyn.get("time_seconds") or []
    version = f"analysis-v{current_version}" if current_version is not None else "analysis-unknown"
    if isinstance(times, list) and times:
        for cid, unit, meaning, key in _ANALYSIS_CURVES:
            vals = dyn.get(key)
            if not isinstance(vals, list) or len(vals) != len(times):
                continue
            curves.append({
                "id": cid,
                "unit": unit,
                "meaning": meaning,
                "scope": "full_track",
                "time_seconds": times,
                "values": vals,
                "valid": [True] * len(vals),
                "source": "mono_decode",
                "method_version": version,
            })
    # 模块曲线：各模块可以有自己的时间网格（契约允许），valid 来自模块自己
    # 的有效掩码（spatial 的 None 桶 = 电平低于计量底噪，不是"完美居中"）。
    for mod_name, key, cid, unit, meaning in _MODULE_CURVES:
        mod = modules.get(mod_name) or {}
        if mod.get("status") not in (None, "ready"):
            continue
        if mod_name == "rhythm_timeline":
            grid = mod.get("local_tempo") or {}
            vals = grid.get("values")
            mtimes = grid.get("time_seconds") or []
        elif mod_name == "spatial":
            grid = mod.get("curves") or {}
            vals = grid.get(key)
            mtimes = mod.get("time_seconds") or []
        else:
            vals = mod.get(key)
            mtimes = mod.get("time_seconds") or []
        if not isinstance(mtimes, list) or not mtimes:
            continue
        if not isinstance(vals, list) or len(vals) != len(mtimes):
            continue
        mask = mod.get("valid")
        if not isinstance(mask, list) or len(mask) != len(vals):
            mask = [v is not None for v in vals]
        curves.append({
            "id": cid,
            "unit": unit,
            "meaning": meaning,
            "scope": "full_track",
            "time_seconds": mtimes,
            "values": vals,
            "valid": [bool(v) and v is not None for v in mask],
            "source": "original_stereo" if mod_name == "spatial" else "mono_decode",
            "method_version": (mod.get("meta") or {}).get("method_version", version),
        })
    return curves


def _loads(raw, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def get_track_by_path(conn: sqlite3.Connection, file_path: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM tracks WHERE file_path=?", (str(file_path),)).fetchone()


def scan_state(conn: sqlite3.Connection) -> dict[str, tuple[int, float, str]]:
    """{file_path: (size, mtime, analysis_version)} for rows that hold a complete analysis.

    Only rows with a stored stat_vector count as analysed: a track written by an
    earlier crash mid-pipeline must be re-done, not skipped forever. `analysis_version`
    travels with the file signature so a row analysed by an older algorithm is NOT skippable
    -- bumping `analysis.version` in config.yaml is what re-extracts the library after a
    structure/dynamics change (docs/MUSIC_ANALYSIS_REFACTOR_PLAN.md 5.3).
    """
    rows = conn.execute(
        "SELECT file_path, file_size, file_mtime, analysis_version AS av FROM tracks "
        "WHERE stat_vector IS NOT NULL AND file_size IS NOT NULL AND file_mtime IS NOT NULL"
    ).fetchall()
    # None (not "") for a row written before versioning existed: the reader must be able to
    # tell "no version was recorded" apart from a genuine empty value.
    return {r["file_path"]: (int(r["file_size"]), float(r["file_mtime"]),
                             None if r["av"] is None else str(r["av"])) for r in rows}


def paths_without_tags(conn: sqlite3.Connection) -> set[str]:
    """Files whose tag columns were never filled from the file itself."""
    rows = conn.execute(
        "SELECT file_path FROM tracks WHERE stat_vector IS NOT NULL "
        "AND (meta_source IS NULL OR meta_source='none')"
    ).fetchall()
    return {r["file_path"] for r in rows}


def update_tags(conn: sqlite3.Connection, file_path: str, tags: dict) -> None:
    """Write only the tag columns for an already-analysed file (no re-analysis).

    `meta_updated_at` is stamped only when a value really changed, because the Music
    Space cache keys off it and a plain re-scan of an unchanged library must not
    invalidate every score.
    """
    before = conn.execute(
        "SELECT title, artist, album, year, language, genre, meta_source FROM tracks "
        "WHERE file_path=:file_path", {"file_path": str(file_path)}).fetchone()
    if before is None:
        return
    new = {"file_path": str(file_path),
           "title": tags.get("title"), "artist": tags.get("artist"),
           "album": tags.get("album"), "year": tags.get("year"),
           "language": tags.get("language"), "genre": tags.get("genre"),
           "meta_source": tags.get("meta_source", "none")}
    if all(before[k] == new[k] for k in new if k != "file_path"):
        return
    conn.execute(
        "UPDATE tracks SET title=:title, artist=:artist, album=:album, year=:year, "
        "language=:language, genre=:genre, meta_source=:meta_source, "
        "meta_updated_at=:now WHERE file_path=:file_path",
        {**new, "now": _dt.datetime.now(_dt.timezone.utc).isoformat()},
    )
    conn.commit()


def display_map(conn: sqlite3.Connection, track_ids) -> dict[str, dict]:
    """{track_id: {title, artist, album, language, genre, file_path, meta_source}} in one query.

    A player renders a recommendation list without N+1 requests, and `meta_source`
    travels with every row so the UI can never present a tag as analysed audio.
    """
    ids = [t for t in dict.fromkeys(track_ids) if t]
    if not ids:
        return {}
    q = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT track_id, title, artist, album, language, genre, file_path, meta_source "
        f"FROM tracks WHERE track_id IN ({q})",
        ids,
    ).fetchall()
    return {r["track_id"]: dict(r) for r in rows}


def all_tracks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM tracks ORDER BY track_id").fetchall()


def track_columns(conn: sqlite3.Connection) -> list[str]:
    return [r[1] for r in conn.execute("PRAGMA table_info(tracks)")]


def rows_with_columns(conn: sqlite3.Connection, columns: tuple[str, ...] | list[str]) -> list[sqlite3.Row]:
    """Load only the named feature columns.

    The caller owns the column list; columns absent from the schema are dropped rather
    than raising, so an older database file still loads.
    """
    have = set(track_columns(conn))
    cols = [c for c in columns if c in have]
    if not cols:
        return []
    return conn.execute("SELECT {} FROM tracks ORDER BY track_id".format(", ".join(cols))).fetchall()


def library_version(conn: sqlite3.Connection) -> tuple:
    """Fingerprint of the feature data behind the cached Music Space.

    Any upsert_track rewrites analyzed_at, so adding, removing or re-analysing a track
    changes this value and invalidates the cache. Tag writes are covered too, but only
    the ones that actually changed a value (`update_tags` stamps `meta_updated_at`):
    language/genre are now read by category hard filters and by discovered-cluster
    naming, so a tag backfill must not be able to leave a stale Music Space alive.
    MIN/MAX track_id are part of the fingerprint so two different libraries can never
    collide on an equal row count and timestamp -- in-memory and temp-file databases
    are both in play.
    """
    r = conn.execute(
        "SELECT COUNT(*) AS c, MAX(rowid) AS r, MAX(analyzed_at) AS a, "
        "MAX(meta_updated_at) AS m, MIN(track_id) AS lo, MAX(track_id) AS hi FROM tracks"
    ).fetchone()
    return (r["c"], r["r"], r["a"], r["m"], r["lo"], r["hi"])


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
