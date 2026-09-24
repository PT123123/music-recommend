"""Read/write user behaviour + interest snapshots (spec 56-57)."""
from __future__ import annotations

import sqlite3
import time
from typing import Optional

import numpy as np


def log_interaction(conn: sqlite3.Connection, user_id: str, track_id: str, event: str,
                    timestamp: Optional[float] = None, play_seconds: Optional[float] = None,
                    completion_ratio: Optional[float] = None) -> int:
    ts = float(timestamp if timestamp is not None else time.time())
    cur = conn.execute(
        "INSERT INTO interactions (user_id, track_id, event, timestamp, play_seconds, completion_ratio) "
        "VALUES (?,?,?,?,?,?)",
        (user_id, track_id, event, ts, play_seconds, completion_ratio),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_interactions(conn: sqlite3.Connection, user_id: str, since: Optional[float] = None,
                     limit: Optional[int] = None) -> list[sqlite3.Row]:
    q = "SELECT * FROM interactions WHERE user_id=?"
    params: list = [user_id]
    if since is not None:
        q += " AND timestamp>=?"
        params.append(since)
    q += " ORDER BY timestamp DESC"
    if limit:
        q += " LIMIT ?"
        params.append(int(limit))
    return conn.execute(q, params).fetchall()


def recent_track_ids(conn: sqlite3.Connection, user_id: str, n: int) -> list[str]:
    """Tracks the user has just touched, for feed repeat-exclusion (spec 65).

    'like' counts too: a like only happens on a track already heard, so resurfacing
    it one second later is the exact failure spec 65 guards against. 'dislike' is
    excluded via negative interest weight, not recency.
    """
    rows = conn.execute(
        "SELECT track_id FROM interactions WHERE user_id=? AND event IN ('play','complete','replay','like') "
        "ORDER BY timestamp DESC LIMIT ?", (user_id, int(n))).fetchall()
    return [r["track_id"] for r in rows]


def local_popularity(conn: sqlite3.Connection, user_id: str) -> dict[str, float]:
    """Local popularity = positive behavioural signal counts. NEVER external platforms (spec 33/66)."""
    rows = conn.execute(
        "SELECT track_id, SUM(CASE WHEN event IN ('like','complete','replay') THEN 1 ELSE 0 END) AS pos "
        "FROM interactions WHERE user_id=? GROUP BY track_id", (user_id,)).fetchall()
    counts = {r["track_id"]: float(r["pos"]) for r in rows}
    mx = max(counts.values()) if counts else 0.0
    return {t: (c / mx if mx else 0.0) for t, c in counts.items()}


def save_interest_snapshot(conn: sqlite3.Connection, user_id: str, scope: str,
                           vector: Optional[np.ndarray], metadata: dict, timestamp: Optional[float] = None) -> None:
    ts = float(timestamp if timestamp is not None else time.time())
    blob = None if vector is None else np.asarray(vector, dtype="float32").tobytes()
    import json
    conn.execute(
        "INSERT INTO interest_snapshots (user_id, timestamp, scope, vector, metadata_json) VALUES (?,?,?,?,?)",
        (user_id, ts, scope, blob, json.dumps(metadata, ensure_ascii=False)))
    conn.commit()


def reset_scope(conn: sqlite3.Connection, user_id: str, scope: str = "short") -> int:
    """Clear interactions within a scope window. Long data is preserved unless explicitly requested."""
    windows = {"short": 3600, "medium": 86400, "long": 2592000}
    if scope == "short":
        since = time.time() - windows["short"]
        cur = conn.execute("DELETE FROM interactions WHERE user_id=? AND timestamp>=?", (user_id, since))
    elif scope == "all":
        cur = conn.execute("DELETE FROM interactions WHERE user_id=?", (user_id,))
    else:
        since = time.time() - windows.get(scope, windows["short"])
        cur = conn.execute("DELETE FROM interactions WHERE user_id=? AND timestamp>=?", (user_id, since))
    conn.commit()
    return int(cur.rowcount or 0)


# ---- human evaluation (spec 80) ----

def record_evaluation(conn: sqlite3.Connection, seed_track_id: str, recommended_track_id: str,
                      label: int, note: Optional[str] = None) -> int:
    """label: 1 = good/recommended, 0 = not. Mirrors the pairwise like/dislike eval."""
    import datetime as _dt
    cur = conn.execute(
        "INSERT INTO evaluations (seed_track_id, recommended_track_id, label, note, created_at) "
        "VALUES (?,?,?,?,?)",
        (seed_track_id, recommended_track_id, int(label), note,
         _dt.datetime.now(_dt.timezone.utc).isoformat()))
    conn.commit()
    return int(cur.lastrowid)


def export_evaluations_csv(conn: sqlite3.Connection, out_path) -> int:
    """Write `seed,recommendation,label` rows (spec 80) and return the row count."""
    import csv
    from pathlib import Path
    rows = conn.execute("SELECT seed_track_id, recommended_track_id, label FROM evaluations ORDER BY id").fetchall()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "recommendation", "label"])
        for r in rows:
            w.writerow([r["seed_track_id"], r["recommended_track_id"], r["label"]])
    return len(rows)


def evaluation_stats(conn: sqlite3.Connection) -> dict:
    row = conn.execute("SELECT COUNT(*) AS n, COALESCE(SUM(label),0) AS pos FROM evaluations").fetchone()
    n = int(row["n"])
    return {"count": n, "positive": int(row["pos"]), "agreement": (row["pos"] / n) if n else None}
