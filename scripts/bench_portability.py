"""Measure what a non-Python (mobile) implementation would actually have to carry.

Answers three questions with numbers instead of adjectives:
  1. payload  -- bytes per track for the *scoring* data vs the whole database row;
  2. arithmetic -- operations per query, so a phone cost estimate is not a guess;
  3. parity   -- exports the real library so rust/musicspace can be checked against
     Python's own ranking (scripts/bench_portability.py --export, then cargo run).

Run:  PYTHONPATH=src .venv/Scripts/python.exe scripts/bench_portability.py --json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from music_recommender.database import repository                      # noqa: E402
from music_recommender.recommendation.sequence_sim import (            # noqa: E402
    levenshtein_similarity, ngram_similarity)
from music_recommender.recommendation.space import (                   # noqa: E402
    FEATURE_GROUPS, SEQUENCE_COLS, SPACE_COLUMNS, MusicSpace)
from music_recommender.recommender import Recommender                  # noqa: E402
from music_recommender.utils.config import get_config                  # noqa: E402


def column_bytes(conn: sqlite3.Connection) -> dict[str, int]:
    """Stored bytes per column. Names come from PRAGMA, never from user input."""
    out = {}
    for col in [r[1] for r in conn.execute("PRAGMA table_info(tracks)")]:
        try:
            out[col] = int(conn.execute(
                f"SELECT COALESCE(SUM(LENGTH({col})),0) FROM tracks").fetchone()[0])
        except sqlite3.OperationalError:
            out[col] = 0
    return out


def op_counts(space: MusicSpace, weights: dict, ids: list[str], sample: int = 20) -> dict:
    """Count the arithmetic one song->song query performs, on a sample of seeds."""
    cells = {"levenshtein": 0, "ngram": 0, "cosine": 0}
    pairs = groups = 0
    step = max(1, len(ids) // sample)
    for seed in ids[::step][:sample]:
        cands = [t for t in ids if t != seed]
        pairs += len(cands)
        for cand in cands:
            for group in weights:
                if group == "embedding":
                    continue
                groups += 1
                col = SEQUENCE_COLS.get(group)
                if not col:
                    continue
                a = space._seq(space.rows[seed], col, seed)
                b = space._seq(space.rows[cand], col, cand)
                if len(a) >= 2 and len(b) >= 2:
                    levenshtein_similarity(a, b)
                    ngram_similarity(a, b)
                    cells["levenshtein"] += len(a) * len(b)
                    cells["ngram"] += len(a) + len(b)
    return {"candidate_pairs": pairs, "group_similarities": groups,
            "levenshtein_cells": cells["levenshtein"], "ngram_tokens": cells["ngram"],
            "seeds_sampled": len(ids[::step][:sample])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print machine-readable output")
    ap.add_argument("--export", default=str(ROOT / "data" / "portability"),
                    help="directory for the Rust-parity export")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    cfg = get_config()
    rec = Recommender(cfg=cfg)
    conn = rec.conn
    ids = [r["track_id"] for r in repository.all_tracks(conn)]
    n = len(ids)
    space = MusicSpace(conn)

    db_bytes = cfg.db_path.stat().st_size if cfg.db_path.exists() else 0
    faiss_dir = cfg.faiss_dir
    index_bytes = sum(p.stat().st_size for p in faiss_dir.iterdir() if p.is_file()) if faiss_dir.exists() else 0
    col_bytes = column_bytes(conn)
    score_bytes = sum(col_bytes.get(c, 0) for c in SPACE_COLUMNS)
    seq_bytes = sum(col_bytes.get(c, 0) for c in SEQUENCE_COLS.values())
    emb_dim = int((faiss_dir / "embedding_dimension.txt").read_text()) if (faiss_dir / "embedding_dimension.txt").exists() else 0

    # latency, after one warm-up so the cached Music Space is not charged to the first query
    rec.similar(ids[0], limit=20)
    def timed(fn, reps=5):
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t) / reps
    lat = {"similar_s": timed(lambda: rec.similar(ids[0], limit=20)),
           "category_s": timed(lambda: rec.category(cfg.categories[0]["id"], limit=20)),
           "feed_s": timed(lambda: rec.feed_next("local-user", limit=20))}

    counts = op_counts(space, rec._core.weights, ids)
    top_k = int(cfg.retrieval.get("faiss_top_k", 200))

    report = {
        "library": {"tracks": n, "embedding_dim": emb_dim, "faiss_top_k": top_k,
                    "db_bytes": db_bytes, "db_bytes_per_track": round(db_bytes / n) if n else 0,
                    "index_bytes": index_bytes, "index_bytes_per_track": round(index_bytes / n) if n else 0},
        "payload_per_track": {
            "all_columns": round(sum(col_bytes.values()) / n) if n else 0,
            "scoring_columns_only": round(score_bytes / n) if n else 0,
            "sequence_columns": round(seq_bytes / n) if n else 0,
            "embedding_floats": emb_dim * 4,
            "note": "scoring_columns excludes embeddings/segments/JSON dumps; a phone needs "
                    "scoring columns + the stored embedding, never the analysis intermediates",
        },
        "per_song_to_song_query": counts | {
            "scored_pairs": min(n - 1, top_k),
            "weight_groups": len(rec._core.weights),
        },
        "python_desktop_latency_s": {k: round(v, 4) for k, v in lat.items()},
    }

    if not args.no_export:
        export(conn, rec, Path(args.export), ids)
        report["export"] = {"dir": str(args.export), "tracks": n}

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for section, body in report.items():
            print(f"[{section}]")
            if isinstance(body, dict):
                for k, v in body.items():
                    print(f"  {k} = {v}")
        # projections a phone decision needs, kept next to the measured numbers
        per_track = report["payload_per_track"]
        scale = 5000
        print(f"\n[scale projection] {scale} tracks -> "
              f"~{(per_track['scoring_columns_only'] + per_track['embedding_floats']) * scale / 1e6:.1f} MB "
              f"of scoring payload (audio excluded)")


def export(conn, rec, out_dir: Path, ids: list[str]) -> None:
    """Write the scoring payload + Python's own top-20 so rust/musicspace can be checked.

    Deliberately JSON-free: fields are tab-separated, multi-valued cells use \\x1f, and
    the parameters live in a key/value file. A std-only Rust reader is ~20 lines, and
    this artifact is a *parity harness*, not a data store (SQLite stays the truth).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    space = MusicSpace(conn)
    embeddings = {tid: vec for _, tid, vec in repository.ordered_embeddings(conn)}
    scalar_cols = sorted({c for cols in FEATURE_GROUPS.values() for c in cols})
    header = ["track_id", *scalar_cols, *SEQUENCE_COLS.values(), "energy_curve", "has_vocal", "embedding"]
    us = "\x1f"

    def cell(v) -> str:
        return "" if v is None else repr(float(v))

    with (out_dir / "tracks.tsv").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(header) + "\n")
        for tid in ids:
            row = space.rows[tid]
            seqs = [us.join(str(x) for x in space._seq(row, col, tid)) for col in SEQUENCE_COLS.values()]
            fields = [tid, *(cell(row.get(c)) for c in scalar_cols), *seqs,
                      us.join(repr(float(v)) for v in space._curve(tid)),
                      "1" if row.get("has_vocal") else "0",
                      us.join(f"{float(v):.9g}" for v in embeddings[tid])]
            fh.write("\t".join(fields) + "\n")

    groups = "|".join(f"{g}:{','.join(cols)}" for g, cols in FEATURE_GROUPS.items())
    seq_groups = "|".join(f"{g}:{c}" for g, c in SEQUENCE_COLS.items())
    weights = "|".join(f"{g}:{v}" for g, v in rec._core.weights.items())
    kv = {
        "scalar_cols": ",".join(scalar_cols),
        "seq_cols": ",".join(SEQUENCE_COLS.values()),
        "feature_groups": groups,
        "sequence_cols_by_group": seq_groups,
        "weights": weights,
        "blend_scalar_sequence": "0.5",
        "structure_curve_blend": "0.6|0.4",
        "combined_seq_weights": "0.35|0.25|0.2|0.2",
        "ngram_n": "2",
        "limit": "20",
        "tracks": str(len(ids)),
    }
    with (out_dir / "manifest.kv").open("w", encoding="utf-8", newline="\n") as fh:
        for k, v in kv.items():
            fh.write(f"{k}\t{v}\n")
    with (out_dir / "expected_top20.tsv").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("seed\trank\ttrack_id\tscore\n")
        for tid in ids[:12]:
            for rank, r in enumerate(rec.similar(tid, limit=20)):
                fh.write(f"{tid}\t{rank}\t{r.track_id}\t{r.score!r}\n")


if __name__ == "__main__":
    main()
