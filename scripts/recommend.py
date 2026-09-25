"""Song -> Song recommendation CLI (spec section 71).

Usage:
    python scripts/recommend.py D:/Music/a.mp3 --limit 20
    python scripts/recommend.py --track-id <id> --json
    python scripts/recommend.py D:/Music/a.mp3 --category high_energy
"""
from __future__ import annotations

import argparse
import json

from common import get_logger, track_label
from music_recommender.database import repository
from music_recommender.database.schema import connect
from music_recommender.recommender import Recommender
from music_recommender.utils.config import get_config


def _resolve_track_id(rec: Recommender, path: str) -> str | None:
    from pathlib import Path
    row = repository.get_track_by_path(rec.conn, str(Path(path).resolve()))
    return row["track_id"] if row else None


def _name(rec: Recommender, track_id: str) -> str:
    return track_label(repository.get_track(rec.conn, track_id), track_id)


def main():
    ap = argparse.ArgumentParser(description="Recommend similar local tracks")
    ap.add_argument("file", nargs="?", help="seed audio file path")
    ap.add_argument("--track-id", help="seed track id (if already indexed)")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--category", help="fixed-category id (Phase 4; MVP lists available)")
    args = ap.parse_args()

    log = get_logger()
    cfg = get_config()
    rec = Recommender(cfg)

    if args.category:
        try:
            items, meta = rec.category(args.category, limit=args.limit)
        except KeyError as exc:
            print(str(exc)); return
        if args.json:
            print(json.dumps({"category": args.category, "meta": meta, "recommendations": items},
                             ensure_ascii=False, indent=2))
        else:
            if meta.get("ignored"):
                print(f"(ignored non-local dims: {meta['ignored']})")
            if not items:
                print("No category results — features for this category are not derivable locally.")
                return
            print(f"Category: {args.category}")
            for i, it in enumerate(items, 1):
                print(f"{i}. {it['file_name']}   {it['score']:.3f}")
        return

    if not args.file and not args.track_id:
        ap.error("provide a seed file path or --track-id")

    tid = args.track_id or _resolve_track_id(rec, args.file)
    if tid:
        recs = rec.similar(tid, limit=args.limit)
        seed = tid
    else:
        log.info("seed file not in library; using ad-hoc embedding query")
        recs = rec.similar_by_path(args.file, limit=args.limit)
        seed = args.file

    if args.json:
        print(json.dumps({"seed": seed, "recommendations": [
            {"track_id": r.track_id, "score": r.score, "reasons": r.reasons, "breakdown": r.breakdown}
            for r in recs]}, ensure_ascii=False, indent=2))
    else:
        print(f"Seed: {seed}")
        for i, r in enumerate(recs, 1):
            print(f"{i}. {_name(rec, r.track_id)}   {r.score:.3f}   {', '.join(r.reasons)}")


if __name__ == "__main__":
    main()
