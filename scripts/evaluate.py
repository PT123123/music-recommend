"""Human evaluation harness (spec 80): label recommendation pairs, export CSV.

Math distance != perceived similarity, so we capture ground-truth labels to tune
weights/models later.

Usage:
    python scripts/evaluate.py                       # interactive: label pairs
    python scripts/evaluate.py --record --seed-id X --rec-id Y --label 1 --note ok
    python scripts/evaluate.py --export data/evaluations.csv
    python scripts/evaluate.py --stats
"""
from __future__ import annotations

import argparse
import sys

from common import get_logger, track_label
from music_recommender.database import behavior, repository
from music_recommender.database.schema import connect
from music_recommender.recommender import Recommender
from music_recommender.utils.config import get_config


def main():
    ap = argparse.ArgumentParser(description="Human evaluation of recommendations")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--seed-id")
    ap.add_argument("--rec-id")
    ap.add_argument("--label", type=int, choices=[0, 1])
    ap.add_argument("--note")
    ap.add_argument("--export")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--limit", type=int, default=5, help="candidates per seed in interactive mode")
    args = ap.parse_args()

    cfg = get_config()
    conn = connect(cfg.db_path)
    log = get_logger()

    if args.export:
        n = behavior.export_evaluations_csv(conn, args.export)
        print(f"exported {n} evaluations -> {args.export}")
        return

    if args.stats:
        print(behavior.evaluation_stats(conn))
        return

    if args.record:
        if not (args.seed_id and args.rec_id and args.label is not None):
            ap.error("--record needs --seed-id --rec-id --label")
        iid = behavior.record_evaluation(conn, args.seed_id, args.rec_id, args.label, args.note)
        print(f"recorded evaluation id={iid}")
        return

    # interactive
    rec = Recommender(cfg)
    rows = repository.all_tracks(conn)
    if not rows:
        print("no tracks indexed; run scan_library.py first")
        return
    if not sys.stdin.isatty():
        print("non-interactive stdin; use --record / --export instead")
        return

    print("For each seed, label pairs: y=good, n=bad, s=skip, q=quit\n")
    for r in rows:
        seed = r["track_id"]
        seed_name = track_label(r, seed)
        recs = rec.similar(seed, limit=args.limit)
        for c in recs:
            cname = track_label(repository.get_track(conn, c.track_id), c.track_id)
            try:
                ans = input(f"seed={seed_name}  ->  {cname} ({c.score:.3f})  [y/n/s]? ").strip().lower()
            except EOFError:
                ans = "q"
            if ans == "q":
                print("bye")
                return
            if ans in ("y", "n"):
                behavior.record_evaluation(conn, seed, c.track_id, 1 if ans == "y" else 0, note="interactive")
                log.info("labeled %s->%s as %s", seed, c.track_id, ans)
    print("done. export with: python scripts/evaluate.py --export data/evaluations.csv")


if __name__ == "__main__":
    main()
