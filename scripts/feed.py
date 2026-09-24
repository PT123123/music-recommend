"""Dynamic Feed CLI (deliverable 11): get next tracks, send feedback.

Usage:
    python scripts/feed.py next --limit 10
    python scripts/feed.py feedback --track-id <id> --event like
    python scripts/feed.py state
    python scripts/feed.py reset --scope short
"""
from __future__ import annotations

import argparse

from common import get_logger
from music_recommender.database import repository
from music_recommender.database.schema import connect
from music_recommender.recommender import Recommender
from music_recommender.utils.config import get_config


def _name(conn, tid):
    from pathlib import Path
    row = repository.get_track(conn, tid)
    return Path(row["file_path"]).name if row else tid


def main():
    ap = argparse.ArgumentParser(description="Dynamic feed (no-brain stream)")
    ap.add_argument("cmd", choices=["next", "feedback", "state", "reset"])
    ap.add_argument("--user", default="local-user")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--track-id")
    ap.add_argument("--event", default="play",
                    choices=["impression", "play", "skip", "like", "dislike", "complete", "replay", "partial"])
    ap.add_argument("--completion", type=float)
    ap.add_argument("--scope", default="short", choices=["short", "medium", "long", "all"])
    args = ap.parse_args()

    cfg = get_config()
    conn = connect(cfg.db_path)
    rec = Recommender(cfg)
    log = get_logger()

    if args.cmd == "next":
        items = rec.feed_next(args.user, limit=args.limit)
        for i, it in enumerate(items, 1):
            sc = f"{it['score']:.3f}" if it.get("score") is not None else "-"
            print(f"{i}. {_name(conn, it['track_id']):28s} {sc}  ({it.get('reason','')})")
    elif args.cmd == "feedback":
        if not args.track_id:
            ap.error("feedback needs --track-id")
        iid = rec.feedback(args.user, args.track_id, args.event, completion_ratio=args.completion)
        print(f"logged event '{args.event}' for {args.track_id} (id={iid})")
    elif args.cmd == "state":
        import json
        print(json.dumps(rec.feed_state(args.user), ensure_ascii=False, indent=2))
    elif args.cmd == "reset":
        n = rec.feed_reset(args.user, args.scope)
        log.info("reset scope=%s removed=%d", args.scope, n)
        print(f"reset scope={args.scope}, removed {n} interactions")


if __name__ == "__main__":
    main()
