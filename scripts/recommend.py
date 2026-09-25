"""Song -> Song recommendation CLI (spec section 71).

Usage:
    python scripts/recommend.py D:/Music/a.mp3 --limit 20
    python scripts/recommend.py --track-id <id> --json
    python scripts/recommend.py D:/Music/a.mp3 --category high_energy
    python scripts/recommend.py --text "来点安静又明亮的纯音乐" --limit 10
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
    ap.add_argument("--category", help="fixed-category id (preset or auto-discovered)")
    ap.add_argument("--text", help='free Chinese request, e.g. "来点安静又明亮的纯音乐"')
    args = ap.parse_args()

    log = get_logger()
    cfg = get_config()
    rec = Recommender(cfg)

    if args.text or args.category:
        seed_label = args.text if args.text else args.category
        try:
            if args.text:
                items, meta = rec.category_text(args.text, limit=args.limit)
            else:
                items, meta = rec.category(args.category, limit=args.limit)
        except KeyError as exc:
            print(str(exc)); return
        if args.json:
            print(json.dumps({"category": seed_label, "meta": meta, "recommendations": items},
                             ensure_ascii=False, indent=2))
        else:
            if meta.get("ignored"):
                print(f"(ignored dims with no local feature: {meta['ignored']})")
            for t in meta.get("unmatched", []):
                print(f"（「{t['term']}」你的文件答不了：{t['reason']}）")
            for p in (meta.get("genre_proxies", []) or []) + (meta.get("vocal_proxies", []) or []):
                print(f"（「{p['term']}」按 {p.get('genre') or p.get('guess')} 近似："
                      f"{', '.join(p['dims'])}）")
            print(f"类别={seed_label}  命中维度={meta.get('used_dims')}  "
                  f"support={meta.get('support')}/{meta.get('candidate_pool')}"
                  f"{'  [支持度偏低]' if meta.get('low_support') else ''}")
            if meta.get("filters_applied"):
                est = set(meta.get("filter_estimated") or [])
                shown = [f"{k}{'(估计值)' if k in est else ''}" for k in meta["filters_applied"]]
                print(f"硬过滤={', '.join(shown)}")
            if meta.get("unscored_dims"):
                print(f"（{', '.join(meta['unscored_dims'])} 在过滤后的池子里没有数据，"
                      f"所以这些维度没有参与排序）")
            if not items:
                print("No category results — features for this category are not derivable locally.")
                return
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
