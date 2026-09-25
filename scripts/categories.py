"""List every category the recommender can answer for the current library.

Preset categories come from configs/categories.yaml; `auto-*` ones are discovered by
clustering the current Music Space, so their names and membership change with the
library (ADR-2). `--refresh` re-runs the clustering instead of reusing the cache.

Usage:
    python scripts/categories.py
    python scripts/categories.py --json
    python scripts/categories.py --refresh
"""
from __future__ import annotations

import argparse
import json

from music_recommender.recommender import Recommender
from music_recommender.utils.config import get_config


def main():
    ap = argparse.ArgumentParser(description="List preset + discovered categories")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="re-run cluster discovery")
    ap.add_argument("--no-discovered", action="store_true", help="preset categories only")
    args = ap.parse_args()

    rec = Recommender(get_config())
    if args.refresh:
        rec.discovered(force=True)
    cats = rec.categories(include_discovered=not args.no_discovered)

    if args.json:
        print(json.dumps(cats, ensure_ascii=False, indent=2))
        return

    _, meta = rec.discovered()
    print(f"曲库 {meta.get('library_size')} 首 | 聚类状态 {meta.get('status')}"
          f" | k={meta.get('k')} | 轮廓系数={meta.get('silhouette')}"
          f" | 特征维度={meta.get('features')}")
    for c in cats:
        support = c.get("support")
        flag = "  [支持度偏低]" if support is not None and c.get("low_support") else ""
        print(f"  {c['id']:<24} {c['name']:<28} 来源={c.get('source', 'preset')}"
              f"{'  成员=' + str(support) if support else ''}{flag}")
        if c.get("note"):
            print(f"  {'':<24} {c['note']}")


if __name__ == "__main__":
    main()
