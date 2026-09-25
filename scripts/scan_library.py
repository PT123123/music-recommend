"""Full one-shot pipeline: scan -> extract -> embed -> build FAISS.

Usage:
    python scripts/scan_library.py D:/Music [--no-recursive]
"""
from __future__ import annotations

import argparse

from common import get_logger
from music_recommender.library import MusicLibrary


def main():
    ap = argparse.ArgumentParser(description="Index a music folder end-to-end (features + embeddings + FAISS)")
    ap.add_argument("root", help="music folder to scan")
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="parallel feature-extraction processes")
    ap.add_argument("--force", action="store_true", help="re-analyse files whose size+mtime are unchanged")
    args = ap.parse_args()

    log = get_logger()
    lib = MusicLibrary()
    summary = lib.index(args.root, recursive=not args.no_recursive, workers=args.workers,
                        force=args.force)
    lib.rebuild_embeddings()
    lib.build_faiss()
    log.info("pipeline done")
    print(f"found={summary['found']} indexed={summary['indexed']} "
          f"skipped={summary['skipped']} failed={summary['failed']}")


if __name__ == "__main__":
    main()
