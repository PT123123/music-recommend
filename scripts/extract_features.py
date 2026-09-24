"""Scan + preprocess + extract MIR features into SQLite.

Usage:
    python scripts/extract_features.py D:/Music [--no-recursive]
"""
from __future__ import annotations

import argparse

from common import get_logger
from music_recommender.library import MusicLibrary


def main():
    ap = argparse.ArgumentParser(description="Extract features from a music folder into SQLite")
    ap.add_argument("root", help="music folder to scan")
    ap.add_argument("--no-recursive", action="store_true")
    args = ap.parse_args()

    log = get_logger()
    lib = MusicLibrary()
    summary = lib.index(args.root, recursive=not args.no_recursive)
    log.info("features stored: %d tracks in DB", summary["indexed"])
    print(f"found={summary['found']} indexed={summary['indexed']} failed={summary['failed']}")


if __name__ == "__main__":
    main()
