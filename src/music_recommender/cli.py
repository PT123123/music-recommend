"""Minimal `music-recommender` console entry point (optional convenience)."""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser(prog="music-recommender")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan", help="see scripts/scan_library.py")
    sub.add_parser("recommend", help="see scripts/recommend.py")
    sub.add_parser("serve", help="see scripts/run_server.py")
    args = ap.parse_args()
    print(f"Run the dedicated script under scripts/ for '{args.cmd}'. "
          f"e.g. python scripts/{args.cmd if args.cmd != 'serve' else 'run_server'}.py")


if __name__ == "__main__":
    main()
