"""Run the local HTTP API on 127.0.0.1.

Usage:
    python scripts/run_server.py [--port 8000]
"""
from __future__ import annotations

import argparse

import uvicorn

from common import get_logger
from music_recommender.api.server import create_app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    log = get_logger()
    log.info("starting API on http://%s:%d", args.host, args.port)
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
