"""(Re)fit the embedding pipeline over stored stat vectors and save embeddings.

Usage:
    python scripts/build_embeddings.py
"""
from __future__ import annotations

from common import get_logger
from music_recommender.library import MusicLibrary


def main():
    log = get_logger()
    lib = MusicLibrary()
    emb = lib.rebuild_embeddings()
    log.info("embedding pipeline fitted: dim=%d", emb.dim)
    print(f"embedded_dim={emb.dim}")


if __name__ == "__main__":
    main()
