"""Build the FAISS index from embeddings stored in SQLite.

Usage:
    python scripts/build_faiss.py
"""
from __future__ import annotations

from common import get_logger
from music_recommender.library import MusicLibrary


def main():
    log = get_logger()
    lib = MusicLibrary()
    store = lib.build_faiss()
    log.info("FAISS index saved with %d vectors", store.index.ntotal)
    print(f"faiss_vectors={store.index.ntotal} dim={store.dim}")


if __name__ == "__main__":
    main()
