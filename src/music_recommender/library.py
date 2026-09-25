"""MusicLibrary: end-to-end orchestration of scan -> preprocess -> extract ->
store -> embed -> FAISS. This is the Python API entry point (spec 53).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .database import repository
from .database.schema import connect, init_db
from .embedding.pca_embedder import PcaStatsEmbedder
from .features import extract
from .preprocess.audio import file_signature, normalize, track_id_for
from .preprocess.metadata import read_tags
from .utils.config import Config, get_config
from .utils.logging import get_logger
from .vector_store.faiss_store import FaissStore

log = get_logger()


def analyze_file(file_path_str: str):
    """Decode + extract features + read tags (no DB), so it can run in a worker process.

    Returns (file_path, track_id, wav_path, TrackFeatures, tags) or None on failure.
    A single bad file must never abort the batch (spec 69).
    """
    try:
        file_path = Path(file_path_str)
        wav, duration, sr = normalize(file_path)
        feats = extract.extract(str(wav), sr, duration)
        return (str(file_path.resolve()), track_id_for(file_path), str(wav), feats, read_tags(file_path))
    except Exception as exc:  # noqa: BLE001
        log.exception("analyze failed for %s: %s", file_path_str, exc)
        return None


def _safe_signature(file_path: str):
    """(size, mtime) or None when the file vanished between scan and stat."""
    try:
        return file_signature(file_path)
    except OSError:
        return None


class MusicLibrary:
    def __init__(self, cfg: Config | None = None, conn: sqlite3.Connection | None = None):
        self.cfg = cfg or get_config()
        self.conn = conn or connect(self.cfg.db_path)
        init_db(self.conn)

    # ---- scanning ----
    def scan(self, root: str | Path, recursive: bool = True) -> list[Path]:
        root = Path(root)
        exts = set(self.cfg.audio.get("extensions", [".mp3", ".wav"]))
        globber = root.rglob if recursive else root.glob
        return sorted(p for p in globber("*") if p.suffix.lower() in exts and p.is_file())

    def _store(self, result, embedding_model: str) -> str | None:
        if result is None:
            return None
        file_path, tid, wav_path, feats, tags = result
        try:
            repository.upsert_track(self.conn, tid, file_path, wav_path, feats,
                                    embedding_model=embedding_model, tags=tags)
            repository.insert_segments(self.conn, tid, feats.extras.get("segment_list", []))
            log.info("indexed %s", Path(file_path).name)
            return tid
        except Exception as exc:  # noqa: BLE001
            log.exception("failed to store %s: %s", file_path, exc)
            return None

    def index_one(self, file_path: Path) -> str | None:
        model = self.cfg.embedding.get("backend", "librosa_pca")
        return self._store(analyze_file(str(file_path)), model)

    def index(self, root: str | Path, recursive: bool = True, workers: int = 1,
              force: bool = False) -> dict:
        files = self.scan(root, recursive)
        model = self.cfg.embedding.get("backend", "librosa_pca")
        paths = [str(f) for f in files]

        # Incremental scan: a file whose size+mtime match the stored row is already
        # analysed, and re-decoding it is by far the most expensive thing a re-scan
        # can do (tens of seconds per track). `force` re-analyses everything, which
        # is what a feature-version change needs.
        skipped: list[str] = []
        if force:
            todo = paths
        else:
            known = repository.scan_state(self.conn)
            todo = []
            for p in paths:
                sig = _safe_signature(p)
                if sig is not None and known.get(str(Path(p).resolve())) == sig:
                    skipped.append(p)
                else:
                    todo.append(p)
            if skipped:
                log.info("skipped %d unchanged file(s), analysing %d", len(skipped), len(todo))
                self._backfill_tags(skipped)

        if workers and workers > 1 and len(todo) > 1:
            import os
            from concurrent.futures import ProcessPoolExecutor

            # keep each worker single-threaded so N workers don't oversubscribe cores
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            with ProcessPoolExecutor(max_workers=min(int(workers), len(todo))) as ex:
                results = list(ex.map(analyze_file, todo, chunksize=1))
        else:
            results = [analyze_file(p) for p in todo]

        ok, failed = [], []
        for r in results:
            tid = self._store(r, model)
            (ok if tid else failed).append(r[0] if r else "<analyze-failed>")
        summary = {"found": len(files), "indexed": len(ok), "skipped": len(skipped),
                   "failed": len(failed), "failed_paths": failed}
        log.info("scan complete: %s", summary)
        return summary

    def _backfill_tags(self, paths: list[str]) -> None:
        """Read tags for files whose audio we skipped, so a tag-only upgrade still lands."""
        missing = repository.paths_without_tags(self.conn)
        if not missing:
            return
        for p in paths:
            rp = str(Path(p).resolve())
            if rp in missing:
                repository.update_tags(self.conn, rp, read_tags(p))

    # ---- embedding + faiss ----
    def _stat_vectors(self) -> list[tuple[str, np.ndarray]]:
        rows = self.conn.execute("SELECT track_id, stat_vector FROM tracks WHERE stat_vector IS NOT NULL ORDER BY track_id").fetchall()
        out = []
        for r in rows:
            v = np.frombuffer(r["stat_vector"], dtype="float32")
            out.append((r["track_id"], v))
        return out

    def rebuild_embeddings(self) -> PcaStatsEmbedder:
        pairs = self._stat_vectors()
        if not pairs:
            raise RuntimeError("no tracks to embed; run index first")
        matrix = np.stack([v for _, v in pairs], axis=0)
        embedder = PcaStatsEmbedder(n_components=int(self.cfg.embedding.get("pca_components", 24)),
                                    name=self.cfg.embedding.get("backend", "librosa_pca"))
        embedder.fit(matrix)
        embs = embedder.embed_many(matrix)
        for i, (tid, _) in enumerate(pairs):
            repository.upsert_embedding(self.conn, tid, embs[i], i, embedder.name)
        embedder.save(self.cfg.abs(self.cfg.embedding.get("model_store", "data/faiss/embedding_pipeline.joblib")))
        log.info("embedded %d tracks (dim=%d)", len(pairs), embedder.dim)
        return embedder

    def build_faiss(self, embedder: PcaStatsEmbedder | None = None) -> FaissStore:
        ordered = repository.ordered_embeddings(self.conn)
        if not ordered:
            raise RuntimeError("no embeddings; run rebuild_embeddings first")
        dim = ordered[0][2].shape[0]
        store = FaissStore(self.cfg.faiss_dir, dim=dim,
                           model_name=self.cfg.embedding.get("backend", "librosa_pca"),
                           vs_cfg=self.cfg.vector_store)
        store.build(ordered)
        store.save()
        log.info("built FAISS index with %d vectors (dim=%d)", store.index.ntotal, dim)
        return store

    def embed_query_file(self, file_path: Path, embedder: PcaStatsEmbedder) -> np.ndarray | None:
        try:
            wav, duration, sr = normalize(file_path)
            feats = extract.extract(str(wav), sr, duration)
            return embedder.embed(feats.stat_vector)
        except Exception as exc:
            log.exception("embed_query_file failed for %s: %s", file_path, exc)
            return None

    def close(self):
        self.conn.close()
