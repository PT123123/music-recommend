"""FastAPI HTTP server exposing the /v1 API (spec 43-52).

All Phase 1-7 endpoints are live: health, tracks, categories, recommend/similar,
recommend/category, feed/next, feed/feedback, feed/state, feed/reset, evaluate,
library/scan, tracks/index. HTTP and Python API share one Recommendation Core.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..database import repository
from ..database.schema import connect
from ..library import MusicLibrary
from ..recommender import Recommender
from ..utils.config import get_config
from ..utils.logging import get_logger

log = get_logger()


class SimilarRequest(BaseModel):
    track_id: str | None = None
    file_path: str | None = None
    limit: int = 20


class ScanRequest(BaseModel):
    root: str
    recursive: bool = True
    workers: int = 1


class IndexRequest(BaseModel):
    file_path: str


class CategoryRequest(BaseModel):
    category_id: str | None = None
    query: dict | None = None
    hard_filter: dict | None = None
    limit: int = 30


class FeedbackRequest(BaseModel):
    user_id: str = "local-user"
    track_id: str
    event: str
    timestamp: float | None = None
    play_seconds: float | None = None
    completion_ratio: float | None = None


class FeedNextRequest(BaseModel):
    user_id: str = "local-user"
    limit: int = 20
    exclude_track_ids: list[str] | None = None


class EvaluateRequest(BaseModel):
    seed_track_id: str
    recommended_track_id: str
    label: int
    note: str | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="Local Music MIR Recommender", version="0.1.0")

    @app.get("/v1/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/tracks/{track_id}")
    def get_track(track_id: str):
        cfg = get_config()
        conn = connect(cfg.db_path)
        row = repository.get_track(conn, track_id)
        if row is None:
            raise HTTPException(404, "track not found")
        d = dict(row)
        for blob in ("mfcc_means", "mfcc_stds", "chroma_mean", "stat_vector"):
            d.pop(blob, None)
        return d

    @app.get("/v1/categories")
    def categories():
        return Recommender().categories()

    @app.post("/v1/recommend/similar")
    def recommend_similar(req: SimilarRequest):
        rec = Recommender()
        if req.track_id:
            items = rec.similar(req.track_id, limit=req.limit)
        elif req.file_path:
            try:
                items = rec.similar_by_path(req.file_path, limit=req.limit)
            except Exception as exc:
                raise HTTPException(400, str(exc))
        else:
            raise HTTPException(400, "provide track_id or file_path")
        return {"seed": req.track_id or req.file_path,
                "recommendations": [{"track_id": r.track_id, "score": r.score, "reasons": r.reasons}
                                    for r in items]}

    @app.post("/v1/library/scan")
    def library_scan(req: ScanRequest):
        if not Path(req.root).exists():
            raise HTTPException(400, "root does not exist")
        lib = MusicLibrary()
        summary = lib.index(req.root, recursive=req.recursive, workers=req.workers)
        lib.rebuild_embeddings()
        lib.build_faiss()
        return summary

    @app.post("/v1/tracks/index")
    def track_index(req: IndexRequest):
        if not Path(req.file_path).exists():
            raise HTTPException(400, "file not found")
        lib = MusicLibrary()
        tid = lib.index_one(Path(req.file_path))
        if tid is None:
            raise HTTPException(500, "indexing failed; see logs/errors.log")
        return {"track_id": tid}

    @app.post("/v1/recommend/category")
    def recommend_category(req: CategoryRequest):
        rec = Recommender()
        try:
            if req.category_id:
                items, meta = rec.category(req.category_id, limit=req.limit)
                seed = req.category_id
            elif req.query:
                items, meta = rec.category_query(req.query, req.hard_filter, limit=req.limit)
                seed = "custom_query"
            else:
                raise HTTPException(400, "provide category_id or query")
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        return {"category": seed, "ignored_dims": meta.get("ignored", []),
                "usable_dims": meta.get("usable", 0),
                "recommendations": items}

    @app.post("/v1/feed/next")
    def feed_next(req: FeedNextRequest):
        rec = Recommender()
        try:
            items = rec.feed_next(req.user_id, limit=req.limit, exclude_track_ids=req.exclude_track_ids)
        except Exception as exc:
            raise HTTPException(500, str(exc))
        return {"items": items}

    @app.post("/v1/feed/feedback")
    def feed_feedback(req: FeedbackRequest):
        valid = {"impression", "play", "skip", "like", "dislike", "complete", "replay", "partial"}
        if req.event not in valid:
            raise HTTPException(400, f"event must be one of {sorted(valid)}")
        rec = Recommender()
        iid = rec.feedback(req.user_id, req.track_id, req.event, req.timestamp,
                           req.play_seconds, req.completion_ratio)
        return {"status": "ok", "interaction_id": iid}

    @app.get("/v1/feed/state")
    def feed_state(user_id: str = "local-user"):
        return Recommender().feed_state(user_id)

    @app.post("/v1/evaluate")
    def evaluate(req: EvaluateRequest):
        conn = connect(get_config().db_path)
        from ..database import behavior
        eid = behavior.record_evaluation(conn, req.seed_track_id, req.recommended_track_id,
                                         int(req.label), req.note)
        return {"status": "ok", "evaluation_id": eid}

    @app.post("/v1/feed/reset")
    def feed_reset(user_id: str = "local-user", scope: str = "short"):
        if scope not in {"short", "medium", "long", "all"}:
            raise HTTPException(400, "scope must be short|medium|long|all")
        removed = Recommender().feed_reset(user_id, scope)
        return {"status": "ok", "removed": removed, "scope": scope}

    return app
