"""Logging setup: a pipeline log plus a dedicated error log (spec section 69)."""
from __future__ import annotations

import logging
from pathlib import Path

from .config import get_config

_CONFIGURED = False


def setup_logging() -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("music_recommender")
    if _CONFIGURED:
        return logger

    cfg = get_config()
    logs_dir = cfg.abs(cfg.logging.get("logs_dir", "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, str(cfg.logging.get("level", "INFO")).upper(), logging.INFO)

    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    pipeline = logging.FileHandler(cfg.abs(cfg.logging.get("pipeline_log", "logs/pipeline.log")), encoding="utf-8")
    pipeline.setFormatter(fmt)
    pipeline.setLevel(level)

    errors = logging.FileHandler(cfg.abs(cfg.logging.get("errors_log", "logs/errors.log")), encoding="utf-8")
    errors.setFormatter(fmt)
    errors.setLevel(logging.ERROR)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.setLevel(level)

    logger.handlers.clear()
    for h in (pipeline, errors, console):
        logger.addHandler(h)

    _CONFIGURED = True
    return logger


def get_logger() -> logging.Logger:
    return setup_logging()
