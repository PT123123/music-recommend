"""Configuration loader (YAML) for the music recommender.

Single source of truth for paths, weights, categories and thresholds.
Nothing in core algorithms should hard-code these values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels above this file (src/music_recommender/utils/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Config:
    audio: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    embedding: dict = field(default_factory=dict)
    vector_store: dict = field(default_factory=dict)
    retrieval: dict = field(default_factory=dict)
    logging: dict = field(default_factory=dict)
    feature_version: str = "v1"
    weights: dict = field(default_factory=dict)
    categories: list = field(default_factory=list)
    # The one place to tune the whole category path: {engine, discovery, text, extra}.
    # `extra` holds hand-written categories merged into `categories` by id, so a user
    # can add or override preset categories without editing the shipped preset file.
    category_system: dict = field(default_factory=dict)

    # ---- path helpers (absolute) ----
    def abs(self, rel: str | os.PathLike) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def db_path(self) -> Path:
        return self.abs(self.paths["db_file"])

    @property
    def faiss_dir(self) -> Path:
        return self.abs(self.paths["faiss_dir"])

    @property
    def normalized_dir(self) -> Path:
        return self.abs(self.audio["normalized_dir"])

    def active_weights(self) -> dict[str, float]:
        """Return {group: weight} for weight groups enabled in weights.yaml."""
        out = {}
        for group, spec in self.weights.get("weights", {}).items():
            if isinstance(spec, dict):
                if spec.get("active", True):
                    out[group] = float(spec.get("value", 0.0))
            else:  # plain number
                out[group] = float(spec)
        return out


def load_categories(extra: list | None = None) -> list:
    """Shipped presets, then `category_system.extra` merged by id (a duplicate id wins,
    so local tuning always beats the preset)."""
    preset = {c["id"]: c for c in _load_yaml("categories.yaml").get("categories", [])
              if isinstance(c, dict) and "id" in c}
    for c in extra or []:
        if isinstance(c, dict) and "id" in c:
            preset[c["id"]] = c
    return list(preset.values())


@lru_cache(maxsize=1)
def get_config() -> Config:
    cfg = _load_yaml("config.yaml")
    weights = _load_yaml("weights.yaml")
    settings = cfg.get("category_system", {}) or {}
    return Config(
        audio=cfg.get("audio", {}),
        paths=cfg.get("paths", {}),
        features=cfg.get("features", {}),
        embedding=cfg.get("embedding", {}),
        vector_store=cfg.get("vector_store", {}),
        retrieval=cfg.get("retrieval", {}),
        logging=cfg.get("logging", {}),
        feature_version=cfg.get("feature_version", "v1"),
        weights=weights,
        categories=load_categories(settings.get("extra")),
        category_system=settings,
    )
