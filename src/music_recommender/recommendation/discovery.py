"""Unsupervised category discovery: cluster the Music Space, then name the clusters.

The fixed categories in `categories.yaml` can only ever express what a human thought
to ask for. This module asks the library itself: KMeans over the same per-library
percentiles the category engine already ranks on, then each cluster centroid is turned
into a Chinese name by taking the dimensions that deviate most from the library middle
(0.5) and reading their words out of the shared lexicon.

Consequences worth stating:
  * Discovered targets are *library-relative by construction*, so a 93-track library
    and a 5000-track library both get meaningful regions without retuning numbers.
  * Because percentiles depend on the current library (ADR-2), cluster ids and names
    are only stable for one library version. They are cached on the MusicSpace and
    recomputed when the library fingerprint changes.
  * Language/genre may be appended to a name **only** from the file's own tag
    (ADR-14), and only when that tag is both well covered and near-unanimous inside
    the cluster. Otherwise the tag is not mentioned at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.logging import get_logger
from .lexicon import BY_COL, FIELD_BY_COL, LEXICON, name_from_deviations
from .space import MusicSpace

log = get_logger()

# binary fact, ranked by mean not percentile - it drives the instrumental filter
PRESENCE_COL = "has_vocal"


@dataclass
class DiscoveryConfig:
    enabled: bool = True
    k_min: int = 4
    k_max: int = 12
    min_cluster_size: int = 4
    min_cluster_share: float = 0.04
    name_min_deviation: float = 0.15
    name_max_terms: int = 3
    neutral_impute: float = 0.5
    presence_split: float = 0.25
    tag_min_coverage: float = 0.6
    tag_min_purity: float = 0.7
    random_state: int = 0

    @classmethod
    def from_settings(cls, section: dict | None) -> "DiscoveryConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (section or {}).items() if k in known})


def _matrix(space: MusicSpace, cols: list[str], impute: float) -> tuple[list[str], np.ndarray, int]:
    """Percentile matrix over `cols`; NaN (e.g. vocal dims on instrumentals) imputed
    to the library middle so missingness cannot masquerade as a feature."""
    ids = sorted(space.rows)
    X = np.full((len(ids), len(cols)), np.nan)
    for i, tid in enumerate(ids):
        p = space.pct.get(tid, {})
        for j, c in enumerate(cols):
            if c in p:
                X[i, j] = p[c]
    missing = int(np.isnan(X).any(axis=1).sum())
    return ids, np.nan_to_num(X, nan=float(impute)), missing


def discover(space: MusicSpace, dcfg: DiscoveryConfig | None = None) -> tuple[list[dict], dict]:
    """Return (category specs compatible with categories.yaml, meta)."""
    dcfg = dcfg or DiscoveryConfig()
    meta: dict = {"status": "ok", "library_size": len(space.rows), "k": 0, "silhouette": None}
    if not dcfg.enabled:
        meta["status"] = "disabled"
        return [], meta
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError as exc:  # pragma: no cover
        meta.update(status="no-sklearn", detail=str(exc))
        return [], meta

    feature_cols = sorted(space.sorted_vals)
    if not feature_cols:
        meta["status"] = "no-percentile-features"
        return [], meta
    ids, X, imputed = _matrix(space, feature_cols, dcfg.neutral_impute)
    n = len(ids)
    meta.update(features=len(feature_cols), rows_imputed=imputed)

    min_size = max(dcfg.min_cluster_size, int(round(dcfg.min_cluster_share * n)))
    hi = min(dcfg.k_max, max(2, n // min_size))
    meta["min_cluster_size"] = min_size
    if n < 2 * min_size or hi < dcfg.k_min:
        # too small to cluster honestly: report it instead of emitting 1-track "categories"
        meta["status"] = "library-too-small"
        return [], meta

    best = None
    for k in range(dcfg.k_min, hi + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=dcfg.random_state).fit_predict(X)
        sizes = np.bincount(labels, minlength=k)
        if sizes.min() < min_size:
            continue
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(X, labels))
        if best is None or score > best[0]:
            best = (score, k, labels)
    if best is None:
        meta["status"] = "no-k-satisfied-min-cluster-size"
        return [], meta

    score, k, labels = best
    meta.update(k=k, silhouette=round(score, 4))

    has_presence = PRESENCE_COL in space.rows[ids[0]]
    specs = []
    for c in range(k):
        member_ids = [ids[i] for i in range(n) if labels[i] == c]
        centroid = X[labels == c].mean(axis=0)
        devs = sorted(
            ((feature_cols[j], float(centroid[j]) - 0.5)
             for j in range(len(feature_cols))
             if feature_cols[j] in BY_COL and abs(centroid[j] - 0.5) >= dcfg.name_min_deviation),
            key=lambda t: abs(t[1]), reverse=True)
        picked_devs = devs[:dcfg.name_max_terms]
        name = name_from_deviations(picked_devs, dcfg.name_max_terms)

        hard: dict = {}
        if has_presence:
            presence = float(np.mean([bool(space.rows[t].get(PRESENCE_COL)) for t in member_ids]))
            if presence <= dcfg.presence_split:
                name = "无人声·" + name
                hard["instrumental"] = True
            elif presence >= 1.0 - dcfg.presence_split:
                hard["has_vocal"] = True
            meta.setdefault("presence_by_cluster", {})[f"auto-{c + 1:02d}"] = round(presence, 3)

        tag_note = _tag_suffix(space, member_ids, dcfg)
        if tag_note:
            name += f"（tag：{tag_note[0]}）"
            specs_note = tag_note[1]
        else:
            specs_note = None

        query = {FIELD_BY_COL[col]: round(0.5 + dev, 3) for col, dev in picked_devs}
        if not query:
            # a cluster with no dimension far from the middle still needs a rankable
            # region; use its two strongest (even sub-threshold) deviations
            top = sorted(((feature_cols[j], float(centroid[j]) - 0.5)
                          for j in range(len(feature_cols)) if feature_cols[j] in FIELD_BY_COL),
                         key=lambda t: abs(t[1]), reverse=True)[:2]
            query = {FIELD_BY_COL[col]: round(0.5 + dev, 3) for col, dev in top}
        specs.append({
            "id": f"auto-{c + 1:02d}",
            "name": name,
            "source": "discovered",
            "hard_filter": hard,
            "query": query,
            "member_track_ids": member_ids,
            "note": f"KMeans 簇（k={k}，轮廓系数 {round(score, 3)}），"
                    f"依据维度：{', '.join(LABEL_BY_COL.get(c, c) for c, _ in picked_devs) or '无显著偏离'}",
            "tag_evidence": specs_note,
        })
    specs.sort(key=lambda s: -len(s["member_track_ids"]))
    return specs, meta


LABEL_BY_COL = {d.col: d.label for d in LEXICON}


def _tag_suffix(space: MusicSpace, member_ids: list[str], dcfg: DiscoveryConfig) -> tuple[str, dict] | None:
    """Name the cluster after a language/genre tag only if the files themselves say so."""
    for field, label in (("language", "语种"), ("genre", "曲风")):
        vals = [str(space.rows[t].get(field)).strip() for t in member_ids
                if space.rows[t].get(field)]
        if len(vals) < dcfg.tag_min_coverage * max(1, len(member_ids)):
            continue
        top, count = max(_counts(vals).items(), key=lambda kv: kv[1])
        if count / len(vals) < dcfg.tag_min_purity:
            continue
        return f"{label} {top}", {"field": field, "value": top,
                                  "coverage": round(len(vals) / len(member_ids), 3),
                                  "purity": round(count / len(vals), 3)}
    return None


def _counts(vals: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in vals:
        out[v] = out.get(v, 0) + 1
    return out
