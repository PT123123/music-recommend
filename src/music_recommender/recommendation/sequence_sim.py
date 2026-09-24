"""Sequence similarity utilities (spec section 10).

Combines Levenshtein + n-gram + Jaccard + transition-pattern similarity so we do
not rely on Jaccard alone. All return [0, 1].
"""
from __future__ import annotations

from collections import Counter


def levenshtein_similarity(a: list, b: list) -> float:
    if not a and not b:
        return 1.0
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    dist = prev[n]
    return 1.0 - dist / max(m, n)


def ngram_similarity(a: list, b: list, n: int = 2) -> float:
    ga = Counter(tuple(a[i:i + n]) for i in range(len(a) - n + 1))
    gb = Counter(tuple(b[i:i + n]) for i in range(len(b) - n + 1))
    if not ga and not gb:
        return 1.0
    inter = sum((ga & gb).values())
    union = sum((ga | gb).values())
    return inter / union if union else 0.0


def jaccard_similarity(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def transition_similarity(a: list, b: list) -> float:
    ta = Counter(zip(a, a[1:]))
    tb = Counter(zip(b, b[1:]))
    if not ta and not tb:
        return 1.0
    inter = sum((ta & tb).values())
    union = sum((ta | tb).values())
    return inter / union if union else 0.0


def combined_sequence_similarity(a: list, b: list, weights=(0.35, 0.25, 0.2, 0.2)) -> float:
    if not a or not b:
        return 0.0
    lev = levenshtein_similarity(a, b)
    ng = ngram_similarity(a, b)
    jac = jaccard_similarity(a, b)
    tr = transition_similarity(a, b)
    return float(weights[0] * lev + weights[1] * ng + weights[2] * jac + weights[3] * tr)
