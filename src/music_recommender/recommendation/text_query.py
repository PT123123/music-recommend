"""Free Chinese text -> a Music Space category query.

No NLP model and no vocabulary pretending to be audio analysis: a word counts only if
the shared lexicon can back it with a measured dimension (`lexicon.py`). Everything
else is returned as an unmatched term with a reason, so the caller can say "this part
I cannot answer from your files" instead of quietly mapping it to something unrelated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..utils.logging import get_logger
from .lexicon import (BY_COL, DIM_MAP, GENRE_TERMS, LANGUAGE_TERMS, VOCAL_PROXY_TERMS,
                      WORDS_HIGH, WORDS_LOW, match_terms)
from .space import MusicSpace

log = get_logger()

_NEGATIONS = ("不要", "别", "不想", "没有", "去掉", "不是")
_FILLER = re.compile(r"[\s,，。.!！?？、〜~·…]*(?:来点|来首|来一支|放点|放首|听点|听首|我想|我要|帮我|推荐|一些|一点|一首|首歌|的歌|的音|谢谢|请|的)+[\s,，。.!！?？、〜~·…]*")
# only connective particles: never split inside a word (柔和, 和谐)
_RESIDUE_SPLIT = re.compile(r"[\s,，。.!！?？、〜~·…/|的又而是]+")
_RESIDUE_NOISE = re.compile(r"^(?:很|太|非常|超级|超|比较|挺|更|最|有点|有些|稍微|特别|不太|略微)+"
                            r"|(?:一点|一些|点儿|一些些|点)+$")


@dataclass
class TextQueryConfig:
    target_high: float = 0.85
    target_low: float = 0.15
    # a dimension already set by a stronger term is not overwritten by a later one
    keep_first: bool = True

    @classmethod
    def from_settings(cls, section: dict | None) -> "TextQueryConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (section or {}).items() if k in known})


def parse(text: str, space: MusicSpace, tcfg: TextQueryConfig | None = None) -> dict:
    """Return {query, hard_filter, matched, unmatched, cleaned, *_proxies} for one request."""
    tcfg = tcfg or TextQueryConfig()
    text = text or ""
    query: dict[str, float] = {}
    hard: dict = {}
    matched: list[dict] = []
    unmatched: list[dict] = []
    proxies: list[dict] = []

    for word, col, direction in match_terms(text):
        field = _field_for(col)
        if field is None:
            unmatched.append({"term": word, "reason": "no-lexicon-entry"})
            continue
        if col not in space.sorted_vals:
            unmatched.append({"term": word, "reason": "dimension-not-in-library", "dimension": col})
            continue
        if tcfg.keep_first and field in query:
            continue
        side = direction > 0
        target = tcfg.target_high if side else tcfg.target_low
        if _negated(text, word):
            target = tcfg.target_low if side else tcfg.target_high
            side = not side
        query[field] = target
        matched.append({"term": word, "dimension": col, "field": field,
                        "side": "high" if side else "low",
                        "word": WORDS_HIGH.get(col) if side else WORDS_LOW.get(col)})

    for term, lang in LANGUAGE_TERMS.items():
        if term not in text:
            continue
        if lang is None:  # 纯音乐 / 器乐
            hard["instrumental"] = True
            matched.append({"term": term, "dimension": "has_vocal", "side": "low", "word": "无人声"})
            continue
        missing = sum(1 for r in space.rows.values() if not r.get("language"))
        if missing == len(space.rows):
            unmatched.append({"term": term, "reason": "no-language-tag-in-library"})
        else:
            hard["language_is"] = lang
            matched.append({"term": term, "dimension": "language", "side": "tag",
                            "word": f"语种 {lang}", "tag_missing_tracks": missing})
    for term, (tag, proxy) in GENRE_TERMS.items():
        if term not in text:
            continue
        missing = sum(1 for r in space.rows.values() if not r.get("genre"))
        if missing != len(space.rows):
            hard["genre_is"] = tag
            matched.append({"term": term, "dimension": "genre", "side": "tag",
                            "word": f"曲风 {tag}", "tag_missing_tracks": missing})
            continue
        used = {f: t for f, t in proxy.items() if DIM_MAP.get(f) in space.sorted_vals}
        if not used:
            unmatched.append({"term": term, "reason": "no-genre-tag-in-library"})
            continue
        # No tag to quote, so fall back to the acoustic shape this word usually has.
        # Flagged as a proxy: the answer describes sound, never claims a genre.
        for f, t in used.items():
            query.setdefault(f, t)
        proxies.append({"term": term, "genre": tag, "dims": sorted(used)})
        matched.append({"term": term, "dimension": "genre-proxy", "side": "proxy",
                        "word": f"{term}（按听感近似，非曲风标签）"})

    pitch: list[dict] = []
    for term, (label, dims) in VOCAL_PROXY_TERMS.items():
        if term not in text:
            continue
        used = {f: t for f, t in dims.items() if DIM_MAP.get(f) in space.sorted_vals}
        if not used:
            unmatched.append({"term": term, "reason": "no-vocal-dimensions-in-library"})
            continue
        # No feature in this project identifies a singer's gender, so the word maps to
        # the register it usually stands for and is labelled as an approximation.
        for f, t in used.items():
            query.setdefault(f, t)
        pitch.append({"term": term, "guess": label, "dims": sorted(used)})
        matched.append({"term": term, "dimension": "vocal-pitch-proxy", "side": "proxy",
                        "word": f"{term}（{label}，不识别性别）"})

    cleaned = _clean(text, matched, unmatched)
    for term in _residue_terms(cleaned, {u["term"] for u in unmatched}):
        unmatched.append({"term": term, "reason": "unrecognized-descriptor"})
    return {"query": query, "hard_filter": hard, "matched": matched,
            "unmatched": unmatched, "cleaned": cleaned, "genre_proxies": proxies,
            "vocal_proxies": pitch, "usable_dims": len(query)}


def _field_for(col: str) -> str | None:
    spec = BY_COL.get(col)
    return spec.fields[0] if spec else None


def _negated(text: str, word: str) -> bool:
    idx = text.find(word)
    return idx > 0 and any(n in text[max(0, idx - 4):idx] for n in _NEGATIONS)


def _clean(text: str, matched: list[dict], unmatched: list[dict]) -> str:
    """What the caller typed with every understood term and filler removed."""
    rest = _FILLER.sub(" ", text)
    for m in matched:
        for term in (m["term"], m.get("word", "")):
            if term:
                rest = rest.replace(term, " ")
    seen = {m["term"] for m in matched}
    for u in unmatched:
        if u["term"] not in seen:
            rest = rest.replace(u["term"], " ")
    for word in _NEGATIONS:  # function words that only modulate, never describe
        rest = rest.replace(word, " ")
    return re.sub(r"\s+", " ", rest).strip()


def _residue_terms(cleaned: str, already: set[str]) -> list[str]:
    """Fragments left after everything the lexicon understands: words it cannot answer.

    Splitting on particles keeps a report like "很治愈的高级感" as two honest refusals
    instead of one opaque blob; degree adverbs are stripped so the term echoes back
    what the caller actually meant.
    """
    out = []
    for frag in _RESIDUE_SPLIT.split(cleaned):
        term = _RESIDUE_NOISE.sub("", frag).strip()
        if len(term) >= 2 and term not in already:
            already.add(term)
            out.append(term)
    return out
