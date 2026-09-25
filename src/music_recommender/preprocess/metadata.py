"""Read a file's own embedded tags (ID3 / Vorbis comment / MP4 atom) with mutagen.

This is the one place where non-audio information enters the system, and it is
deliberately kept separate from feature extraction: tags are the file's *claim*
about itself, not a measurement of its sound (spec 81/83 honesty rules). Every
value returned here is stored together with `meta_source='tag'` so a consumer can
tell tag data apart from MIR output. Nothing is guessed: a missing tag stays None
rather than being parsed out of the filename.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..utils.logging import get_logger

log = get_logger()

# easy-mode keys across ID3/Vorbis/MP4, in decreasing trust order
_KEYS = {
    "title": ("title",),
    "artist": ("artist",),
    "album": ("album",),
    "albumartist": ("albumartist",),
    "language": ("language", "lang"),
    "genre": ("genre",),
}

_YEAR_RE = re.compile(r"(1[0-9]{3}|20[0-9]{2}|30[0-9]{2})")


def _first_str(tags: dict, names: tuple[str, ...]) -> str | None:
    for name in names:
        val = tags.get(name)
        if isinstance(val, (list, tuple)):
            val = val[0] if val else None
        if val is not None:
            text = str(val).strip()
            if text:
                return text
    return None


def _year(value: str | None) -> int | None:
    if not value:
        return None
    m = _YEAR_RE.search(value)
    return int(m.group(1)) if m else None


def _plain(tags: dict) -> dict:
    """mutagen's EasyDict-like object -> {key: [str, ...]}, dropping binary frames."""
    out = {}
    for k, v in (tags.items() if hasattr(tags, "items") else []):
        if isinstance(v, (list, tuple)):
            vals = [str(x) for x in v if isinstance(x, (str, bytes)) and not _looks_binary(x)]
            if vals:
                out[str(k).lower()] = vals
    return out


def _looks_binary(v) -> bool:
    if isinstance(v, bytes):
        return b"\x00" in v or not _decodable(v)
    return "\x00" in str(v)


def _decodable(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def read_tags(file_path: Path | str) -> dict:
    """Return {title, artist, album, year, language, genre, meta_source}.

    Values are None when the file carries no such tag; `meta_source` is 'tag' when
    at least one value came from the file, 'none' otherwise. Unsupported formats,
    unreadable files and a missing mutagen all yield 'none' rather than an error,
    because tags are optional context and must never block analysis.
    """
    empty = {"title": None, "artist": None, "album": None, "year": None,
             "language": None, "genre": None, "meta_source": "none"}
    p = Path(file_path)
    try:
        from mutagen import File as MutagenFile  # optional dependency, imported lazily
    except ImportError:  # pragma: no cover - environment without mutagen
        log.warning("mutagen not installed; embedded tags will be ignored")
        return empty

    try:
        f = MutagenFile(str(p), easy=True)
    except Exception as exc:  # noqa: BLE001 - a broken tag block must not kill the scan
        log.warning("tag read failed for %s: %s", p.name, exc)
        return empty
    if f is None or not getattr(f, "tags", None):
        return empty

    tags = _plain(f.tags)
    artist = _first_str(tags, _KEYS["artist"]) or _first_str(tags, _KEYS["albumartist"])
    out = {
        "title": _first_str(tags, _KEYS["title"]),
        "artist": artist,
        "album": _first_str(tags, _KEYS["album"]),
        "year": _year(_first_str(tags, ("date", "originaldate", "year"))),
        "language": _first_str(tags, _KEYS["language"]),
    }
    out["meta_source"] = "tag" if any(v is not None for v in out.values()) else "none"
    return out
