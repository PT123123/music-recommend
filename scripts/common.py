"""Shared script bootstrap: make `src/music_recommender` importable and set up paths."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from music_recommender.utils.config import get_config  # noqa: E402
from music_recommender.utils.logging import get_logger  # noqa: E402


def track_label(row, fallback: str) -> str:
    """Readable CLI label for a track row: the file's own tag when present, else
    its filename. Tag text is only a label — provenance stays in `meta_source`,
    so nothing tag-derived is presented as MIR output.
    """
    if row is None:
        return fallback
    title, artist = row["title"], row["artist"]
    if title and artist:
        return f"{artist} - {title}"
    if title:
        return str(title)
    from pathlib import Path as _P
    return _P(row["file_path"]).name if row["file_path"] else fallback
