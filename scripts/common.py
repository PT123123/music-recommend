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


def display_name(db_path: str, track_row) -> str:
    """Prefer a readable filename for CLI output."""
    from pathlib import Path as _P
    return _P(track_row["file_path"]).name if track_row and track_row["file_path"] else db_path
