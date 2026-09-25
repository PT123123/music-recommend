"""Audio preprocessing: convert MP3/WAV to normalized 44.1kHz mono WAV (spec section 3)."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import librosa
import soundfile as sf

from ..utils.config import get_config
from ..utils.logging import get_logger

log = get_logger()


def _ffmpeg_bin() -> str | None:
    return shutil.which("ffmpeg")


_CHUNK = 256 * 1024


def track_id_for(file_path: Path) -> str:
    """Stable id derived from file *content*, not path.

    A path hash made renaming or reorganising a folder silently orphan every
    interaction/evaluation row for that track (spec 49/56 history becomes
    unreachable). Size + first/last byte blocks is stable across moves and cheap
    to compute; re-encoded files correctly get a new identity.
    Falls back to the path if the file cannot be read.
    """
    p = Path(file_path)
    h = hashlib.sha1()
    try:
        size = p.stat().st_size
        h.update(b"v2:" + str(size).encode())
        with open(p, "rb") as fh:
            h.update(fh.read(_CHUNK))
            if size > 2 * _CHUNK:
                fh.seek(-_CHUNK, os.SEEK_END)
                h.update(fh.read(_CHUNK))
    except OSError:
        h.update(b"path:" + str(p.resolve()).encode())
    return h.hexdigest()[:16]


def file_signature(file_path: Path) -> tuple[int, float]:
    """(size, mtime) used to skip unchanged files on re-scan."""
    st = Path(file_path).stat()
    return int(st.st_size), float(st.st_mtime)


def normalize(file_path: Path, out_dir: Path | None = None) -> tuple[Path, float, int]:
    """Return (normalized_wav_path, duration_seconds, sample_rate).

    Prefers ffmpeg for lossy formats (MP3); falls back to librosa/soundfile
    which also decodes mp3 via audioread/ffmpeg when available.
    """
    cfg = get_config()
    sr = int(cfg.audio.get("sample_rate", 44100))
    out_dir = Path(out_dir or cfg.normalized_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tid = track_id_for(file_path)
    out_path = out_dir / f"{tid}.wav"

    ext = file_path.suffix.lower()
    if ext == ".mp3" and _ffmpeg_bin():
        if not out_path.exists():
            cmd = [_ffmpeg_bin(), "-y", "-i", str(file_path),
                   "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(out_path)]
            log.debug("ffmpeg %s", " ".join(cmd[1:]))
            subprocess.run(cmd, check=True, capture_output=True)
        y, sr_out = sf.read(str(out_path))
        duration = len(y) / sr_out
        return out_path, float(duration), int(sr_out)

    # WAV (or fallback decode path)
    y, sr_out = librosa.load(str(file_path), sr=sr, mono=True)
    sf.write(str(out_path), y, sr_out)
    return out_path, float(len(y) / sr_out), int(sr_out)
