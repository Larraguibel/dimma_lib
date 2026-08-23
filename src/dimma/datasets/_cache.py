"""OS-aware cache directory resolution for dimma datasets."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def get_cache_dir(subdir: str = "") -> Path:
    """Return an absolute path under the dimma cache root, creating it.

    ``DIMMA_HOME`` overrides the OS-appropriate user cache directory.
    """
    override = os.environ.get("DIMMA_HOME")
    if override:
        root = Path(override).expanduser()
    elif sys.platform == "darwin":
        root = Path("~/Library/Caches/dimma").expanduser()
    elif sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "~/AppData/Local")
        root = (Path(local) / "dimma" / "Cache").expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME", "~/.cache")
        root = (Path(xdg) / "dimma").expanduser()

    path = root / subdir if subdir else root
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()
