"""OS-aware cache directory resolution for dimma datasets."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def get_cache_dir(subdir: str = "") -> Path:
    """Return the dimma cache directory, creating it if needed.

    Resolution order:

    1. ``DIMMA_HOME``, if set in the environment.
    2. Otherwise the OS-appropriate user cache directory:

       - Linux: ``$XDG_CACHE_HOME/dimma`` or ``~/.cache/dimma``
       - macOS: ``~/Library/Caches/dimma``
       - Windows: ``%LOCALAPPDATA%\\dimma\\Cache``

    Parameters
    ----------
    subdir : str, optional
        Subdirectory under the cache root, e.g. ``"datasets"``.

    Returns
    -------
    pathlib.Path
        Absolute path to the directory, which exists on return.
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
