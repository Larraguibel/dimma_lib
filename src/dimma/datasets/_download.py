"""Idempotent download with SHA256 verification.

A pinned digest is what makes a dataset a fixed input rather than
whatever the upstream URL served today: two runs that agree on the digest
were trained on the same bytes.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

_CHUNK = 1024 * 1024


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_checksum(
    url: str,
    destination: Path,
    expected_sha256: str,
    chunk_size: int = _CHUNK,
) -> Path:
    """Download ``url`` to ``destination``, verifying its SHA256.

    Idempotent: on return ``destination`` holds bytes whose digest is
    ``expected_sha256``, and a file already there with that digest is
    left alone rather than fetched again. Nothing partial is ever left
    at ``destination`` itself.

    Parameters
    ----------
    url : str
        HTTPS URL of the file to download.
    destination : Path
        Final on-disk path. Its parent directory must already exist.
    expected_sha256 : str
        64-character hex digest, compared case-insensitively.
    chunk_size : int, default 1048576
        Streaming chunk size, in bytes.

    Returns
    -------
    Path
        The verified ``destination``.

    Raises
    ------
    RuntimeError
        If the downloaded bytes do not match ``expected_sha256``. The
        partial file is left in place, and the message names both
        digests.

    Notes
    -----
    Uses the standard library only, and does not retry: a network failure
    surfaces as the underlying ``URLError`` for the caller to retry.
    """
    expected = expected_sha256.lower()
    destination = Path(destination)

    if destination.exists():
        if _compute_sha256(destination) == expected:
            return destination
        destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".partial")
    h = hashlib.sha256()
    with urllib.request.urlopen(url) as resp, open(partial, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"Downloaded file SHA256 mismatch for {url}.\n"
            f"Expected: {expected}\n"
            f"Got: {actual}\n"
            f"Partial file kept at: {partial}"
        )
    partial.rename(destination)
    return destination
