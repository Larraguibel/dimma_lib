"""The digest is what makes a dataset a fixed input.

The three outcomes: a matching cached file is never re-fetched, a
corrupted one is replaced, and a mismatched download never lands at the
destination path.
"""

from __future__ import annotations

import hashlib
from unittest import mock

import pytest

from dimma.datasets._download import download_with_checksum


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stub_response(payload: bytes):
    """A context-manager stub yielding ``payload`` then EOF."""
    resp = mock.MagicMock()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = None
    resp.read.side_effect = [payload, b""]
    return resp


def test_matching_cached_file_is_not_refetched(tmp_path):
    dest = tmp_path / "f.bin"
    data = b"hello world"
    dest.write_bytes(data)

    with mock.patch("urllib.request.urlopen", side_effect=AssertionError):
        out = download_with_checksum("https://example.invalid/f", dest, sha256(data))

    assert out == dest
    assert out.read_bytes() == data


def test_digest_comparison_is_case_insensitive(tmp_path):
    dest = tmp_path / "f.bin"
    data = b"hello world"
    dest.write_bytes(data)

    with mock.patch("urllib.request.urlopen", side_effect=AssertionError):
        download_with_checksum("https://example.invalid/f", dest, sha256(data).upper())


def test_corrupted_cached_file_is_replaced(tmp_path):
    dest = tmp_path / "f.bin"
    dest.write_bytes(b"bad cached")
    good = b"good fresh data"

    with mock.patch("urllib.request.urlopen", return_value=stub_response(good)):
        out = download_with_checksum("https://example.invalid/f", dest, sha256(good))

    assert out.read_bytes() == good


def test_mismatch_raises_and_names_both_digests(tmp_path):
    dest = tmp_path / "f.bin"
    payload = b"actual bytes"
    wrong = sha256(b"different content")

    with mock.patch("urllib.request.urlopen", return_value=stub_response(payload)):
        with pytest.raises(RuntimeError) as excinfo:
            download_with_checksum("https://example.invalid/f", dest, wrong)

    message = str(excinfo.value)
    assert wrong in message
    assert sha256(payload) in message


def test_mismatched_bytes_never_reach_the_destination(tmp_path):
    """Otherwise the next run finds a file and trusts it."""
    dest = tmp_path / "f.bin"
    payload = b"actual bytes"

    with mock.patch("urllib.request.urlopen", return_value=stub_response(payload)):
        with pytest.raises(RuntimeError):
            download_with_checksum(
                "https://example.invalid/f", dest, sha256(b"something else")
            )

    assert not dest.exists()
    assert dest.with_suffix(".bin.partial").exists()


def test_success_leaves_no_partial_file(tmp_path):
    dest = tmp_path / "f.bin"
    payload = b"good"

    with mock.patch("urllib.request.urlopen", return_value=stub_response(payload)):
        download_with_checksum("https://example.invalid/f", dest, sha256(payload))

    assert dest.read_bytes() == payload
    assert not dest.with_suffix(".bin.partial").exists()
