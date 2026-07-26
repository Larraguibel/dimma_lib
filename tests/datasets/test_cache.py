"""Where a downloaded dataset lands, and who gets to decide.

Every test redirects the cache root at ``tmp_path``: a test suite that
writes into the user's real cache directory is a test suite that behaves
differently on the second run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dimma.datasets._cache import get_cache_dir


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Point every resolution branch at ``tmp_path``."""
    monkeypatch.delenv("DIMMA_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    return tmp_path


def test_returns_an_existing_directory(isolated_env):
    path = get_cache_dir()
    assert isinstance(path, Path)
    assert path.is_dir()


def test_subdir_is_created(isolated_env):
    path = get_cache_dir("datasets")
    assert path.is_dir()
    assert path.name == "datasets"
    assert path.parent.name == "dimma"


def test_dimma_home_wins_over_the_os_default(isolated_env, tmp_path, monkeypatch):
    override = tmp_path / "elsewhere"
    monkeypatch.setenv("DIMMA_HOME", str(override))
    assert get_cache_dir("datasets") == (override / "datasets").resolve()


def test_idempotent(isolated_env):
    assert get_cache_dir("datasets") == get_cache_dir("datasets")


def test_path_is_absolute(isolated_env, monkeypatch):
    monkeypatch.setenv("DIMMA_HOME", ".")
    assert get_cache_dir().is_absolute()
