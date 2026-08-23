"""The public surface of `models`, as decided rather than as inherited.

The package this was ported from re-exported its whole surface from
`__init__` and aliased the colliding names; ADR-0004 rules that out, so
it is pinned here rather than left to the path of least resistance.
"""

from __future__ import annotations

import importlib

import pytest

from dimma import models


def test_models_re_exports_nothing():
    assert models.__all__ == []


@pytest.mark.parametrize("name", ["logreg", "losses"])
def test_the_modules_resolve(name):
    assert importlib.import_module(f"dimma.models.{name}") is not None


@pytest.mark.parametrize("name", [
    "init_params", "forward", "forward_sparse", "per_sample_bce_loss",
    "batch_bce_loss",
    "hashed_init_params", "hashed_forward", "hash_buckets",
    "per_sample_hashed_bce_loss", "MLP",
])
def test_no_function_is_reachable_from_the_package(name):
    assert not hasattr(models, name)


@pytest.mark.parametrize("name", ["mlp", "hashed_logreg"])
def test_the_dropped_models_did_not_come_along(name):
    """Neither is ported: the MLP by decision, the hashed model earlier."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"dimma.models.{name}")
