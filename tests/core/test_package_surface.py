"""The public surface of `core`, as decided rather than as it happens to be.

`core` re-exports no functions: a caller imports from the stage module,
so the import line says which stage the call belongs to. Nothing enforces
that at runtime, so it is pinned here — a convenience re-export added
later fails these tests rather than quietly widening a surface the README
commits to keeping stable.
"""

from __future__ import annotations

import importlib

import pytest

import dimma
from dimma import core
from dimma.core import sampling

STAGE_MODULES = [
    "aggregation",
    "clipping",
    "gradients",
    "noise",
    "projection",
    "pytree",
    "sampling",
    "updates",
]


def test_core_exports_modules_only():
    assert set(core.__all__) == set(STAGE_MODULES)


@pytest.mark.parametrize("name", STAGE_MODULES)
def test_each_listed_module_resolves(name):
    assert importlib.import_module(f"dimma.core.{name}") is getattr(core, name)


@pytest.mark.parametrize("name", [
    "per_sample_clip", "per_sample_norms", "add_gaussian", "add_laplace",
    "sum_over_batch", "average_over_batch", "per_sample_grads", "batch_grads",
    "project_l1_ball", "project_l1_ball_pytree", "global_norm",
])
def test_core_does_not_re_export_stage_functions(name):
    assert not hasattr(core, name)


def test_sampling_exports_each_sampler_separately():
    """Separate modules, so the choice is visible in the import line.

    Three samplers, two of them mechanisms: `shuffled` is the ordinary
    draw the non-private baselines take, and no accounting is stated
    against it at all.
    """
    assert set(sampling.__all__) == {
        "poisson", "poisson_truncated", "shuffled",
    }


@pytest.mark.parametrize("name", ["subsample", "padded_batch_size", "batches"])
def test_sampling_does_not_flatten_the_samplers(name):
    """Flattening would hide which mechanism the accounting applies to,
    and whether any does."""
    assert not hasattr(sampling, name)


def test_top_level_package_re_exports_nothing():
    assert dimma.__all__ == []


def test_version_is_exposed():
    assert dimma.__version__ == "0.1.0"
