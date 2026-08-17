"""The public surface of `transforms`, as decided rather than as it happens
to be.

Same rule as `core` (ADR-0004): no functions are re-exported, so the
import line names the transform being applied.
"""

from __future__ import annotations

import importlib

import pytest

from dimma import transforms

TRANSFORM_MODULES = [
    "projection",
]


def test_transforms_exports_modules_only():
    assert set(transforms.__all__) == set(TRANSFORM_MODULES)


@pytest.mark.parametrize("name", TRANSFORM_MODULES)
def test_each_listed_module_resolves(name):
    assert importlib.import_module(f"dimma.transforms.{name}") is \
        getattr(transforms, name)


@pytest.mark.parametrize("name", ["l1_projected"])
def test_transforms_does_not_re_export_functions(name):
    assert not hasattr(transforms, name)
