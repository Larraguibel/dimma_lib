"""The public surface of `metrics`, held to the same rule as the rest.

ADR-0004 for what the import line must say, ADR-0016 for the absences:
`roc_auc` and `accuracy` are barred everywhere, and anything else
needing an operating point from every module that owns none.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

from dimma import metrics

MODULES = [
    "_binning", "_inputs", "_ranked", "calibration", "decomposition",
    "operating_point", "ranking", "scoring",
]

PUBLIC_MODULES = [
    "calibration", "decomposition", "operating_point", "ranking", "scoring",
]

#: Modules that own no operating point, and so may offer no name needing one.
WITHOUT_AN_OPERATING_POINT = [m for m in MODULES if m != "operating_point"]

#: Modules that read no ranking, and so may offer no curve over every cut.
WITHOUT_A_RANKING = ["calibration", "decomposition", "scoring"]


def test_metrics_re_exports_nothing():
    assert metrics.__all__ == []


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_the_modules_resolve(name):
    assert importlib.import_module(f"dimma.metrics.{name}") is not None


@pytest.mark.parametrize("name", [
    "log_loss", "brier_score", "normalized_entropy", "reliability_curve",
    "expected_calibration_error", "calibration_ratio", "brier_decomposition",
    "log_loss_decomposition", "pr_curve", "best_f1_threshold", "confusion_at",
])
def test_no_function_is_reachable_from_the_package(name):
    assert not hasattr(metrics, name)


@pytest.mark.parametrize("name", ["roc_auc", "accuracy"])
def test_roc_auc_and_accuracy_are_offered_nowhere(name):
    """Barred in every module, including the ones that take a cut.

    Absent by decision. See ADR-0016.
    """
    for module in MODULES:
        assert not hasattr(importlib.import_module(f"dimma.metrics.{module}"), name)


@pytest.mark.parametrize("name", ["f1_score", "confusion_matrix", "best_threshold"])
def test_only_operating_point_offers_a_name_that_needs_a_cut(name):
    """Absent wherever the cut is not the module's own. See ADR-0016."""
    for module in WITHOUT_AN_OPERATING_POINT:
        assert not hasattr(importlib.import_module(f"dimma.metrics.{module}"), name)


@pytest.mark.parametrize("name", ["precision_recall_curve", "average_precision"])
def test_only_ranking_offers_the_curve_over_every_cut(name):
    """Absent wherever the order is not what the module reads.

    ADR-0016: these were barred by name under the old rule and are not
    operating-point metrics at all — `ranking` owns them.
    """
    for module in WITHOUT_A_RANKING:
        assert not hasattr(importlib.import_module(f"dimma.metrics.{module}"), name)


def test_no_metric_module_imports_jax():
    """These run on arrays already off the device, and say so.

    Read off the import statements rather than the module namespace, so
    that ``from jax import ...``, which binds no name a scan would
    recognise, is caught too.
    """
    for name in MODULES:
        module = importlib.import_module(f"dimma.metrics.{name}")
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                roots = {(node.module or "").split(".")[0]}
            else:
                continue
            assert "jax" not in roots, f"dimma.metrics.{name} imports jax"
