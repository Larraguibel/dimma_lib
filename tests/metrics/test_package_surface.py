"""The public surface of `metrics`, held to the same rule as the rest.

ADR-0004: a caller writes `from dimma.metrics.scoring import log_loss`,
so the import line says which question the number answers. Nothing
enforces that at runtime, so it is pinned here.

The second half pins an absence. A metrics package is where `accuracy`
and `f1_score` arrive by default, and they are missing on purpose:
both need an operating point, and this task has none to give them. A
later convenience import fails a test rather than quietly putting a
threshold back into a comparison that was built not to need one.
"""

from __future__ import annotations

import importlib

import pytest

from dimma import metrics

MODULES = ["_binning", "_inputs", "calibration", "decomposition", "scoring"]


def test_metrics_re_exports_nothing():
    assert metrics.__all__ == []


@pytest.mark.parametrize("name", ["calibration", "decomposition", "scoring"])
def test_the_modules_resolve(name):
    assert importlib.import_module(f"dimma.metrics.{name}") is not None


@pytest.mark.parametrize("name", [
    "log_loss", "brier_score", "normalized_entropy", "reliability_curve",
    "expected_calibration_error", "calibration_ratio", "brier_decomposition",
    "log_loss_decomposition",
])
def test_no_function_is_reachable_from_the_package(name):
    assert not hasattr(metrics, name)


@pytest.mark.parametrize("name", [
    "accuracy", "f1_score", "confusion_matrix", "precision_recall_curve",
    "roc_auc", "average_precision", "best_threshold",
])
def test_nothing_that_needs_an_operating_point_is_offered(name):
    """Absent by decision. See `dimma.metrics` for the reasoning."""
    for module in ("scoring", "calibration", "decomposition"):
        assert not hasattr(importlib.import_module(f"dimma.metrics.{module}"), name)


def test_no_metric_module_imports_jax():
    """These run on arrays already off the device, and say so.

    Importing JAX here would make a reporting path depend on the
    accelerator stack, and would invite someone to make a metric
    differentiable — which is how a threshold-free score turns into a
    training objective nobody chose.
    """
    import sys

    for name in MODULES:
        module = importlib.import_module(f"dimma.metrics.{name}")
        assert "jax" not in {
            n.split(".")[0] for n in getattr(module, "__dict__", {})
            if isinstance(sys.modules.get(n), type(sys))
        }
        assert not hasattr(module, "jnp")
