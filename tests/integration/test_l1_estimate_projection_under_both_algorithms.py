"""The estimate-side projection, both private loops, neither aware of it.

`tests/transforms` covers the wrapper against the seam's contract; this
file passes the same wrapped optimizer through both training loops. The
estimate leaves no direct trace in what `train` returns, so the checks
work the ball's two ends: a zero radius zeroes every estimate and pins
the trajectory to its start, and a radius no estimate reaches leaves an
equal-seed run indistinguishable from the unwrapped one. A middle
radius must then differ from both, or the projection never bound.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dimma.algorithms.dp_sgd import train as dp_sgd_train
from dimma.algorithms.spiderboost import train as spiderboost_train
from dimma.core import updates
from dimma.models.logreg import init_params
from dimma.models.losses import per_sample_bce_loss
from dimma.transforms.projection import l1_projected_estimate

from tests.helpers import tree_allclose

N_FEATURES = 5
N_ROWS = 1000


@pytest.fixture
def problem():
    gen = np.random.default_rng(0)
    x = gen.standard_normal((N_ROWS, N_FEATURES)).astype(np.float32)
    w_true = np.array([1.5, -0.5, 2.0, 0.0, -1.0], dtype=np.float32)
    logits = x @ w_true
    y = gen.uniform(size=N_ROWS) < 1.0 / (1.0 + np.exp(-logits))
    return jnp.asarray(x), jnp.asarray(y, dtype=jnp.float32)


def initial():
    return init_params(jax.random.key(0), N_FEATURES)


def run_dp_sgd(problem, optimizer):
    x, y = problem
    return dp_sgd_train.train(
        per_sample_bce_loss, initial(), optimizer, x, y,
        jax.random.key(1), np.random.default_rng(2),
        steps=30, expected_batch_size=100, clip_norm=1.0,
        noise_multiplier=0.5,
    )


def run_spiderboost(problem, optimizer):
    x, y = problem
    return spiderboost_train.train(
        per_sample_bce_loss, initial(), optimizer, x, y,
        jax.random.key(1), np.random.default_rng(2),
        steps=30, anchor_interval=5, anchor_expected_batch_size=100,
        variation_expected_batch_size=50, anchor_noise_scale=0.05,
        variation_noise_rate=0.1, variation_noise_cap=0.1,
    )


def test_a_zero_radius_pins_dp_sgd_to_its_start(problem):
    """Every estimate projects to the origin, so no step moves."""
    trained = run_dp_sgd(problem, l1_projected_estimate(updates.sgd(0.5), 0.0))
    assert tree_allclose(trained, initial())


def test_a_zero_radius_pins_spiderboost_to_its_start(problem):
    output, final = run_spiderboost(
        problem, l1_projected_estimate(updates.sgd(0.5), 0.0)
    )
    assert tree_allclose(output, initial())
    assert tree_allclose(final, initial())


def test_a_radius_no_estimate_reaches_changes_nothing(problem):
    """The control at the other end: equal seeds, unwrapped against a
    ball the estimates never leave, on both loops."""
    bare = run_dp_sgd(problem, updates.sgd(0.5))
    wrapped = run_dp_sgd(
        problem, l1_projected_estimate(updates.sgd(0.5), 1e6)
    )
    assert tree_allclose(bare, wrapped)

    bare_out, bare_final = run_spiderboost(problem, updates.sgd(0.5))
    out, final = run_spiderboost(
        problem, l1_projected_estimate(updates.sgd(0.5), 1e6)
    )
    assert tree_allclose(bare_out, out)
    assert tree_allclose(bare_final, final)


def test_a_binding_radius_alters_both_runs(problem):
    """Between the two ends the projection must actually bind."""
    radius = 0.05
    bare = run_dp_sgd(problem, updates.sgd(0.5))
    wrapped = run_dp_sgd(
        problem, l1_projected_estimate(updates.sgd(0.5), radius)
    )
    assert not tree_allclose(bare, wrapped)
    assert not tree_allclose(wrapped, initial())

    _, bare_final = run_spiderboost(problem, updates.sgd(0.5))
    _, final = run_spiderboost(
        problem, l1_projected_estimate(updates.sgd(0.5), radius)
    )
    assert not tree_allclose(bare_final, final)
    assert not tree_allclose(final, initial())
