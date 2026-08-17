"""One transform, both private loops, neither aware of it.

`tests/transforms` covers the wrapper against the seam's contract; what
nothing else does is pass the same wrapped optimizer through both
training loops and check the constraint held over a whole run. That is
the transforms front's claim — a transform composes across algorithms —
exercised rather than stated.

The unconstrained control matters as much as the constrained run: a
radius the trajectory never reaches would let every assertion below
pass with the projection never once binding.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from dimma.algorithms.dp_sgd import train as dp_sgd_train
from dimma.algorithms.spiderboost import train as spiderboost_train
from dimma.core import updates
from dimma.models.logreg import init_params
from dimma.models.losses import per_sample_bce_loss
from dimma.transforms.projection import l1_projected

N_FEATURES = 5
N_ROWS = 1000
RADIUS = 0.5


@pytest.fixture
def problem():
    gen = np.random.default_rng(0)
    x = gen.standard_normal((N_ROWS, N_FEATURES)).astype(np.float32)
    w_true = np.array([1.5, -0.5, 2.0, 0.0, -1.0], dtype=np.float32)
    logits = x @ w_true
    y = gen.uniform(size=N_ROWS) < 1.0 / (1.0 + np.exp(-logits))
    return jnp.asarray(x), jnp.asarray(y, dtype=jnp.float32)


def run_dp_sgd(problem, optimizer):
    x, y = problem
    return dp_sgd_train.train(
        per_sample_bce_loss, init_params(jax.random.key(0), N_FEATURES),
        optimizer, x, y, jax.random.key(1), np.random.default_rng(2),
        steps=30, expected_batch_size=100, clip_norm=1.0,
        noise_multiplier=0.5,
    )


def run_spiderboost(problem, optimizer):
    x, y = problem
    return spiderboost_train.train(
        per_sample_bce_loss, init_params(jax.random.key(0), N_FEATURES),
        optimizer, x, y, jax.random.key(1), np.random.default_rng(2),
        steps=30, anchor_interval=5, anchor_expected_batch_size=100,
        variation_expected_batch_size=50, anchor_noise_scale=0.05,
        variation_noise_rate=0.1, variation_noise_cap=0.1,
    )


def global_l1(tree) -> float:
    flat, _ = ravel_pytree(tree)
    return float(jnp.sum(jnp.abs(flat)))


def test_the_unconstrained_runs_leave_the_ball(problem):
    """The control: the radius below is one both trajectories exceed
    when nothing projects, so the tests after this one bind."""
    assert global_l1(run_dp_sgd(problem, updates.sgd(0.5))) > RADIUS
    _, final = run_spiderboost(problem, updates.sgd(0.5))
    assert global_l1(final) > RADIUS


def test_dp_sgd_takes_the_wrapped_optimizer_unchanged(problem):
    trained = run_dp_sgd(problem, l1_projected(updates.sgd(0.5), RADIUS))
    assert global_l1(trained) <= RADIUS + 1e-4


def test_spiderboost_takes_the_same_wrapped_optimizer(problem):
    """Both returned iterates lie in the ball: every iterate was
    projected, so the output rule cannot draw one outside it."""
    output, final = run_spiderboost(
        problem, l1_projected(updates.sgd(0.5), RADIUS)
    )
    assert global_l1(output) <= RADIUS + 1e-4
    assert global_l1(final) <= RADIUS + 1e-4
