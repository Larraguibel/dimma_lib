"""The chain ADR-0012 built, end to end, on the model dimma ships.

Every link is covered elsewhere; what is pinned here is the claim
ADR-0009 declines to enforce — that the constants the accountant was
handed really do bound the gradients the loop went on to produce.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dimma.accounting import spiderboost as spiderboost_accounting
from dimma.accounting.lipschitz import logreg_bce_constants
from dimma.algorithms.dp_sgd import train as dp_sgd_train
from dimma.algorithms.spiderboost import train as spiderboost_train
from dimma.core import clipping, gradients, pytree, updates
from dimma.datasets.preprocessing import cap_feature_norms
from dimma.models.logreg import init_params
from dimma.models.losses import batch_bce_loss, per_sample_bce_loss

BOUND = 1.0
N_FEATURES = 5
N_ROWS = 2000


@pytest.fixture
def capped_problem():
    """A separable-ish click problem, capped the way a loader caps it.

    Heavy-tailed on purpose: a feature set whose norms are all under
    ``R`` already would make the cap a no-op and the bound below
    vacuous.
    """
    gen = np.random.default_rng(0)
    x = gen.standard_t(df=2, size=(N_ROWS, N_FEATURES)).astype(np.float32)
    w_true = np.array([1.5, -0.5, 2.0, 0.0, -1.0], dtype=np.float32)
    logits = x @ w_true
    y = (gen.uniform(size=N_ROWS) < 1.0 / (1.0 + np.exp(-logits)))

    x_capped, bound = cap_feature_norms(x, BOUND)
    return (jnp.asarray(x_capped), jnp.asarray(y, dtype=jnp.float32),
            jnp.asarray(x), bound)


@pytest.fixture
def constants():
    return logreg_bce_constants(BOUND, has_bias=True)


def parameter_settings():
    """Where the bounds have to hold: at init, and far from it.

    ``L0`` is approached only as the logit diverges, so a check at the
    initial parameters alone would pass against a constant several times
    too small.
    """
    key = jax.random.key(0)
    return [
        init_params(key, N_FEATURES),
        {"w": jnp.array([3.0, -2.0, 5.0, 1.0, -4.0]), "b": jnp.array(2.0)},
        {"w": jnp.full((N_FEATURES,), 40.0), "b": jnp.array(-30.0)},
    ]


# --------------------------------------------------------------------
# The premise ADR-0009 leaves to the caller's word
# --------------------------------------------------------------------


@pytest.mark.parametrize("params", parameter_settings())
def test_no_per_sample_gradient_exceeds_the_lipschitz_constant(
    capped_problem, constants, params
):
    """`L0` is what bounds the anchor branch's sensitivity. Supply one
    the data exceeds and the noise is calibrated against a sensitivity
    that is not the real one."""
    x, y, _, _ = capped_problem
    per_sample = gradients.per_sample_grads(per_sample_bce_loss)(params, x, y)
    norms = clipping.per_sample_norms(per_sample)
    assert float(jnp.max(norms)) <= constants.lipschitz_constant + 1e-5


def test_the_bound_would_not_hold_without_the_cap(capped_problem, constants):
    """The negative control. Without it the test above passes for a
    reason that has nothing to do with the cap."""
    _, y, x_uncapped, _ = capped_problem
    params = parameter_settings()[1]
    per_sample = gradients.per_sample_grads(per_sample_bce_loss)(
        params, x_uncapped, y)
    norms = clipping.per_sample_norms(per_sample)
    assert float(jnp.max(norms)) > constants.lipschitz_constant


def test_no_gradient_difference_exceeds_the_smoothness_bound(
    capped_problem, constants
):
    """`L1` is what bounds the variation branch below its cap: the
    per-sample gradient difference over the distance moved."""
    x, y, _, _ = capped_problem
    first, second = parameter_settings()[0], parameter_settings()[1]

    grads = gradients.per_sample_grads(per_sample_bce_loss)
    difference = jax.tree.map(
        lambda a, b: a - b, grads(first, x, y), grads(second, x, y))
    moved = float(pytree.global_norm(
        jax.tree.map(lambda a, b: a - b, first, second)))

    assert float(jnp.max(clipping.per_sample_norms(difference))) <= (
        constants.smoothness_constant * moved + 1e-5)


def test_the_lipschitz_constant_is_not_slack_enough_to_be_meaningless(
    capped_problem, constants
):
    """A constant ten times the truth would satisfy every test above.
    At a diverging logit the gradient norm approaches `L0`, so pin that
    the bound is nearly attained rather than merely respected."""
    x, y, _, _ = capped_problem
    far = {"w": jnp.full((N_FEATURES,), 40.0), "b": jnp.array(-30.0)}
    per_sample = gradients.per_sample_grads(per_sample_bce_loss)(far, x, y)
    attained = float(jnp.max(clipping.per_sample_norms(per_sample)))
    assert attained > 0.9 * constants.lipschitz_constant


# --------------------------------------------------------------------
# Both algorithms, the same model, the constants the bound implies
# --------------------------------------------------------------------


def test_dp_sgd_trains_the_shipped_model(capped_problem, constants):
    """Clipping at `L0` is the operation SpiderBoost assumes instead of
    performing, so this is the same bound spent two ways."""
    x, y, _, _ = capped_problem
    params = init_params(jax.random.key(0), N_FEATURES)
    before = float(batch_bce_loss(params, x, y))

    trained = dp_sgd_train.train(
        per_sample_bce_loss, params, updates.sgd(constants.step_size),
        x, y, jax.random.key(1), np.random.default_rng(2),
        steps=300, expected_batch_size=200,
        clip_norm=constants.lipschitz_constant, noise_multiplier=1.0,
    )
    assert float(batch_bce_loss(trained, x, y)) < before


def test_spiderboost_trains_the_shipped_model(capped_problem, constants):
    """The whole chain: R fixes the constants, the constants and a
    budget fix the noise scales, and the loop runs on them."""
    x, y, _, _ = capped_problem
    params = init_params(jax.random.key(0), N_FEATURES)
    before = float(batch_bce_loss(params, x, y))

    scales = spiderboost_accounting.noise_scales(
        lipschitz_constant=constants.lipschitz_constant,
        smoothness_constant=constants.smoothness_constant,
        target_epsilon=2.0, target_delta=1e-5, steps=300, anchor_interval=20,
        anchor_expected_batch_size=200, variation_expected_batch_size=100,
        dataset_size=N_ROWS,
    )

    # The second return, deliberately: the first is a uniform draw from
    # {w_1, ..., w_steps-1}, so it can land one step from the start and
    # say nothing about whether the trajectory descended. It is the
    # reportable result, not the probe for this question.
    _, final = spiderboost_train.train(
        per_sample_bce_loss, params, updates.sgd(constants.step_size),
        x, y, jax.random.key(1), np.random.default_rng(2),
        steps=300, anchor_interval=20, anchor_expected_batch_size=200,
        variation_expected_batch_size=100,
        anchor_noise_scale=scales.anchor_noise_scale,
        variation_noise_rate=scales.variation_noise_rate,
        variation_noise_cap=scales.variation_noise_cap,
    )
    assert float(batch_bce_loss(final, x, y)) < before


def test_the_two_algorithms_take_the_same_model_unchanged(capped_problem):
    """What the library is for. Neither loop imports the model, and
    both take it through the same `(params, x, y) -> scalar` seam, so a
    comparison between them is a comparison of the algorithms."""
    x, y, _, _ = capped_problem
    params = init_params(jax.random.key(0), N_FEATURES)

    dp_sgd_params = dp_sgd_train.train(
        per_sample_bce_loss, params, updates.sgd(0.5), x, y,
        jax.random.key(1), np.random.default_rng(2),
        steps=5, expected_batch_size=32, clip_norm=1.0, noise_multiplier=1.0,
    )
    spiderboost_params, _ = spiderboost_train.train(
        per_sample_bce_loss, params, updates.sgd(0.5), x, y,
        jax.random.key(1), np.random.default_rng(2),
        steps=5, anchor_interval=2, anchor_expected_batch_size=32,
        variation_expected_batch_size=32, anchor_noise_scale=0.1,
        variation_noise_rate=0.1, variation_noise_cap=0.2,
    )
    assert jax.tree_util.tree_structure(dp_sgd_params) == \
        jax.tree_util.tree_structure(spiderboost_params)
    assert jax.tree_util.tree_structure(dp_sgd_params) == \
        jax.tree_util.tree_structure(params)
