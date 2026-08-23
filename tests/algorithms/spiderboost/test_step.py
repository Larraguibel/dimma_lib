"""The two mechanisms, against Algorithm 2's lines 9 and 13.

The only seam at which a mechanism-level property is observable: by the
time `train` returns, both releases are an update.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dimma.algorithms.spiderboost import step as spider_step
from dimma.core import aggregation, gradients, pytree, updates
from dimma.core.sampling import poisson

from ...helpers import tree_equal
from .conftest import squared_error

B1, B2 = 100, 60


@pytest.fixture
def batch(problem, rng):
    """A drawn Poisson batch: padded inputs plus the mask."""
    x, y, _ = problem
    n = x.shape[0]
    b_max = poisson.padded_batch_size(B1, n)
    indices, mask = poisson.subsample(rng, n, B1 / n, b_max)
    return x[indices], y[indices], jnp.asarray(mask)


@pytest.fixture
def grad_fn():
    return gradients.per_sample_grads(squared_error)


def _distance(a, b):
    return float(pytree.global_norm(pytree.sub(a, b)))


def test_the_anchor_release_at_zero_noise_is_the_core_chain(grad_fn,
                                                            zero_params,
                                                            batch, key):
    """Line 9 without ``g_t``: the per-sample gradient mean, bit for bit."""
    x_b, y_b, mask = batch
    got = spider_step.anchor_release(
        grad_fn, zero_params, x_b, y_b, mask, key,
        expected_batch_size=B1, noise_scale=0.0,
    )
    want = aggregation.average_over_batch(
        grad_fn(zero_params, x_b, y_b), B1, mask=mask
    )
    assert tree_equal(got, want)


def test_the_variation_release_at_zero_noise_is_the_core_chain(grad_fn,
                                                               zero_params,
                                                               moved_params,
                                                               batch, key):
    """Line 13 without ``g_t``: the mean per-sample gradient *difference*."""
    x_b, y_b, mask = batch
    got = spider_step.variation_release(
        grad_fn, moved_params, zero_params, x_b, y_b, mask, key,
        expected_batch_size=B2, noise_rate=0.0, noise_cap=0.0,
    )
    want = aggregation.average_over_batch(
        pytree.sub(grad_fn(moved_params, x_b, y_b),
                   grad_fn(zero_params, x_b, y_b)),
        B2, mask=mask,
    )
    assert tree_equal(got, want)


def test_the_variation_release_aggregates_gradient_differences(grad_fn,
                                                                zero_params,
                                                                moved_params,
                                                                batch, key):
    """Differences over one batch, not two independently drawn gradients.

    Checked against gradients computed separately at the two parameter
    sets: a release that evaluated `previous_params` on some other batch
    would still look like a difference, and would not match this.
    """
    x_b, y_b, mask = batch
    got = spider_step.variation_release(
        grad_fn, moved_params, zero_params, x_b, y_b, mask, key,
        expected_batch_size=B2, noise_rate=0.0, noise_cap=0.0,
    )
    at_current = aggregation.average_over_batch(
        grad_fn(moved_params, x_b, y_b), B2, mask=mask)
    at_previous = aggregation.average_over_batch(
        grad_fn(zero_params, x_b, y_b), B2, mask=mask)
    assert jnp.allclose(got["w"], (at_current["w"] - at_previous["w"]),
                        atol=1e-6)


def test_the_anchor_noise_has_the_anchor_noise_scale(grad_fn, zero_params,
                                                     batch):
    """``sigma_1`` is the scale on the released mean. Measured, not assumed."""
    x_b, y_b, mask = batch
    scale = 0.4
    draws = jnp.stack([
        spider_step.anchor_release(
            grad_fn, zero_params, x_b, y_b, mask, k,
            expected_batch_size=B1, noise_scale=scale,
        )["w"]
        for k in jax.random.split(jax.random.key(0), 3000)
    ])
    assert np.allclose(draws.std(axis=0), scale, rtol=0.08)


def test_the_variation_noise_is_the_rate_times_the_distance_moved(
        grad_fn, zero_params, moved_params, batch):
    """Below the cap the scale tracks ``||w_t - w_{t-1}||``, per line 12."""
    x_b, y_b, mask = batch
    rate, cap = 0.9, 100.0
    draws = jnp.stack([
        spider_step.variation_release(
            grad_fn, moved_params, zero_params, x_b, y_b, mask, k,
            expected_batch_size=B2, noise_rate=rate, noise_cap=cap,
        )["w"]
        for k in jax.random.split(jax.random.key(0), 3000)
    ])
    expected = rate * _distance(moved_params, zero_params)
    assert expected < cap
    assert np.allclose(draws.std(axis=0), expected, rtol=0.08)


def test_the_variation_noise_is_capped_when_the_parameters_move_far(
        grad_fn, zero_params, moved_params, batch):
    """The other branch of line 12's minimum: ``sigma-hat_2`` takes over."""
    x_b, y_b, mask = batch
    rate, cap = 50.0, 0.5
    draws = jnp.stack([
        spider_step.variation_release(
            grad_fn, moved_params, zero_params, x_b, y_b, mask, k,
            expected_batch_size=B2, noise_rate=rate, noise_cap=cap,
        )["w"]
        for k in jax.random.split(jax.random.key(0), 3000)
    ])
    assert rate * _distance(moved_params, zero_params) > cap
    assert np.allclose(draws.std(axis=0), cap, rtol=0.08)


def test_the_anchor_divisor_is_not_the_batch_length(grad_fn, zero_params,
                                                    batch, key):
    """``b_1`` is a constant; a data-dependent divisor would leak."""
    x_b, y_b, mask = batch
    got = spider_step.anchor_release(
        grad_fn, zero_params, x_b, y_b, mask, key,
        expected_batch_size=B1, noise_scale=0.0)
    doubled = spider_step.anchor_release(
        grad_fn, zero_params, x_b, y_b, mask, key,
        expected_batch_size=2 * B1, noise_scale=0.0)
    assert jnp.allclose(got["w"], 2 * doubled["w"], atol=1e-7)


def test_the_variation_divisor_is_not_the_batch_length(grad_fn, zero_params,
                                                       moved_params, batch,
                                                       key):
    """``b_2`` likewise, and separately: the two are not one parameter."""
    x_b, y_b, mask = batch
    args = dict(noise_rate=0.0, noise_cap=0.0)
    got = spider_step.variation_release(
        grad_fn, moved_params, zero_params, x_b, y_b, mask, key,
        expected_batch_size=B2, **args)
    doubled = spider_step.variation_release(
        grad_fn, moved_params, zero_params, x_b, y_b, mask, key,
        expected_batch_size=2 * B2, **args)
    assert jnp.allclose(got["w"], 2 * doubled["w"], atol=1e-7)


def test_padding_does_not_contribute(grad_fn, zero_params, moved_params,
                                     problem, key):
    """Masked slots hold index 0, a real row; only the mask stops it."""
    x, y, _ = problem
    real = 8
    indices = np.concatenate([np.arange(real), np.zeros(20, dtype=np.int64)])
    mask = jnp.asarray(
        np.concatenate([np.ones(real), np.zeros(20)]), jnp.float32
    )
    padded_anchor = spider_step.anchor_release(
        grad_fn, zero_params, x[indices], y[indices], mask, key,
        expected_batch_size=B1, noise_scale=0.0)
    exact_anchor = spider_step.anchor_release(
        grad_fn, zero_params, x[:real], y[:real], jnp.ones(real), key,
        expected_batch_size=B1, noise_scale=0.0)
    assert jnp.allclose(padded_anchor["w"], exact_anchor["w"], atol=1e-6)

    args = dict(expected_batch_size=B2, noise_rate=0.0, noise_cap=0.0)
    padded_variation = spider_step.variation_release(
        grad_fn, moved_params, zero_params, x[indices], y[indices], mask, key,
        **args)
    exact_variation = spider_step.variation_release(
        grad_fn, moved_params, zero_params, x[:real], y[:real],
        jnp.ones(real), key, **args)
    assert jnp.allclose(padded_variation["w"], exact_variation["w"],
                        atol=1e-6)


def test_the_key_controls_the_noise(grad_fn, zero_params, moved_params,
                                    batch):
    """Same key, same release; different key, different release."""
    x_b, y_b, mask = batch
    anchor = partial(spider_step.anchor_release, grad_fn, zero_params, x_b,
                     y_b, mask)
    variation = partial(spider_step.variation_release, grad_fn, moved_params,
                        zero_params, x_b, y_b, mask)

    a = anchor(jax.random.key(0), expected_batch_size=B1, noise_scale=1.0)
    same = anchor(jax.random.key(0), expected_batch_size=B1, noise_scale=1.0)
    other = anchor(jax.random.key(1), expected_batch_size=B1, noise_scale=1.0)
    assert tree_equal(a, same)
    assert not jnp.allclose(a["w"], other["w"])

    args = dict(expected_batch_size=B2, noise_rate=1.0, noise_cap=10.0)
    b = variation(jax.random.key(0), **args)
    b_same = variation(jax.random.key(0), **args)
    b_other = variation(jax.random.key(1), **args)
    assert tree_equal(b, b_same)
    assert not jnp.allclose(b["w"], b_other["w"])


def test_the_accumulation_is_exactly_addition(grad_fn, zero_params,
                                              moved_params, batch, key):
    """Line 14 is ``nabla_{t-1} + Delta_t`` and nothing else.

    Nothing privacy-relevant may hide on the post-processing side of the
    seam, so the estimate the step carries out has to be the previous
    one plus the release, bit for bit.
    """
    x_b, y_b, mask = batch
    opt = updates.sgd(0.1)
    previous_estimate = {"w": jnp.array([0.7, -1.3, 0.2])}
    args = dict(expected_batch_size=B2, noise_rate=1.0, noise_cap=10.0)

    increment = spider_step.variation_release(
        grad_fn, moved_params, zero_params, x_b, y_b, mask, key, **args)
    _, estimate, _ = spider_step.variation_step(
        grad_fn, opt, moved_params, zero_params, previous_estimate,
        updates.init(opt, moved_params), x_b, y_b, mask, key, **args)
    assert tree_equal(estimate, pytree.add(previous_estimate, increment))


def test_the_anchor_estimate_replaces_rather_than_accumulates(grad_fn,
                                                              zero_params,
                                                              batch, key):
    """Line 9 assigns ``nabla_t``: a phase starts over, it does not add.

    The anchor step takes no previous estimate at all, which is the
    shape of that decision; this pins that what it carries out is its
    own release.
    """
    x_b, y_b, mask = batch
    opt = updates.sgd(0.1)
    args = dict(expected_batch_size=B1, noise_scale=0.5)
    release = spider_step.anchor_release(
        grad_fn, zero_params, x_b, y_b, mask, key, **args)
    _, estimate, _ = spider_step.anchor_step(
        grad_fn, opt, zero_params, updates.init(opt, zero_params),
        x_b, y_b, mask, key, **args)
    assert tree_equal(estimate, release)


def test_each_branch_descends_along_the_running_estimate(grad_fn, zero_params,
                                                         moved_params, batch,
                                                         key):
    """Line 16, ``w_{t+1} = w_t - eta nabla_t``, in both branches."""
    x_b, y_b, mask = batch
    lr = 0.3
    opt = updates.sgd(lr)

    anchor_params, anchor_estimate, _ = spider_step.anchor_step(
        grad_fn, opt, zero_params, updates.init(opt, zero_params),
        x_b, y_b, mask, key, expected_batch_size=B1, noise_scale=0.5)
    assert jnp.allclose(anchor_params["w"],
                        zero_params["w"] - lr * anchor_estimate["w"],
                        atol=1e-7)

    previous_estimate = {"w": jnp.array([0.7, -1.3, 0.2])}
    variation_params, variation_estimate, _ = spider_step.variation_step(
        grad_fn, opt, moved_params, zero_params, previous_estimate,
        updates.init(opt, moved_params), x_b, y_b, mask, key,
        expected_batch_size=B2, noise_rate=1.0, noise_cap=10.0)
    assert jnp.allclose(variation_params["w"],
                        moved_params["w"] - lr * variation_estimate["w"],
                        atol=1e-7)


def test_each_branch_is_jittable_and_traces_once(zero_params, moved_params,
                                                 batch, key):
    """grad_fn and optimizer are statics; binding them costs no retrace.

    The variation branch traces the loss twice — once per parameter set,
    since line 13 evaluates at both — and compiles once, so the count
    does not grow with the number of calls.
    """
    anchor_traces = variation_traces = 0

    def counted_anchor_loss(params, x, y):
        nonlocal anchor_traces
        anchor_traces += 1
        return squared_error(params, x, y)

    def counted_variation_loss(params, x, y):
        nonlocal variation_traces
        variation_traces += 1
        return squared_error(params, x, y)

    x_b, y_b, mask = batch
    opt = updates.sgd(0.1)
    anchor = jax.jit(partial(
        spider_step.anchor_step,
        gradients.per_sample_grads(counted_anchor_loss), opt,
        expected_batch_size=B1, noise_scale=0.5,
    ))
    variation = jax.jit(partial(
        spider_step.variation_step,
        gradients.per_sample_grads(counted_variation_loss), opt,
        expected_batch_size=B2, noise_rate=1.0, noise_cap=10.0,
    ))

    params, previous = moved_params, zero_params
    state = updates.init(opt, params)
    estimate = {"w": jnp.zeros(3)}
    for _ in range(5):
        key, anchor_key, variation_key = jax.random.split(key, 3)
        _, estimate, _ = anchor(params, state, x_b, y_b, mask, anchor_key,)
        _, estimate, state = variation(params, previous, estimate, state,
                                       x_b, y_b, mask, variation_key)

    assert anchor_traces == 1
    assert variation_traces == 2
    assert jnp.all(jnp.isfinite(estimate["w"]))
