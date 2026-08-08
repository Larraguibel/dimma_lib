"""One DP-SGD iteration, against Algorithm 1 line by line."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dimma.algorithms.dp_sgd import step as dp_step
from dimma.core import aggregation, clipping, gradients, noise, updates
from dimma.core.sampling import poisson

from .conftest import squared_error

B_EXPECTED, CLIP = 100, 0.7


@pytest.fixture
def batch(problem, rng):
    """A drawn Poisson batch: padded inputs plus the mask."""
    x, y, _ = problem
    n = x.shape[0]
    b_max = poisson.padded_batch_size(B_EXPECTED, n)
    indices, mask = poisson.subsample(rng, n, B_EXPECTED / n, b_max)
    return x[indices], y[indices], jnp.asarray(mask)


@pytest.fixture
def grad_fn():
    return gradients.per_sample_grads(squared_error)


def _clipped(grad_fn, params, batch):
    x_b, y_b, _ = batch
    return clipping.per_sample_clip(grad_fn(params, x_b, y_b), CLIP)


def test_noiseless_estimate_equals_the_core_chain(grad_fn, zero_params, batch,
                                                  key):
    """With sigma = 0, sum-then-scale is the clipped batch average."""
    x_b, y_b, mask = batch
    got = dp_step.privatized_gradient(
        grad_fn, zero_params, x_b, y_b, mask, key,
        expected_batch_size=B_EXPECTED, clip_norm=CLIP, noise_multiplier=0.0,
    )
    want = aggregation.average_over_batch(
        _clipped(grad_fn, zero_params, batch), B_EXPECTED, mask=mask
    )
    assert jnp.array_equal(got["w"], want["w"])


def test_noise_is_added_to_the_sum_at_sigma_times_clip(grad_fn, zero_params,
                                                        batch, key):
    """``(sum + N(0, (sigma C)^2)) / L``, not ``mean + N(0, (sigma C / L)^2)``
    written out - the two agree, and this pins that they do."""
    x_b, y_b, mask = batch
    sigma = 4.0
    got = dp_step.privatized_gradient(
        grad_fn, zero_params, x_b, y_b, mask, key,
        expected_batch_size=B_EXPECTED, clip_norm=CLIP, noise_multiplier=sigma,
    )
    mean = aggregation.average_over_batch(
        _clipped(grad_fn, zero_params, batch), B_EXPECTED, mask=mask
    )
    want = noise.add_gaussian(mean, key, sigma * CLIP / B_EXPECTED)
    assert jnp.allclose(got["w"], want["w"], atol=1e-7)


def test_released_noise_has_standard_deviation_sigma_c_over_l(grad_fn,
                                                               zero_params,
                                                               batch):
    """The scale the accountant assumes, measured on the output."""
    x_b, y_b, mask = batch
    sigma = 4.0
    draws = jnp.stack([
        dp_step.privatized_gradient(
            grad_fn, zero_params, x_b, y_b, mask, k,
            expected_batch_size=B_EXPECTED, clip_norm=CLIP,
            noise_multiplier=sigma,
        )["w"]
        for k in jax.random.split(jax.random.key(0), 3000)
    ])
    expected = sigma * CLIP / B_EXPECTED
    assert np.allclose(draws.std(axis=0), expected, rtol=0.08)


def test_sensitivity_is_bounded_by_clip_over_the_expected_batch_size(
        grad_fn, zero_params, batch, key):
    """What clipping buys: the noiseless estimate is bounded a priori."""
    x_b, y_b, mask = batch
    got = dp_step.privatized_gradient(
        grad_fn, zero_params, x_b, y_b, mask, key,
        expected_batch_size=B_EXPECTED, clip_norm=CLIP, noise_multiplier=0.0,
    )
    bound = CLIP * float(mask.sum()) / B_EXPECTED
    assert jnp.linalg.norm(got["w"]) <= bound + 1e-5


def test_padding_does_not_contribute(grad_fn, zero_params, problem, key):
    """Masked slots hold index 0, a real row; only the mask stops it."""
    x, y, _ = problem
    real = 8
    indices = np.concatenate([np.arange(real), np.zeros(20, dtype=np.int64)])
    mask = jnp.asarray(
        np.concatenate([np.ones(real), np.zeros(20)]), jnp.float32
    )
    padded = dp_step.privatized_gradient(
        grad_fn, zero_params, x[indices], y[indices], mask, key,
        expected_batch_size=B_EXPECTED, clip_norm=CLIP, noise_multiplier=0.0,
    )
    exact = dp_step.privatized_gradient(
        grad_fn, zero_params, x[:real], y[:real], jnp.ones(real), key,
        expected_batch_size=B_EXPECTED, clip_norm=CLIP, noise_multiplier=0.0,
    )
    assert jnp.allclose(padded["w"], exact["w"], atol=1e-6)


def test_the_divisor_is_not_the_batch_length(grad_fn, zero_params, batch, key):
    """L is a constant; a data-dependent divisor would leak."""
    x_b, y_b, mask = batch
    args = dict(expected_batch_size=B_EXPECTED, clip_norm=CLIP,
                noise_multiplier=0.0)
    got = dp_step.privatized_gradient(grad_fn, zero_params, x_b, y_b, mask,
                                      key, **args)
    doubled = dp_step.privatized_gradient(
        grad_fn, zero_params, x_b, y_b, mask, key,
        **{**args, "expected_batch_size": 2 * B_EXPECTED}
    )
    assert jnp.allclose(got["w"], 2 * doubled["w"], atol=1e-7)


def test_the_key_controls_the_noise(grad_fn, zero_params, batch):
    x_b, y_b, mask = batch
    args = dict(expected_batch_size=B_EXPECTED, clip_norm=CLIP,
                noise_multiplier=1.0)
    a = dp_step.privatized_gradient(grad_fn, zero_params, x_b, y_b, mask,
                                    jax.random.key(0), **args)
    same = dp_step.privatized_gradient(grad_fn, zero_params, x_b, y_b, mask,
                                       jax.random.key(0), **args)
    other = dp_step.privatized_gradient(grad_fn, zero_params, x_b, y_b, mask,
                                        jax.random.key(1), **args)
    assert jnp.array_equal(a["w"], same["w"])
    assert not jnp.allclose(a["w"], other["w"])


def test_step_descends_along_the_privatized_gradient(grad_fn, zero_params,
                                                     batch, key):
    """Algorithm 1's descent line: theta - eta * g~, via updates.sgd."""
    x_b, y_b, mask = batch
    lr = 0.3
    opt = updates.sgd(lr)
    args = dict(expected_batch_size=B_EXPECTED, clip_norm=CLIP,
                noise_multiplier=1.0)

    grad = dp_step.privatized_gradient(grad_fn, zero_params, x_b, y_b, mask,
                                       key, **args)
    new_params, _ = dp_step.step(grad_fn, opt, zero_params,
                                 updates.init(opt, zero_params),
                                 x_b, y_b, mask, key, **args)
    assert jnp.allclose(new_params["w"], zero_params["w"] - lr * grad["w"],
                        atol=1e-7)


def test_the_step_is_jittable_and_traces_once(grad_fn, zero_params, batch,
                                              key):
    """grad_fn and optimizer are statics; binding them costs no retrace."""
    traces = 0

    def counted_loss(params, x, y):
        nonlocal traces
        traces += 1
        return squared_error(params, x, y)

    x_b, y_b, mask = batch
    opt = updates.sgd(0.1)
    compiled = jax.jit(partial(
        dp_step.step, gradients.per_sample_grads(counted_loss), opt,
        expected_batch_size=B_EXPECTED, clip_norm=CLIP, noise_multiplier=1.0,
    ))
    params, state = zero_params, updates.init(opt, zero_params)
    for _ in range(5):
        key, subkey = jax.random.split(key)
        params, state = compiled(params, state, x_b, y_b, mask, subkey)

    assert traces == 1
    assert jnp.all(jnp.isfinite(params["w"]))
