"""The stages composed, as an algorithm would compose them.

Each stage is covered in isolation elsewhere. This checks that they fit
together: that the layouts one stage produces are the ones the next
accepts, and that the whole chain survives `jax.jit`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dimma.core import aggregation, clipping, gradients, noise, updates
from dimma.core.sampling import poisson


def squared_error(params, x, y):
    residual = jnp.dot(params["w"], x) - y
    return 0.5 * residual ** 2


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(500, 3)), dtype=jnp.float32)
    y = jnp.asarray(rng.normal(size=(500,)), dtype=jnp.float32)
    return x, y


def private_step(grad_fn, params, x_batch, y_batch, mask, batch_size,
                 clip_norm, std, key):
    """Stages 3 through 6: gradients, clip, aggregate, perturb."""
    per_sample = grad_fn(params, x_batch, y_batch)
    per_sample = clipping.per_sample_clip(per_sample, clip_norm)
    averaged = aggregation.average_over_batch(
        per_sample, batch_size, mask=mask
    )
    return noise.add_gaussian(averaged, key, std)


def test_a_private_step_runs_end_to_end(data):
    x, y = data
    n, b_expected = x.shape[0], 50
    b_max = poisson.padded_batch_size(b_expected, n)
    rng = np.random.default_rng(0)

    indices, mask_np = poisson.subsample(rng, n, b_expected / n, b_max)
    grad = private_step(
        gradients.per_sample_grads(squared_error),
        {"w": jnp.zeros(3)},
        x[indices], y[indices], jnp.asarray(mask_np),
        b_expected, 1.0, 0.1, jax.random.key(0),
    )
    assert grad["w"].shape == (3,)
    assert jnp.all(jnp.isfinite(grad["w"]))


def test_the_private_step_is_jittable(data):
    """Stage 1 stays outside jit; everything after it must compile."""
    x, y = data
    n, b_expected = x.shape[0], 50
    b_max = poisson.padded_batch_size(b_expected, n)
    rng = np.random.default_rng(0)
    indices, mask_np = poisson.subsample(rng, n, b_expected / n, b_max)

    grad_fn = gradients.per_sample_grads(squared_error)
    step = jax.jit(
        lambda p, xb, yb, m, k: private_step(
            grad_fn, p, xb, yb, m, b_expected, 1.0, 0.1, k
        )
    )
    out = step({"w": jnp.zeros(3)}, x[indices], y[indices],
               jnp.asarray(mask_np), jax.random.key(0))
    assert jnp.all(jnp.isfinite(out["w"]))


def test_sensitivity_is_bounded_by_the_clip_before_noise(data):
    """What stage 4 buys stage 6: the pre-noise average is bounded.

    With every per-sample gradient clipped to `clip_norm` and the sum
    divided by `batch_size`, the aggregate cannot exceed
    `clip_norm * unmasked / batch_size`.
    """
    x, y = data
    n, b_expected, clip_norm = x.shape[0], 50, 0.7
    b_max = poisson.padded_batch_size(b_expected, n)
    rng = np.random.default_rng(0)
    indices, mask_np = poisson.subsample(rng, n, b_expected / n, b_max)
    mask = jnp.asarray(mask_np)

    per_sample = gradients.per_sample_grads(squared_error)(
        {"w": jnp.zeros(3)}, x[indices], y[indices]
    )
    averaged = aggregation.average_over_batch(
        clipping.per_sample_clip(per_sample, clip_norm), b_expected, mask=mask
    )
    bound = clip_norm * float(mask.sum()) / b_expected
    assert jnp.linalg.norm(averaged["w"]) <= bound + 1e-5


def test_training_loop_reduces_the_loss(data):
    """All seven stages, twenty steps, with noise small enough to learn."""
    x, y = data
    n, b_expected, T = x.shape[0], 100, 20
    b_max = poisson.padded_batch_size(b_expected, n)
    rng = np.random.default_rng(0)
    key = jax.random.key(0)

    grad_fn = gradients.per_sample_grads(squared_error)
    params = {"w": jnp.zeros(3)}
    opt = updates.sgd(0.5)
    state = updates.init(opt, params)

    def full_loss(p):
        return jnp.mean(jax.vmap(squared_error, in_axes=(None, 0, 0))(p, x, y))

    before = full_loss(params)
    for _ in range(T):
        indices, mask_np = poisson.subsample(rng, n, b_expected / n, b_max)
        key, subkey = jax.random.split(key)
        grad = private_step(
            grad_fn, params, x[indices], y[indices], jnp.asarray(mask_np),
            b_expected, 1.0, 0.01, subkey,
        )
        params, state = updates.apply(opt, params, grad, state)

    assert full_loss(params) < before


def test_the_non_private_path_drops_stages_four_and_six(data):
    """A baseline is the same pipeline, minus clipping and noise.

    Stage 3 becomes a batch gradient, which is exactly the mean of the
    per-sample gradients the private path would have clipped.
    """
    x, y = data
    params = {"w": jnp.zeros(3)}

    batched = gradients.batch_grads(squared_error)(params, x[:64], y[:64])
    per_sample = gradients.per_sample_grads(squared_error)(
        params, x[:64], y[:64]
    )
    averaged = aggregation.average_over_batch(per_sample, 64.0)
    assert jnp.allclose(batched["w"], averaged["w"], atol=1e-5)
