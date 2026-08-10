"""One SGD iteration, against the stages it is and the ones it is not."""

from __future__ import annotations

import inspect
from functools import partial

import jax
import jax.numpy as jnp
import pytest

from dimma.algorithms.dp_sgd import step as dp_step
from dimma.algorithms.sgd import step as sgd_step
from dimma.core import gradients, updates

from ...helpers import tree_allclose
from .conftest import squared_error

BATCH_SIZE = 100


@pytest.fixture
def batch(problem):
    """A fixed-size batch: inputs and targets, and no mask."""
    x, y, _ = problem
    return x[:BATCH_SIZE], y[:BATCH_SIZE]


@pytest.fixture
def grad_fn():
    return gradients.batch_grads(squared_error)


def test_the_step_is_the_gradient_scaled_by_the_rate(grad_fn, zero_params,
                                                     batch):
    """theta - eta * g, rebuilt from the primitives."""
    x_b, y_b = batch
    eta = 0.1
    optimizer = updates.sgd(eta)
    params, _ = sgd_step.step(
        grad_fn, optimizer, zero_params, updates.init(optimizer, zero_params),
        x_b, y_b,
    )
    expected = jax.tree.map(
        lambda p, g: p - eta * g, zero_params, grad_fn(zero_params, x_b, y_b)
    )
    assert tree_allclose(params, expected)


def test_the_gradient_is_the_mean_not_the_sum(grad_fn, zero_params, batch):
    """`batch_grads` differentiates the mean, so no divisor appears in
    the step and duplicating the batch does not double the move."""
    x_b, y_b = batch
    doubled = jnp.concatenate([x_b, x_b]), jnp.concatenate([y_b, y_b])
    assert tree_allclose(
        grad_fn(zero_params, x_b, y_b),
        grad_fn(zero_params, *doubled),
        atol=1e-5,
    )


def test_the_step_advances_the_optimizer_count(grad_fn, zero_params, batch):
    x_b, y_b = batch
    optimizer = updates.sgd(0.1)
    _, opt_state = sgd_step.step(
        grad_fn, optimizer, zero_params, updates.init(optimizer, zero_params),
        x_b, y_b,
    )
    assert int(opt_state.count) == 1


def test_the_step_compiles(grad_fn, zero_params, batch):
    """grad_fn and optimizer are static; everything else traces."""
    x_b, y_b = batch
    optimizer = updates.sgd(0.1)
    compiled = jax.jit(partial(sgd_step.step, grad_fn, optimizer))
    params, _ = compiled(
        zero_params, updates.init(optimizer, zero_params), x_b, y_b
    )
    assert jnp.all(jnp.isfinite(params["w"]))


def test_a_repeated_step_is_deterministic(grad_fn, zero_params, batch):
    """Nothing here is random. The private counterpart's step is not."""
    x_b, y_b = batch
    optimizer = updates.sgd(0.1)
    state = updates.init(optimizer, zero_params)
    first, _ = sgd_step.step(grad_fn, optimizer, zero_params, state, x_b, y_b)
    second, _ = sgd_step.step(grad_fn, optimizer, zero_params, state, x_b, y_b)
    assert tree_allclose(first, second)


def test_the_step_takes_no_key():
    """One random stream, not two. An accepted-and-unread key would be a
    way two runs differ without the difference being reported."""
    assert "key" not in inspect.signature(sgd_step.step).parameters


def test_the_step_takes_no_mask():
    """Shuffled sampling has fixed cardinality, so nothing is padded."""
    assert "mask" not in inspect.signature(sgd_step.step).parameters


def test_the_step_takes_no_privacy_parameters():
    """Stages 4 and 6 are dropped, not defaulted to something harmless."""
    names = set(inspect.signature(sgd_step.step).parameters)
    assert not names & {
        "clip_norm", "noise_multiplier", "expected_batch_size",
    }


def test_there_is_no_release_function():
    """One function, per ADR-0006: nothing is released here."""
    assert not hasattr(sgd_step, "privatized_gradient")
    assert not hasattr(sgd_step, "gradient")


def test_the_private_release_reduces_to_this_gradient(zero_params, batch, key):
    """The two arms differ in the privacy and in nothing else (ADR-0005).

    Switch the privacy off in DP-SGD's release — a clip bound no
    gradient reaches, and no noise — and what is left is the batch mean
    this step descends along. Anything else between the two loops would
    show up here as a mismatch, so this is where the comparison's
    controlled-ness is pinned rather than inferred from a loss ordering.
    """
    x_b, y_b = batch
    per_sample = gradients.per_sample_grads(squared_error)
    unprivate = dp_step.privatized_gradient(
        per_sample, zero_params, x_b, y_b,
        jnp.ones(BATCH_SIZE), key,
        expected_batch_size=BATCH_SIZE, clip_norm=1e6, noise_multiplier=0.0,
    )
    assert tree_allclose(
        unprivate, gradients.batch_grads(squared_error)(zero_params, x_b, y_b),
        atol=1e-5,
    )
