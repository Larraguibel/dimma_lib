"""Stage 3 - gradients."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from dimma.core import gradients

from tests.helpers import tree_allclose


def squared_error(params, x, y):
    """A dimma per-sample loss: one example in, scalar out.

    ``0.5 * (w @ x - y)^2``, whose gradient ``(w @ x - y) * x`` is known
    in closed form.
    """
    residual = jnp.dot(params["w"], x) - y
    return 0.5 * residual ** 2


@pytest.fixture
def batch():
    x = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    y = jnp.array([1.0, 2.0, 0.0, -1.0])
    return x, y


@pytest.fixture
def w():
    return {"w": jnp.array([0.5, -0.5])}


def test_per_sample_grads_match_the_closed_form(w, batch):
    x, y = batch
    grads = gradients.per_sample_grads(squared_error)(w, x, y)
    residual = x @ w["w"] - y
    expected = residual[:, None] * x
    assert jnp.allclose(grads["w"], expected)


def test_per_sample_grads_keep_one_row_per_example(w, batch):
    """Leaves must be (B, *param_shape) — the layout stages 4 and 5 expect."""
    x, y = batch
    grads = gradients.per_sample_grads(squared_error)(w, x, y)
    assert grads["w"].shape == (x.shape[0],) + w["w"].shape


def test_batch_grads_have_no_leading_batch_axis(w, batch):
    x, y = batch
    grad = gradients.batch_grads(squared_error)(w, x, y)
    assert grad["w"].shape == w["w"].shape


def test_batch_grads_equal_the_mean_of_per_sample_grads(w, batch):
    """The relationship that makes a baseline comparable to its private twin."""
    x, y = batch
    per_sample = gradients.per_sample_grads(squared_error)(w, x, y)
    batched = gradients.batch_grads(squared_error)(w, x, y)
    expected = jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), per_sample)
    assert tree_allclose(batched, expected, atol=1e-6)


def test_both_factories_take_the_same_loss(w, batch):
    """One loss drives the private and the non-private path alike."""
    x, y = batch
    gradients.per_sample_grads(squared_error)(w, x, y)
    gradients.batch_grads(squared_error)(w, x, y)


def test_gradient_functions_are_jittable(w, batch):
    x, y = batch
    grad_fn = gradients.per_sample_grads(squared_error)
    assert tree_allclose(jax.jit(grad_fn)(w, x, y), grad_fn(w, x, y))


def test_per_sample_grads_handle_nested_params(batch):
    x, y = batch

    def nested_loss(params, x_single, y_single):
        residual = jnp.dot(params["layer"]["w"], x_single) + \
            params["bias"] - y_single
        return 0.5 * residual ** 2

    nested = {"layer": {"w": jnp.array([0.5, -0.5])}, "bias": jnp.array(0.1)}
    grads = gradients.per_sample_grads(nested_loss)(nested, x, y)
    assert grads["layer"]["w"].shape == (4, 2)
    assert grads["bias"].shape == (4,)
