"""The shipped logistic regression: one linear layer, no hashing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from dimma.models import logreg


D = 5


@pytest.fixture
def trained() -> dict:
    """Parameters with a known, non-degenerate value."""
    return {"w": jnp.array([1.0, -2.0, 0.5, 0.0, 3.0]), "b": jnp.array(-0.25)}


def test_init_params_shapes_and_bias(key):
    params = logreg.init_params(key, D)
    assert set(params) == {"w", "b"}
    assert params["w"].shape == (D,)
    assert params["b"].shape == ()
    assert params["b"] == 0.0


def test_init_keeps_the_initial_logits_near_zero(key):
    """The point of the small init scale, on unit-scale features."""
    params = logreg.init_params(key, D)
    x = jax.random.normal(jax.random.key(1), (64, D))
    logits = jax.vmap(logreg.forward, in_axes=(None, 0))(params, x)
    assert jnp.max(jnp.abs(logits)) < 0.1


def test_init_is_deterministic_in_the_key():
    a = logreg.init_params(jax.random.key(7), D)
    b = logreg.init_params(jax.random.key(7), D)
    assert jnp.array_equal(a["w"], b["w"])


def test_forward_is_the_scalar_logit(trained):
    x = jnp.array([1.0, 1.0, 2.0, 9.0, -1.0])
    logit = logreg.forward(trained, x)
    assert logit.shape == ()
    assert jnp.allclose(logit, 1.0 - 2.0 + 1.0 + 0.0 - 3.0 - 0.25)


def test_forward_is_affine_in_the_features(trained):
    """No hidden layer: the map is ``w . x + b`` and nothing else."""
    x1 = jnp.array([1.0, 0.0, -1.0, 2.0, 0.5])
    x2 = jnp.array([0.0, 3.0, 1.0, -1.0, 1.5])
    mid = logreg.forward(trained, 0.5 * (x1 + x2))
    ends = 0.5 * (logreg.forward(trained, x1) + logreg.forward(trained, x2))
    assert jnp.allclose(mid, ends, atol=1e-6)


def test_forward_gradient_is_the_feature_vector(trained):
    """``d logit / dw = x`` and ``d logit / db = 1``."""
    x = jnp.array([1.0, -0.5, 2.0, 0.0, 4.0])
    grad = jax.grad(logreg.forward)(trained, x)
    assert jnp.allclose(grad["w"], x)
    assert jnp.allclose(grad["b"], 1.0)


def test_vmap_over_a_batch_matches_the_loop(trained):
    x = jax.random.normal(jax.random.key(2), (6, D))
    batched = jax.vmap(logreg.forward, in_axes=(None, 0))(trained, x)
    looped = jnp.stack([logreg.forward(trained, x[i]) for i in range(6)])
    assert batched.shape == (6,)
    assert jnp.allclose(batched, looped, atol=1e-6)


def test_forward_is_jittable(trained):
    x = jnp.ones((D,))
    assert jnp.allclose(jax.jit(logreg.forward)(trained, x),
                        logreg.forward(trained, x))
