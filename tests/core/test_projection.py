"""l_1-ball projection geometry."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from dimma.core import projection


def l1(x) -> jax.Array:
    return jnp.sum(jnp.abs(x))


def test_matches_the_closed_form_on_a_known_vector():
    """Duchi et al. worked example: theta = 2.5 for radius 2."""
    x = jnp.array([3.0, -4.0, 1.0])
    assert jnp.allclose(
        projection.project_l1_ball(x, 2.0), jnp.array([0.5, -1.5, 0.0])
    )


@pytest.mark.parametrize("radius", [0.5, 1.0, 2.0, 7.0])
def test_result_is_inside_the_ball(radius):
    x = jnp.array([3.0, -4.0, 1.0, 0.2, -6.0])
    assert l1(projection.project_l1_ball(x, radius)) <= radius + 1e-5


def test_vectors_already_inside_are_returned_bit_exactly():
    """No distortion when the constraint is already satisfied."""
    x = jnp.array([0.1, -0.2, 0.05])
    assert jnp.array_equal(projection.project_l1_ball(x, 10.0), x)


def test_all_zero_input_is_handled():
    """The rho >= 1 guard: rho is 0 only at radius 0, and 0/0 only here.

    Any positive radius makes the first coordinate count, so rho >= 1
    already. The zero vector at radius zero is the one input that
    reaches the guard with a zero cumulative sum behind it.
    """
    x = jnp.zeros((5,))
    out = projection.project_l1_ball(x, 0.0)
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(out == 0.0)


def test_zero_radius_projects_to_the_origin():
    x = jnp.array([3.0, -4.0, 1.0])
    assert jnp.allclose(projection.project_l1_ball(x, 0.0), 0.0)


def test_projection_preserves_signs():
    x = jnp.array([3.0, -4.0, 1.0, -6.0])
    out = projection.project_l1_ball(x, 2.0)
    kept = out != 0
    assert jnp.all(jnp.sign(out[kept]) == jnp.sign(x[kept]))


def test_radius_may_be_traced():
    """SpiderBoost derives the radius at runtime; this is load-bearing."""
    x = jnp.array([3.0, -4.0, 1.0])
    jitted = jax.jit(projection.project_l1_ball)
    assert jnp.allclose(
        jitted(x, jnp.float32(2.0)), projection.project_l1_ball(x, 2.0)
    )


def test_negative_concrete_radius_is_rejected():
    """An empty ball would silently project to the origin."""
    x = jnp.array([1.0, 2.0])
    with pytest.raises(AssertionError, match="non-negative"):
        projection.project_l1_ball(x, -1.0)


def test_the_guard_does_not_reject_array_radii():
    """The check inspects the value, not the wrapper JAX put around it."""
    x = jnp.array([3.0, -4.0, 1.0])
    for radius in (np.float32(2.0), jnp.float32(2.0)):
        assert jnp.allclose(
            projection.project_l1_ball(x, radius),
            projection.project_l1_ball(x, 2.0),
        )


def test_pytree_projection_constrains_the_concatenation_not_each_leaf():
    """The ball is global across every leaf; per-leaf would be weaker."""
    tree = {"a": jnp.array([3.0, -4.0]), "b": jnp.array([[1.0], [-6.0]])}
    out = projection.project_l1_ball_pytree(tree, 2.0)
    flat, _ = ravel_pytree(out)
    assert l1(flat) <= 2.0 + 1e-5


def test_pytree_projection_preserves_structure_and_shapes():
    tree = {"a": jnp.array([3.0, -4.0]), "b": jnp.array([[1.0], [-6.0]])}
    out = projection.project_l1_ball_pytree(tree, 2.0)
    assert jax.tree_util.tree_structure(out) == \
        jax.tree_util.tree_structure(tree)
    assert out["a"].shape == (2,)
    assert out["b"].shape == (2, 1)


def test_pytree_projection_agrees_with_the_flat_one():
    tree = {"a": jnp.array([3.0, -4.0]), "b": jnp.array([[1.0], [-6.0]])}
    flat_in, unravel = ravel_pytree(tree)
    out = projection.project_l1_ball_pytree(tree, 2.0)
    flat_out, _ = ravel_pytree(out)
    assert jnp.allclose(flat_out, projection.project_l1_ball(flat_in, 2.0))


def test_pytree_projection_is_jittable():
    tree = {"a": jnp.array([3.0, -4.0]), "b": jnp.array([[1.0], [-6.0]])}
    jitted = jax.jit(projection.project_l1_ball_pytree)
    out_jit, _ = ravel_pytree(jitted(tree, jnp.float32(2.0)))
    out_eager, _ = ravel_pytree(projection.project_l1_ball_pytree(tree, 2.0))
    assert jnp.allclose(out_jit, out_eager)
