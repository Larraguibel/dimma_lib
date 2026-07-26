"""Vector-space operations on pytrees."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from dimma.core import pytree

from tests.helpers import tree_allclose


def test_add_is_elementwise(params):
    out = pytree.add(params, params)
    assert tree_allclose(out, jax.tree.map(lambda x: 2 * x, params))


def test_sub_of_identical_trees_is_zero(params):
    out = pytree.sub(params, params)
    assert all(
        jnp.all(leaf == 0) for leaf in jax.tree_util.tree_leaves(out)
    )


def test_add_and_sub_are_inverse(params):
    other = jax.tree.map(lambda x: x * 0.37 + 1.0, params)
    assert tree_allclose(pytree.sub(pytree.add(params, other), other), params)


def test_scale_multiplies_every_leaf(params):
    out = pytree.scale(params, 2.5)
    assert tree_allclose(out, jax.tree.map(lambda x: 2.5 * x, params))


def test_operations_preserve_structure(params):
    for out in (
        pytree.add(params, params),
        pytree.sub(params, params),
        pytree.scale(params, 2.0),
    ):
        assert jax.tree_util.tree_structure(out) == \
            jax.tree_util.tree_structure(params)


def test_global_norm_spans_all_leaves_and_dims(params):
    """The norm is over the concatenation of every leaf, not per-leaf."""
    flat = jnp.concatenate(
        [leaf.ravel() for leaf in jax.tree_util.tree_leaves(params)]
    )
    assert jnp.allclose(pytree.global_norm(params), jnp.linalg.norm(flat))


def test_global_norm_of_zeros_is_zero(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    assert pytree.global_norm(zeros) == 0.0


def test_global_norm_is_scalar(params):
    assert pytree.global_norm(params).shape == ()


@pytest.mark.parametrize("factor", [0.0, 0.5, 3.0])
def test_scale_accepts_a_traced_factor(params, factor):
    """`scale` is used with runtime-derived factors inside jitted kernels."""
    jitted = jax.jit(pytree.scale)
    assert tree_allclose(
        jitted(params, jnp.float32(factor)), pytree.scale(params, factor)
    )
