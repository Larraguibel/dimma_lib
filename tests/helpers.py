"""Assertion helpers shared across the suite."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def tree_allclose(a, b, **kwargs) -> bool:
    """True if two pytrees share structure and all leaves are close."""
    leaves_a, def_a = jax.tree_util.tree_flatten(a)
    leaves_b, def_b = jax.tree_util.tree_flatten(b)
    if def_a != def_b:
        return False
    return all(jnp.allclose(x, y, **kwargs) for x, y in zip(leaves_a, leaves_b))


def tree_equal(a, b) -> bool:
    """True if two pytrees share structure and all leaves are bit-identical."""
    leaves_a, def_a = jax.tree_util.tree_flatten(a)
    leaves_b, def_b = jax.tree_util.tree_flatten(b)
    if def_a != def_b:
        return False
    return all(jnp.array_equal(x, y) for x, y in zip(leaves_a, leaves_b))
