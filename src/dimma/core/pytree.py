"""Vector-space operations on JAX pytrees.

Implements no pipeline stage. Membership is closed to these four
operations; see :mod:`dimma.core` for the rule.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def add(a, b):
    """Element-wise sum of two pytrees with identical structure."""
    return jax.tree.map(lambda x, y: x + y, a, b)


def sub(a, b):
    """Element-wise difference of two pytrees with identical structure."""
    return jax.tree.map(lambda x, y: x - y, a, b)


def scale(pytree, factor: float | jax.Array):
    """Multiply every leaf by ``factor``, which may be traced."""
    return jax.tree.map(lambda leaf: leaf * factor, pytree)


def global_norm(pytree) -> jax.Array:
    """Global ``l_2`` norm across all leaves and dimensions, as a scalar."""
    leaves = jax.tree_util.tree_leaves(pytree)
    return jnp.sqrt(sum(jnp.sum(leaf ** 2) for leaf in leaves))
