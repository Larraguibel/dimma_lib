"""Euclidean projection onto the ``l_1`` ball, over arrays and pytrees.

Pure geometry, and makes no privacy claim: that projecting a private 
quantity is free post-processing is a statement about the mechanism, 
not about these functions.

Reference: J. Duchi, S. Shalev-Shwartz, Y. Singer, T. Chandra,
"Efficient Projections onto the l1-Ball for Learning in High
Dimensions", ICML 2008.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def project_l1_ball(x: jax.Array, radius: float | jax.Array) -> jax.Array:
    """Project the 1-D vector ``x`` onto ``{z : ‖z‖_1 <= radius}``.

    Solves ``argmin_z ‖z − x‖_2`` by the sort-based Duchi et al. (2008)
    algorithm, ``O(d log d)``. Vectors already inside the ball are
    returned bit-exactly unchanged.

    Branchless, so ``radius`` may be traced; callers deriving it at
    runtime depend on this.
    """

    # Radius is non-negative
    if not isinstance(radius, jax.core.Tracer):
        assert radius >= 0, f"radius must be non-negative, got {radius}."

    abs_x = jnp.abs(x)
    d = x.shape[0]

    # Sort |x| descending and form the running cumulative sum.
    u = jnp.sort(abs_x)[::-1]
    cssv = jnp.cumsum(u)

    # rho = number of coordinates k (1-indexed) with u_k * k > cssv_k − radius.
    k = jnp.arange(1, d + 1, dtype=cssv.dtype)
    rho = jnp.sum(u * k > (cssv - radius))
    rho = jnp.maximum(rho, 1)  # guard the all-zero case

    theta = jnp.maximum((cssv[rho - 1] - radius) / rho, 0.0)
    projected = jnp.sign(x) * jnp.maximum(abs_x - theta, 0.0)

    l1norm = jnp.sum(abs_x)
    return jnp.where(l1norm <= radius, x, projected)


def project_l1_ball_pytree(pytree: Any, radius: float | jax.Array) -> Any:
    """Project a whole pytree onto one global ``l_1`` ball.

    Leaves are flattened into a single vector, projected together, and
    unflattened. The constraint holds across the concatenation of every
    leaf, not per-leaf.
    """
    flat, unravel = ravel_pytree(pytree)
    return unravel(project_l1_ball(flat, radius))
