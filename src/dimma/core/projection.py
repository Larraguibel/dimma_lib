"""Euclidean projection onto the ``l_1`` ball, over arrays and pytrees.

Pure geometry, and makes no privacy claim; ADR-0003.

References
----------
.. [1] J. Duchi, S. Shalev-Shwartz, Y. Singer, T. Chandra, "Efficient
   Projections onto the l1-Ball for Learning in High Dimensions",
   ICML 2008.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def project_l1_ball(x: jax.Array, radius: float | jax.Array) -> jax.Array:
    """Project the 1-D vector ``x`` onto ``{z : ‖z‖_1 <= radius}``.

    Parameters
    ----------
    x : jax.Array, shape ``(d,)``
        The vector to project.
    radius : float >= 0, or a traced scalar
        The ball's radius. No array operation here branches on it, so a
        radius derived at run time may be traced; callers depend on
        that. A *concrete* radius is checked for non-negativity.

    Returns
    -------
    jax.Array, shape ``(d,)``
        ``argmin_z ‖z − x‖_2`` over the ball, by the sort-based Duchi
        et al. (2008) algorithm, ``O(d log d)``. A vector already
        inside the ball is returned bit-exactly unchanged.

    Raises
    ------
    AssertionError
        If ``radius`` is a concrete negative number. This is an
        ``assert``, so it is skipped under ``python -O``; a caller who
        needs the check enforced validates the radius itself, as
        `dimma.transforms.projection` does.
    """
    # Non-negative radius
    if isinstance(radius, (int, float)) and not isinstance(radius, bool):
        assert radius >= 0, f"radius must be non-negative, got {radius}."

    abs_x = jnp.abs(x)
    d = x.shape[0]

    # Sort |x| descending and form the running cumulative sum.
    u = jnp.sort(abs_x)[::-1]
    cssv = jnp.cumsum(u)

    # rho = number of coordinates k (1-indexed) with u_k * k > cssv_k − radius.
    k = jnp.arange(1, d + 1, dtype=cssv.dtype)
    rho = jnp.sum(u * k > (cssv - radius))
    rho = jnp.maximum(rho, 1)  # rho is 0 only at radius 0; keep the index valid

    theta = jnp.maximum((cssv[rho - 1] - radius) / rho, 0.0)
    projected = jnp.sign(x) * jnp.maximum(abs_x - theta, 0.0)

    l1norm = jnp.sum(abs_x)
    return jnp.where(l1norm <= radius, x, projected)


def project_l1_ball_pytree(pytree: Any, radius: float | jax.Array) -> Any:
    """Project a whole pytree onto one global ``l_1`` ball.

    Leaves are flattened into a single vector, projected together, and
    unflattened, so the constraint holds across the concatenation of
    every leaf and not per-leaf.

    Parameters
    ----------
    pytree : pytree of jax.Array
        The point to project.
    radius : float >= 0, or a traced scalar
        The ball's radius, as for :func:`project_l1_ball`.

    Returns
    -------
    pytree
        Same structure and leaf shapes, satisfying
        ``‖concat(leaves)‖_1 <= radius``.
    """
    flat, unravel = ravel_pytree(pytree)
    return unravel(project_l1_ball(flat, radius))
