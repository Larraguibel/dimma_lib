"""Stage 4 - clipping.

Bound each per-sample gradient's norm, bounding the sensitivity of the
aggregate that follows. Leaves have shape ``(B, *param_shape)``, the
layout stage 3 produces.

Makes no privacy claim: clipping bounds sensitivity, the guarantee
comes from stage 6; ADR-0003.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def per_sample_norms(per_sample_pytree) -> jax.Array:
    """Return each per-sample slice's global ``l_2`` norm.

    Parameters
    ----------
    per_sample_pytree : pytree
        Leaves of shape ``(B, *param_shape)``.

    Returns
    -------
    jax.Array, shape ``(B,)``
        One norm per sample, taken across all leaves and every
        non-batch dimension.
    """
    leaves = jax.tree_util.tree_leaves(per_sample_pytree)
    sq = [jnp.sum(leaf.reshape(leaf.shape[0], -1) ** 2, axis=1) for leaf in leaves]
    return jnp.sqrt(sum(sq))


def per_sample_clip(per_sample_pytree, clip_norm: float | jax.Array):
    """Rescale each per-sample slice to global ``l_2`` norm at most ``clip_norm``.

    Parameters
    ----------
    per_sample_pytree : pytree
        Leaves of shape ``(B, *param_shape)``.
    clip_norm : float > 0, in the units of the gradient
        The bound each slice's norm is brought within. May be traced,
        so no Python-level comparison is made on it.

    Returns
    -------
    pytree
        Same structure and shapes, slice ``i`` scaled by
        ``min(1, clip_norm / (norm_i + 1e-12))``. No slice comes out
        with a norm above ``clip_norm``, and an all-zero slice comes
        out unchanged.

    Notes
    -----
    The ``1e-12`` keeps an all-zero slice from dividing by zero. It
    also lands a clipped slice a negligible amount inside the bound
    rather than exactly on it, which is the standard distortion in DP
    practice.
    """
    norms = per_sample_norms(per_sample_pytree)
    factor = jnp.minimum(1.0, clip_norm / (norms + 1e-12))  # (B,)

    def _scale(leaf):
        shape = (leaf.shape[0],) + (1,) * (leaf.ndim - 1)
        return leaf * factor.reshape(shape)

    return jax.tree.map(_scale, per_sample_pytree)
