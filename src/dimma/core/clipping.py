"""Stage 4 - clipping.

Bound each per-sample gradient's norm, bounding the sensitivity of the
aggregate that follows. Leaves have shape ``(B, *param_shape)``, the
layout stage 3 produces.

Makes no privacy claim: clipping bounds sensitivity, the guarantee
comes from stage 6.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def per_sample_norms(per_sample_pytree) -> jax.Array:
    """Per-sample ``l_2`` norms across all leaves and non-batch dims.

    Returns shape ``(B,)``.
    """
    leaves = jax.tree_util.tree_leaves(per_sample_pytree)
    sq = [jnp.sum(leaf.reshape(leaf.shape[0], -1) ** 2, axis=1) for leaf in leaves]
    return jnp.sqrt(sum(sq))


def per_sample_clip(per_sample_pytree, clip_norm: float | jax.Array):
    """Rescale each per-sample slice to global ``l_2`` norm at most ``clip_norm``.

    ``clip_norm`` may be traced, so no Python-level comparison is made
    on it. The ``+ 1e-12`` term avoids division by zero on all-zero
    gradients at the cost of a negligible bias, a standard distortion in
    DP practice.
    """
    norms = per_sample_norms(per_sample_pytree)
    factor = jnp.minimum(1.0, clip_norm / (norms + 1e-12))  # (B,)

    def _scale(leaf):
        shape = (leaf.shape[0],) + (1,) * (leaf.ndim - 1)
        return leaf * factor.reshape(shape)

    return jax.tree.map(_scale, per_sample_pytree)
