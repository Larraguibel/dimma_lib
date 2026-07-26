"""Stage 5 - aggregation.

Sum or average the per-sample gradients from stage 3.

Stage 1 samplers with data-dependent cardinality return a fixed-shape
batch plus a mask marking real slots. Zeroing the padding belongs to
this stage, so the mask is a parameter of the functions that sum rather
than a separate step. It commutes with stage 4's rescaling.

Makes no privacy claim.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dimma.core import pytree


def _apply_mask(per_sample_pytree, mask: jax.Array):
    """Multiply each per-sample slice by ``mask[i]``."""

    def _scale(leaf):
        shape = (leaf.shape[0],) + (1,) * (leaf.ndim - 1)
        return leaf * mask.reshape(shape)

    return jax.tree.map(_scale, per_sample_pytree)


def sum_over_batch(per_sample_pytree, mask: jax.Array | None = None):
    """Sum along the leading (batch) axis, zeroing masked-out slots.

    ``mask`` has shape ``(B,)``, typically in ``{0.0, 1.0}``. Omit it
    when the batch has no padding.
    """
    if mask is not None:
        per_sample_pytree = _apply_mask(per_sample_pytree, mask)
    return jax.tree.map(lambda leaf: jnp.sum(leaf, axis=0), per_sample_pytree)


def average_over_batch(per_sample_pytree, batch_size: float | jax.Array,
                       mask: jax.Array | None = None):
    """Sum over the batch axis and divide by ``batch_size``.

    ``batch_size`` is caller-supplied, not the leading axis length and
    not the number of unmasked entries. Algorithms subsampling with
    random cardinality normally pass the *expected* batch size; which to
    pass follows from the algorithm's analysis, so this stage takes it
    as a parameter.
    """
    return pytree.scale(
        sum_over_batch(per_sample_pytree, mask), 1.0 / batch_size
    )
