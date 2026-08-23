"""Stage 5 - aggregation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dimma.core import aggregation, clipping

from tests.helpers import tree_equal


def test_sum_without_mask_is_a_plain_batch_sum(per_sample_tree):
    out = aggregation.sum_over_batch(per_sample_tree)
    assert tree_equal(
        out, jax.tree.map(lambda leaf: jnp.sum(leaf, axis=0), per_sample_tree)
    )


def test_sum_drops_the_batch_axis(per_sample_tree):
    out = aggregation.sum_over_batch(per_sample_tree)
    assert out["w"].shape == (2,)
    assert out["b"].shape == ()


def test_mask_excludes_padded_slots(per_sample_tree):
    """Padding is index 0, a real row, so only the mask makes it harmless."""
    mask = jnp.array([1.0, 1.0, 0.0, 0.0])
    masked = aggregation.sum_over_batch(per_sample_tree, mask=mask)
    manual = jax.tree.map(lambda leaf: jnp.sum(leaf[:2], axis=0),
                          per_sample_tree)
    assert tree_equal(masked, manual)


def test_average_divides_by_the_given_batch_size(per_sample_tree):
    """Not the leading axis length, and not the number of unmasked rows.

    Poisson-subsampled algorithms pass the *expected* batch size, so the
    divisor has to be exactly what the caller supplied.
    """
    mask = jnp.array([1.0, 1.0, 0.0, 0.0])
    batch_size = 10.0
    out = aggregation.average_over_batch(
        per_sample_tree, batch_size, mask=mask
    )
    expected = jax.tree.map(
        lambda leaf: jnp.sum(leaf[:2], axis=0) / batch_size, per_sample_tree
    )
    assert tree_equal(out, expected)


def test_average_batch_size_may_be_traced(per_sample_tree):
    jitted = jax.jit(aggregation.average_over_batch)
    assert tree_equal(
        jitted(per_sample_tree, jnp.float32(4.0)),
        aggregation.average_over_batch(per_sample_tree, 4.0),
    )


def test_masking_commutes_with_clipping(per_sample_tree):
    """`dimma.core.aggregation`'s module docstring claims this; the
    kernels rely on it to reorder."""
    mask = jnp.array([1.0, 0.0, 1.0, 0.0])
    clip_then_mask = aggregation.sum_over_batch(
        clipping.per_sample_clip(per_sample_tree, 1.0), mask=mask
    )
    mask_then_clip = aggregation.sum_over_batch(
        clipping.per_sample_clip(
            jax.tree.map(
                lambda leaf: leaf * mask.reshape(
                    (leaf.shape[0],) + (1,) * (leaf.ndim - 1)
                ),
                per_sample_tree,
            ),
            1.0,
        )
    )
    assert tree_equal(clip_then_mask, mask_then_clip)


def test_full_mask_equals_no_mask(per_sample_tree):
    ones = jnp.ones(4)
    assert tree_equal(
        aggregation.sum_over_batch(per_sample_tree, mask=ones),
        aggregation.sum_over_batch(per_sample_tree),
    )


def test_zero_mask_sums_to_zero(per_sample_tree):
    out = aggregation.sum_over_batch(per_sample_tree, mask=jnp.zeros(4))
    assert all(
        jnp.all(leaf == 0) for leaf in jax.tree_util.tree_leaves(out)
    )
