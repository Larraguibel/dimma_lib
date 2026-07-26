"""Stage 4 - clipping."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dimma.core import clipping

from tests.helpers import tree_equal


def test_per_sample_norms_match_manual_row_norms(per_sample_tree):
    """Norms span every leaf and every non-batch dim, per sample."""
    leaves = jax.tree_util.tree_leaves(per_sample_tree)
    manual = jnp.sqrt(
        sum(jnp.sum(leaf.reshape(leaf.shape[0], -1) ** 2, axis=1)
            for leaf in leaves)
    )
    assert jnp.allclose(clipping.per_sample_norms(per_sample_tree), manual)


def test_per_sample_norms_has_one_entry_per_sample(per_sample_tree):
    assert clipping.per_sample_norms(per_sample_tree).shape == (4,)


def test_clipping_bounds_every_per_sample_norm(per_sample_tree):
    clip_norm = 1.0
    clipped = clipping.per_sample_clip(per_sample_tree, clip_norm)
    norms = clipping.per_sample_norms(clipped)
    assert jnp.all(norms <= clip_norm + 1e-5)


def test_samples_inside_the_ball_are_untouched(per_sample_tree):
    """A sample under the threshold gets factor exactly 1.0, so no bias."""
    clipped = clipping.per_sample_clip(per_sample_tree, 10.0)
    assert tree_equal(clipped, per_sample_tree)


def test_clipping_preserves_direction(per_sample_tree):
    """Clipping rescales; it must not rotate the gradient."""
    clipped = clipping.per_sample_clip(per_sample_tree, 1.0)
    original_row = per_sample_tree["w"][0]
    clipped_row = clipped["w"][0]
    cosine = jnp.dot(original_row, clipped_row) / (
        jnp.linalg.norm(original_row) * jnp.linalg.norm(clipped_row)
    )
    assert jnp.allclose(cosine, 1.0, atol=1e-6)


def test_all_zero_sample_does_not_produce_nan(per_sample_tree):
    """Row 2 is all zeros; the 1e-12 term is what keeps this finite."""
    clipped = clipping.per_sample_clip(per_sample_tree, 1.0)
    assert all(
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree_util.tree_leaves(clipped)
    )
    assert jnp.all(clipped["w"][2] == 0.0)


def test_clip_norm_may_be_traced(per_sample_tree):
    """Kernels derive the threshold at runtime, so no Python comparison."""
    jitted = jax.jit(clipping.per_sample_clip)
    assert tree_equal(
        jitted(per_sample_tree, jnp.float32(1.0)),
        clipping.per_sample_clip(per_sample_tree, 1.0),
    )


def test_clipping_preserves_structure(per_sample_tree):
    clipped = clipping.per_sample_clip(per_sample_tree, 1.0)
    assert jax.tree_util.tree_structure(clipped) == \
        jax.tree_util.tree_structure(per_sample_tree)
