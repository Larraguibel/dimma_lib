"""Stage 6 - perturbation.

Add iid noise of a given scale to every leaf of a pytree. Both
distributions share one traced code path and differ only in the
``jax.random`` sampler passed to it, which is why they share a module.
A distribution needing more than a sampler swap (discrete Gaussian,
Skellam) would earn its own module.

Makes no privacy claim: the caller calibrates the scale.
"""

from __future__ import annotations

import jax


def _add_noise(pytree, key: jax.Array, scale: float | jax.Array, sample_fn):
    """Add iid noise to every leaf via ``sample_fn(key, shape, dtype)``."""
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    keys = jax.random.split(key, len(leaves))
    noisy_leaves = [
        leaf + scale * sample_fn(k, leaf.shape, dtype=leaf.dtype)
        for leaf, k in zip(leaves, keys)
    ]
    return jax.tree_util.tree_unflatten(treedef, noisy_leaves)


def add_gaussian(pytree, key: jax.Array, std: float | jax.Array):
    """Add iid ``N(0, std^2)`` noise. ``std`` may be traced."""
    return _add_noise(pytree, key, std, jax.random.normal)


def add_laplace(pytree, key: jax.Array, scale: float | jax.Array):
    """Add iid ``Lap(0, scale)`` noise. ``scale`` may be traced.

    ``scale`` is the Laplace ``b`` parameter, not the standard
    deviation; each coordinate has variance ``2 * scale ** 2``.
    """
    return _add_noise(pytree, key, scale, jax.random.laplace)
