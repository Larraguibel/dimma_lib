"""Stage 6 - perturbation.

Add iid noise of a given scale to every leaf of a pytree. Both
distributions share one traced code path and differ only in the
``jax.random`` sampler passed to it, which is why they share a module.
A distribution needing more than a sampler swap (discrete Gaussian,
Skellam) would earn its own module.

Makes no privacy claim: the caller calibrates the scale; ADR-0003.
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
    """Add iid ``N(0, std^2)`` noise to every leaf.

    Parameters
    ----------
    pytree : pytree of jax.Array
        The quantity to perturb; leaves of any shape and dtype.
    key : jax.Array
        A PRNG key. It is split once into one subkey per leaf and is
        wholly consumed, so a caller threading randomness through a
        loop passes a fresh key each call: the same key twice adds the
        same noise twice.
    std : float >= 0, in the units of ``pytree``
        Standard deviation per coordinate. May be traced.

    Returns
    -------
    pytree
        Same structure, leaf shapes and leaf dtypes, each perturbed.
    """
    return _add_noise(pytree, key, std, jax.random.normal)


def add_laplace(pytree, key: jax.Array, scale: float | jax.Array):
    """Add iid ``Lap(0, scale)`` noise to every leaf.

    Parameters
    ----------
    pytree : pytree of jax.Array
        The quantity to perturb; leaves of any shape and dtype.
    key : jax.Array
        A PRNG key, split into one subkey per leaf and wholly consumed,
        as in :func:`add_gaussian`.
    scale : float >= 0, in the units of ``pytree``
        The Laplace ``b`` parameter, not the standard deviation; each
        coordinate has variance ``2 * scale ** 2``. May be traced.

    Returns
    -------
    pytree
        Same structure, leaf shapes and leaf dtypes, each perturbed.
    """
    return _add_noise(pytree, key, scale, jax.random.laplace)
