"""Shared fixtures for the dimma test suite."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def key() -> jax.Array:
    """A fixed PRNG key, so every test is reproducible."""
    return jax.random.key(0)


@pytest.fixture
def rng() -> np.random.Generator:
    """A fixed NumPy generator, for the stage 1 samplers."""
    return np.random.default_rng(0)


@pytest.fixture
def params() -> dict:
    """A small nested pytree standing in for model parameters."""
    return {
        "dense": {"w": jnp.array([1.0, -2.0, 3.0]), "b": jnp.array(0.5)},
        "head": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    }


@pytest.fixture
def per_sample_tree() -> dict:
    """A per-sample pytree: every leaf has leading batch axis B = 4.

    Row norms straddle 1.0 so clipping tests exercise both branches, and
    row 2 is all zeros to cover the degenerate case.
    """
    return {
        "w": jnp.array([
            [3.0, 4.0],     # norm 5
            [0.06, 0.08],   # norm 0.1
            [0.0, 0.0],     # norm 0
            [1.0, 0.0],     # norm 1
        ]),
        "b": jnp.array([0.0, 0.0, 0.0, 0.0]),
    }
