"""A small learnable regression problem, shared by the SpiderBoost tests.

The same problem the DP-SGD suite uses, kept local to each algorithm's
tests rather than shared: a fixture the two suites had in common would
be one more thing a change to either has to keep true of both.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest


def squared_error(params, x, y):
    """A dimma per-sample loss: one example, scalar out."""
    residual = jnp.dot(params["w"], x) - y
    return 0.5 * residual ** 2


@pytest.fixture
def problem():
    """``(x, y, w_true)`` with a signal to recover, unlike pure noise."""
    gen = np.random.default_rng(0)
    w_true = np.array([1.5, -0.5, 2.0])
    x = gen.normal(size=(600, 3))
    y = x @ w_true + 0.1 * gen.normal(size=600)
    return (jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.float32),
            jnp.asarray(w_true, jnp.float32))


@pytest.fixture
def zero_params():
    return {"w": jnp.zeros(3)}


@pytest.fixture
def moved_params():
    """A second parameter set, standing in for ``w_t`` against ``w_{t-1}``.

    The variation branch is only exercised where the two differ: at
    equal parameters its release is noise around zero and its noise
    scale is zero, which is the one case that hides a wrong sign.
    """
    return {"w": jnp.array([0.3, -0.1, 0.2])}
