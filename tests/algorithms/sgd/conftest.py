"""A small learnable regression problem, shared by the SGD tests.

Deliberately the DP-SGD suite's numbers, kept local rather than shared:
the baseline's claim is that it differs from its private counterpart in
the privacy alone, so both arms need one problem underneath them.
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
