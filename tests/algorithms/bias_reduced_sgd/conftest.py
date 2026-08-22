"""A small *sparse* regression problem, shared by these tests.

The DP-SGD and SpiderBoost suites use a dense fixture; this one
deliberately does not. Assumption (A.7) — at most ``S`` nonzero
coordinates in every per-sample gradient — is what the whole method
exists for, and on dense rows both the projection's denoising and the
bias it costs are about nothing, so every bound this suite pins would
hold vacuously.

Each row of ``x`` has exactly ``S`` nonzeros out of ``D``, and the
per-sample gradient of the squared error is ``(<w, x> - y) * x``, whose
support is the row's. It is therefore exactly ``S``-sparse at every
parameter value, and ``clip_norm * sqrt(S)`` is a radius at which the
estimator's bound says something.

Kept local to this suite rather than shared with the other two: a
fixture three suites had in common would be one more thing a change to
any of them has to keep true of all.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

N, D, S = 64, 48, 4
"""Examples, coordinates, and nonzeros per row. ``N`` is a power of
two, so `dyadic.max_scale` is 5 and the ladder runs 2, 4, ..., 64."""


def squared_error(params, x, y):
    """A dimma per-sample loss: one example, scalar out."""
    residual = jnp.dot(params["w"], x) - y
    return 0.5 * residual ** 2


def sparse_rows(gen: np.random.Generator, n: int, d: int, s: int):
    """``(n, d)`` float32 rows with exactly ``s`` nonzeros each."""
    x = np.zeros((n, d), dtype=np.float32)
    for row in range(n):
        columns = gen.choice(d, size=s, replace=False)
        x[row, columns] = gen.standard_normal(s)
    return x


@pytest.fixture
def sparse_problem():
    """``(x, y, w_true)``: sparse rows, a signal to recover, light noise."""
    gen = np.random.default_rng(0)
    x = sparse_rows(gen, N, D, S)
    w_true = np.zeros(D, dtype=np.float32)
    w_true[gen.choice(D, size=S, replace=False)] = gen.standard_normal(S)
    y = (x @ w_true + 0.05 * gen.standard_normal(N)).astype(np.float32)
    return jnp.asarray(x), jnp.asarray(y), jnp.asarray(w_true)


@pytest.fixture
def zero_params():
    """The origin, where every per-sample gradient is ``-y * x``."""
    return {"w": jnp.zeros(D)}


@pytest.fixture
def moved_params():
    """A parameter set away from the origin, so a release is not a
    function of the labels alone."""
    gen = np.random.default_rng(7)
    w = np.zeros(D, dtype=np.float32)
    w[gen.choice(D, size=S, replace=False)] = 0.3 * gen.standard_normal(S)
    return {"w": jnp.asarray(w)}
