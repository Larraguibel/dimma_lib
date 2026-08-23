"""The DP-SGD training loop: Algorithm 1's ``for t in [T]``.

This module owns stage 1 and the two streams of state the loop carries.
Sampling is host-side: Poisson draws have data-dependent cardinality, so
they cannot live inside a compiled step. Everything after the draw is
one jitted call, traced once and reused for every step.

Two random streams, deliberately separate. The `numpy.random.Generator`
drives sampling and the `jax` key drives the Gaussian noise; they are
different mechanisms with different accounting roles, and one generator
each keeps a run reproducible from its two seeds.

The loop returns parameters and reports no metrics; see ADR-0006.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from dimma.algorithms.dp_sgd import step as _step
from dimma.core import gradients, updates
from dimma.core.sampling import poisson


def train(
    per_sample_loss_fn: Callable,
    params: Any,
    optimizer: updates.Optimizer,
    x: jax.Array,
    y: jax.Array,
    key: jax.Array,
    rng: np.random.Generator,
    *,
    steps: int,
    expected_batch_size: int,
    clip_norm: float,
    noise_multiplier: float,
    b_max: int | None = None,
) -> Any:
    """Run Algorithm 1 for ``steps`` iterations.

    Parameters
    ----------
    per_sample_loss_fn
        ``(params, x_single, y_single) -> scalar``. Vectorized here
        once, outside the loop, so `jax.jit` traces a single time.
    params
        ``theta_0``: the initial parameters, a pytree of float arrays.
        Not mutated.
    optimizer
        Algorithm 1 is ``updates.sgd(eta)``, or ``updates.sgd(schedule)``
        for the ``eta_t`` subscript. Anything else departs from the
        paper and is the caller's to report.
    x, y
        The training set, leading axis ``n`` on both. ``n`` is the
        denominator of the sampling rate every accounting claim is
        stated over.
    key, rng
        The noise and sampling streams. Pass one ``rng`` for the whole
        run: independent draws are what the sampling assumption means.
    steps
        Algorithm 1's ``T``. Steps, not epochs - privacy composes over
        optimizer steps. One epoch is ``len(x) / expected_batch_size``
        of them.
    expected_batch_size
        Algorithm 1's ``L``. Sets the sampling rate ``q = L / len(x)``
        and is the constant the sum is divided by.
    clip_norm, noise_multiplier
        Algorithm 1's ``C`` and ``sigma``.
    b_max
        Padding cap for the drawn batch, default
        `poisson.padded_batch_size(expected_batch_size, len(x))`. Not a
        privacy parameter. Passing ``len(x)`` makes it exact and
        unraisable at the cost of an ``O(n)`` batch.

    Returns
    -------
    params : pytree
        The trained parameters, Algorithm 1's ``theta_T``, in the
        initial parameters' structure and dtypes.

    Raises
    ------
    ValueError
        If ``expected_batch_size`` is outside ``(0, len(x)]``, before
        the first step.
    RuntimeError
        Propagated from :func:`poisson.subsample` if a draw exceeds
        ``b_max``. Raise ``b_max`` instead of catching it; ADR-0007
        records why.

    Notes
    -----
    No privacy cost is returned. It is a function of ``q``,
    ``noise_multiplier``, ``steps`` and a target ``delta``, and this
    loop is not in a position to claim it; pass those to an accountant.

    No optimizer state is returned. `train` accepts none, so it cannot
    consume what it would hand back, and a caller who resumed from it
    would replay this run's noise stream from the start. The other
    training loops follow this module on that point.
    """
    n = x.shape[0]
    if not 0 < expected_batch_size <= n:
        raise ValueError(
            f"expected_batch_size={expected_batch_size} must be in "
            f"(0, n] with n={n}; the sampling rate "
            f"q = expected_batch_size / n is a probability."
        )
    if b_max is None:
        b_max = poisson.padded_batch_size(expected_batch_size, n)

    q = expected_batch_size / n
    grad_fn = gradients.per_sample_grads(per_sample_loss_fn)
    compiled = jax.jit(partial(
        _step.step, grad_fn, optimizer,
        expected_batch_size=expected_batch_size, clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
    ))

    opt_state = updates.init(optimizer, params)
    for _ in range(steps):
        indices, mask = poisson.subsample(rng, n, q, b_max)
        key, subkey = jax.random.split(key)
        params, opt_state = compiled(
            params, opt_state, x[indices], y[indices],
            jnp.asarray(mask), subkey,
        )
    return params
