"""The DP-SGD training loop: Algorithm 1's ``for t in [T]``.

This module owns stage 1 and the two streams of state the loop carries.
Sampling is host-side: Poisson draws have data-dependent cardinality, so
they cannot live inside a compiled step. Everything after the draw is
one jitted call, traced once and reused for every step.

Two random streams, deliberately separate. The `numpy.random.Generator`
drives sampling and the `jax` key drives the Gaussian noise; they are
different mechanisms with different accounting roles, and one generator
each keeps a run reproducible from its two seeds.

The loop returns parameters and reports no metrics. Evaluating a model
on the training data is another access to it and costs privacy budget
that Algorithm 1 does not account for, so that call belongs to the
caller, where it is visible, and not to a callback hidden in here.
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
    optimizer: updates.GradientTransformation,
    x: jax.Array,
    y: jax.Array,
    key: jax.Array,
    rng: np.random.Generator,
    *,
    steps: int,
    lot_size: int,
    clip_norm: float,
    noise_multiplier: float,
    b_max: int | None = None,
) -> tuple[Any, updates.OptState]:
    """Run Algorithm 1 for ``steps`` iterations.

    Parameters
    ----------
    per_sample_loss_fn
        ``(params, x_single, y_single) -> scalar``. Vectorized here
        once, outside the loop, so `jax.jit` traces a single time.
    optimizer
        Algorithm 1 is ``optax.sgd(eta)``, or ``optax.sgd(schedule)``
        for the ``eta_t`` subscript. Anything else departs from the
        paper and is the caller's to report.
    key, rng
        The noise and sampling streams. Pass one ``rng`` for the whole
        run: independent draws are what the sampling assumption means.
    steps
        Algorithm 1's ``T``. Steps, not epochs - privacy composes over
        optimizer steps. One epoch is ``len(x) / lot_size`` of them.
    lot_size
        Algorithm 1's ``L``. Sets the sampling rate ``q = L / len(x)``
        and is the constant the sum is divided by.
    clip_norm, noise_multiplier
        Algorithm 1's ``C`` and ``sigma``.
    b_max
        Padding cap for the drawn batch, default
        `poisson.padded_batch_size(lot_size, len(x))`. Not a privacy
        parameter. Passing ``len(x)`` makes it exact and unraisable at
        the cost of an ``O(n)`` batch.

    Returns
    -------
    params : Any
        The trained parameters, Algorithm 1's ``theta_T``.
    opt_state : optax.OptState
        The final optimizer state, so a run can be continued.

    Notes
    -----
    No privacy cost is returned. It is a function of ``q``,
    ``noise_multiplier``, ``steps`` and a target ``delta``, and this
    loop is not in a position to claim it; pass those to an accountant.

    Raises
    ------
    RuntimeError
        Propagated from :func:`poisson.subsample` if a draw exceeds
        ``b_max``. Catching it would mean truncating or redrawing, and
        both change the mechanism the accounting assumes. Raise ``b_max``
        instead.
    """
    n = x.shape[0]
    if not 0 < lot_size <= n:
        raise ValueError(
            f"lot_size={lot_size} must be in (0, n] with n={n}; the "
            f"sampling rate q = lot_size / n is a probability."
        )
    if b_max is None:
        b_max = poisson.padded_batch_size(lot_size, n)

    q = lot_size / n
    grad_fn = gradients.per_sample_grads(per_sample_loss_fn)
    compiled = jax.jit(partial(
        _step.step, grad_fn, optimizer,
        lot_size=lot_size, clip_norm=clip_norm,
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
    return params, opt_state
