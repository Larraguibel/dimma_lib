"""The SGD training loop: the baseline's ``for t in [T]``.

This module owns stage 1 and threads the two pieces of state the loop
carries. Sampling is host-side as in the other loops, though here
nothing forces it there — a shuffled draw has fixed cardinality and
would trace. Everything after the draw is one jitted call.

One random stream. Nothing is noised, so there is no second seed to get
wrong.

No metrics. In DP-SGD that is a privacy rule; here it is comparability
— a baseline reporting a training curve the private arm cannot would be
a second difference between the two runs.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
import numpy as np

from dimma.algorithms.sgd import step as _step
from dimma.core import gradients, updates
from dimma.core.sampling import shuffled


def train(
    per_sample_loss_fn: Callable,
    params: Any,
    optimizer: updates.Optimizer,
    x: jax.Array,
    y: jax.Array,
    rng: np.random.Generator,
    *,
    steps: int,
    batch_size: int,
) -> Any:
    """Run non-private SGD for ``steps`` iterations.

    Parameters
    ----------
    per_sample_loss_fn
        ``(params, x_single, y_single) -> scalar``, the same loss the
        private arm is given. Vectorized and averaged here once,
        outside the loop, so `jax.jit` traces a single time.
    optimizer
        `updates.sgd(eta)`, or `updates.sgd(schedule)` for a
        step-dependent rate. Whatever the private arm gets — ADR-0002.
    rng
        The sampling stream. Pass one generator for the whole run.
    steps
        Optimizer updates, not epochs — the unit the private arm is
        counted in. One epoch is ``len(x) // batch_size`` of them.
    batch_size
        Examples per step, exactly. No expected batch size: a shuffled
        draw has fixed cardinality.

    Returns
    -------
    params : Any

    No optimizer state is returned, matching DP-SGD.

    Raises
    ------
    ValueError
        From :func:`shuffled.batches` if ``batch_size`` is outside
        ``(0, len(x)]``, before the first step.
    """
    n = x.shape[0]
    grad_fn = gradients.batch_grads(per_sample_loss_fn)
    compiled = jax.jit(partial(_step.step, grad_fn, optimizer))

    opt_state = updates.init(optimizer, params)
    stream = shuffled.batches(rng, n, batch_size)
    for _ in range(steps):
        indices = next(stream)
        params, opt_state = compiled(
            params, opt_state, x[indices], y[indices],
        )
    return params
