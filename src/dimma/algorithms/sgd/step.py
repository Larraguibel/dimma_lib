"""One SGD iteration: gradient, then descend.

One function, not two: this algorithm composes no mechanism, so there
is no release to split off — ADR-0006.

Everything here compiles; stage 1 is `train`'s. `grad_fn` and
`optimizer` are *static* `jax.jit` arguments. Bind them outside the
loop::

    from functools import partial

    compiled = jax.jit(partial(
        step.step, gradients.batch_grads(loss), updates.sgd(0.1),
    ))
"""

from __future__ import annotations

from typing import Any, Callable

import jax

from dimma.core import updates


def step(
    grad_fn: Callable,
    optimizer: updates.Optimizer,
    params: Any,
    opt_state: updates.OptState,
    x_batch: jax.Array,
    y_batch: jax.Array,
) -> tuple[Any, updates.OptState]:
    """One iteration: ``theta_{t+1} <- theta_t - eta_t g_t``.

    Parameters
    ----------
    grad_fn
        From `dimma.core.gradients.batch_grads`. Built once outside the
        loop, so `jax.jit` sees a stable compilation key.
    optimizer
        `dimma.core.updates.sgd`, the same object DP-SGD is given.
        Another rule makes this a different baseline, and the caller's
        to report.
    x_batch, y_batch
        The batch, leading axis ``batch_size``. No mask: shuffled
        sampling has fixed cardinality, so nothing is padded.

    Returns
    -------
    (params, opt_state)

    Notes
    -----
    No divisor appears: `batch_grads` differentiates the *mean* loss.
    DP-SGD's divisor is the expected lot size, which has to be a
    constant rather than the batch drawn; here the batch is the batch.
    """
    grad = grad_fn(params, x_batch, y_batch)
    return updates.apply(optimizer, params, grad, opt_state)
