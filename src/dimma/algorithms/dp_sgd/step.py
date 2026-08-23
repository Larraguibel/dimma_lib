"""One DP-SGD iteration: the body of Algorithm 1's loop.

Stage 1 is not here. Poisson subsampling has data-dependent cardinality
and runs on the host, so the loop draws the batch and passes this module
a fixed-shape batch plus its mask. Everything in this module compiles.

The two functions are one seam apart. `privatized_gradient` produces
``g~_t``, which is the only thing the mechanism releases and therefore
the only thing an accountant accounts for; `step` applies it, which is
post-processing and free. Splitting there keeps what is privatized
separable from what is done with it.

`grad_fn` and `optimizer` are *static* `jax.jit` arguments;
`dimma.core.updates.Optimizer` says why. Bind them once outside the
loop::

    from functools import partial

    compiled = jax.jit(partial(
        step.step, gradients.per_sample_grads(loss), updates.sgd(0.1),
        expected_batch_size=256, clip_norm=1.0, noise_multiplier=1.1,
    ))
"""

from __future__ import annotations

from typing import Any, Callable

import jax

from dimma.core import aggregation, clipping, noise, pytree, updates


def privatized_gradient(
    grad_fn: Callable,
    params: Any,
    x_batch: jax.Array,
    y_batch: jax.Array,
    mask: jax.Array,
    key: jax.Array,
    *,
    expected_batch_size: float,
    clip_norm: float,
    noise_multiplier: float,
) -> Any:
    """Algorithm 1's ``g~_t``, the privatized gradient estimate.

    Stages 3 through 6, in the paper's order::

        g~_t = (1/L) ( sum_i g_t(x_i)/max(1, |g_t(x_i)|_2 / C)
                       + N(0, sigma^2 C^2 I) )

    Noise is added to the *sum* at scale ``sigma * C`` and the result
    divided by ``L``, rather than the algebraically identical
    ``sigma * C / L`` on the mean. The sensitivity bound ``C`` is a
    property of the sum, and writing it this way keeps that visible.

    Parameters
    ----------
    grad_fn
        From `dimma.core.gradients.per_sample_grads`. Built once, so
        `jax.jit` sees a stable compilation key.
    params
        ``theta_t``: the parameters every per-sample gradient is taken
        at, a pytree of float arrays.
    x_batch, y_batch
        The padded batch, leading axis ``b_max``.
    mask
        Shape ``(b_max,)``, 1.0 for a real example and 0.0 for padding.
    key
        A `jax.random` key for this step's Gaussian draw. One fresh
        subkey per step: reusing one releases the same noise twice and
        the accounting assumes independent draws.
    expected_batch_size
        Algorithm 1's ``L``: the *expected* lot size ``q * N``, a
        constant. Not the leading axis length and not ``mask.sum()``,
        both of which are data-dependent and would leak.
    clip_norm
        Algorithm 1's ``C``, the per-example ``l_2`` bound.
    noise_multiplier
        Algorithm 1's ``sigma``. The standard deviation actually added
        is ``sigma * C``, so the noise tracks the sensitivity and
        ``sigma`` alone determines the privacy cost.

    Returns
    -------
    pytree
        ``g~_t``, in ``params``'s structure. This is the mechanism's
        release: everything downstream of it is post-processing.
    """
    per_sample = grad_fn(params, x_batch, y_batch)
    clipped = clipping.per_sample_clip(per_sample, clip_norm)
    summed = aggregation.sum_over_batch(clipped, mask=mask)
    perturbed = noise.add_gaussian(summed, key, noise_multiplier * clip_norm)
    return pytree.scale(perturbed, 1.0 / expected_batch_size)


def step(
    grad_fn: Callable,
    optimizer: updates.Optimizer,
    params: Any,
    opt_state: updates.OptState,
    x_batch: jax.Array,
    y_batch: jax.Array,
    mask: jax.Array,
    key: jax.Array,
    *,
    expected_batch_size: float,
    clip_norm: float,
    noise_multiplier: float,
) -> tuple[Any, updates.OptState]:
    """One full iteration: privatize, then descend.

    Algorithm 1's descent is ``theta_{t+1} <- theta_t - eta_t g~_t``,
    which is `dimma.core.updates.sgd`. A schedule gives the ``eta_t``
    subscript; any other optimizer departs from the paper, which is
    the caller's call to make and to report.

    Parameters
    ----------
    grad_fn
        From `dimma.core.gradients.per_sample_grads`. Built once, so
        `jax.jit` sees a stable compilation key.
    optimizer
        Algorithm 1 is ``updates.sgd(eta)``. Anything else is the
        caller's departure to report.
    params
        ``theta_t``: the current parameters, a pytree of float arrays.
    opt_state
        The optimizer's state, as `dimma.core.updates.init` built it and
        the previous call returned it.
    x_batch, y_batch
        The padded batch, leading axis ``b_max``.
    mask
        Shape ``(b_max,)``, 1.0 for a real example and 0.0 for padding.
    key
        A `jax.random` key for this step's Gaussian draw; one fresh
        subkey per step.
    expected_batch_size
        Algorithm 1's ``L``, the *expected* lot size ``q * N``.
    clip_norm
        Algorithm 1's ``C``, the per-example ``l_2`` bound.
    noise_multiplier
        Algorithm 1's ``sigma``, the standard deviation over ``C``.

    Returns
    -------
    params : pytree
        ``theta_{t+1}``, in ``params``'s structure.
    opt_state : updates.OptState
        The optimizer's state after this update. Derived from ``g~_t``,
        so it is post-processing and costs no privacy budget.

    See Also
    --------
    privatized_gradient : The release this applies, and where the
        privacy-relevant arguments are documented in full.
    """
    grad = privatized_gradient(
        grad_fn, params, x_batch, y_batch, mask, key,
        expected_batch_size=expected_batch_size, clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
    )
    return updates.apply(optimizer, params, grad, opt_state)
