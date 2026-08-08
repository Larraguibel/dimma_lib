"""One Private SpiderBoost iteration: either branch of Algorithm 2's loop.

Stage 1 is not here: a Poisson draw has data-dependent cardinality and
runs on the host, so the loop passes this module a fixed-shape batch
plus its mask. Everything here compiles.

Two mechanisms, so four functions. `anchor_release` and
`variation_release` return what their mechanism makes public, and so
all an accountant accounts for; `anchor_step` and `variation_step`
apply them, which is post-processing and free. The two releases are
different quantities - a privatized gradient and a privatized increment
- and are accounted separately.

Nothing clips; the sensitivity comes from the assumed function class,
per ADR-0009. The variation branch's noise scale,
``min(rate * ||w_t - w_{t-1}||, cap)``, depends on prior releases only,
so it is chosen adaptively from public information.

`grad_fn` and `optimizer` are *static* `jax.jit` arguments rather than
traced ones - one is a function, the other a pair of them. Bind them
once outside the loop, per branch::

    from functools import partial

    compiled_anchor = jax.jit(partial(
        step.anchor_step, gradients.per_sample_grads(loss), updates.sgd(0.1),
        expected_batch_size=256, noise_scale=0.4,
    ))
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from dimma.core import aggregation, noise, pytree, updates


def anchor_release(
    grad_fn: Callable,
    params: Any,
    x_batch: jax.Array,
    y_batch: jax.Array,
    mask: jax.Array,
    key: jax.Array,
    *,
    expected_batch_size: float,
    noise_scale: float,
):
    """Algorithm 2's line 9: what the anchor mechanism releases.

    ::

        nabla_t = (1/b_1) sum_{x in S_t} grad f(w_t; x) + g_t,
        g_t ~ N(0, I sigma_1^2)

    The mechanism is a Gaussian mechanism over the per-sample gradient
    mean. Its ``l_2`` sensitivity rests on the Lipschitz assumption,
    which bounds a per-sample gradient with no operation making it so —
    there is no clipping line, per ADR-0009. Turning that bound into a
    scale is the accountant's; this function takes the scale, and adds
    it to the *mean* rather than to the sum, as the paper writes it.

    Parameters
    ----------
    grad_fn
        From `dimma.core.gradients.per_sample_grads`. Built once, so
        `jax.jit` sees a stable compilation key.
    x_batch, y_batch
        The padded batch, leading axis the anchor branch's padding cap.
    mask
        Shape ``(b_max,)``, 1.0 for a real example and 0.0 for padding.
    expected_batch_size
        Algorithm 2's ``b_1``. The constant the sum is divided by, fixed
        before the run. Not the leading axis length and not
        ``mask.sum()``, both of which are data-dependent and would leak.
    noise_scale
        Algorithm 2's ``sigma_1``.
    """
    per_sample = grad_fn(params, x_batch, y_batch)
    mean = aggregation.average_over_batch(
        per_sample, expected_batch_size, mask=mask
    )
    return noise.add_gaussian(mean, key, noise_scale)


def variation_release(
    grad_fn: Callable,
    params: Any,
    previous_params: Any,
    x_batch: jax.Array,
    y_batch: jax.Array,
    mask: jax.Array,
    key: jax.Array,
    *,
    expected_batch_size: float,
    noise_rate: float,
    noise_cap: float,
):
    """Algorithm 2's line 13: what the variation mechanism releases.

    ::

        Delta_t = (1/b_2) sum_{x in S_t} [grad f(w_t; x)
                                          - grad f(w_{t-1}; x)] + g_t,
        g_t ~ N(0, I min(sigma_2^2 ||w_t - w_{t-1}||^2, sigma-hat_2^2))

    An increment, not a gradient. The running estimate is formed from it
    by `variation_step`, on the post-processing side of the seam.

    A different mechanism from the anchor's, and accounted separately:
    it aggregates gradient *differences*, at its own sampling rate, and
    its sensitivity rests on the smoothness assumption rather than the
    Lipschitz one. Smoothness bounds a per-sample difference by
    ``L_1 ||w_t - w_{t-1}||``, which is why the scale tracks the
    distance moved; the Lipschitz bound ``2 L_0`` takes over when the
    parameters move far, which is what ``noise_cap`` is.

    Parameters
    ----------
    params, previous_params
        ``w_t`` and ``w_{t-1}``. Both are evaluated against the *same*
        batch, which is what makes the difference a variance-reduced
        estimate rather than two independent ones.
    expected_batch_size
        Algorithm 2's ``b_2``, a constant fixed before the run.
    noise_rate, noise_cap
        Algorithm 2's ``sigma_2`` and ``sigma-hat_2``. The standard
        deviation actually added is
        ``min(noise_rate * ||w_t - w_{t-1}||, noise_cap)``, where the
        norm is global across the parameter pytree.
    """
    per_sample = pytree.sub(
        grad_fn(params, x_batch, y_batch),
        grad_fn(previous_params, x_batch, y_batch),
    )
    mean = aggregation.average_over_batch(
        per_sample, expected_batch_size, mask=mask
    )
    distance = pytree.global_norm(pytree.sub(params, previous_params))
    scale = jnp.minimum(noise_rate * distance, noise_cap)
    return noise.add_gaussian(mean, key, scale)


def anchor_step(
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
    noise_scale: float,
) -> tuple[Any, Any, updates.OptState]:
    """An anchor iteration: release, then descend.

    The anchor release *is* the running estimate — Algorithm 2 assigns
    it to ``nabla_t`` outright, so a phase starts over rather than
    accumulating. Returns ``(params, estimate, opt_state)``; the
    estimate is returned because the loop carries it into the variation
    branch, and it is already released, so passing it on costs nothing.
    """
    estimate = anchor_release(
        grad_fn, params, x_batch, y_batch, mask, key,
        expected_batch_size=expected_batch_size, noise_scale=noise_scale,
    )
    params, opt_state = updates.apply(optimizer, params, estimate, opt_state)
    return params, estimate, opt_state


def variation_step(
    grad_fn: Callable,
    optimizer: updates.Optimizer,
    params: Any,
    previous_params: Any,
    previous_estimate: Any,
    opt_state: updates.OptState,
    x_batch: jax.Array,
    y_batch: jax.Array,
    mask: jax.Array,
    key: jax.Array,
    *,
    expected_batch_size: float,
    noise_rate: float,
    noise_cap: float,
) -> tuple[Any, Any, updates.OptState]:
    """A variation iteration: release, accumulate, then descend.

    Algorithm 2's lines 14 and 16::

        nabla_t = nabla_{t-1} + Delta_t
        w_{t+1} = w_t - eta nabla_t

    Both are post-processing. The accumulation is exactly addition of
    two already-released quantities, which is what keeps it on this side
    of the seam: nothing privacy-relevant happens after
    `variation_release` returns.

    Returns ``(params, estimate, opt_state)``.
    """
    increment = variation_release(
        grad_fn, params, previous_params, x_batch, y_batch, mask, key,
        expected_batch_size=expected_batch_size,
        noise_rate=noise_rate, noise_cap=noise_cap,
    )
    estimate = pytree.add(previous_estimate, increment)
    params, opt_state = updates.apply(optimizer, params, estimate, opt_state)
    return params, estimate, opt_state
