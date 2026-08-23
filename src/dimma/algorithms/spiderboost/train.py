"""The Private SpiderBoost training loop: Algorithm 2's ``for t = 0..T``.

This module owns stage 1, which is host-side because a Poisson draw's
cardinality is data-dependent. Each branch is one jitted call bound
before the loop, so a run compiles twice however long it is.

Four things are threaded: current parameters, previous parameters, the
running estimate and optimizer state. The previous parameters belong to
the estimator, not the optimizer - folding them in would make the
optimizer algorithm-specific and break the one stage both sides of a
comparison have to name identically.

Two random streams, as in DP-SGD: a `numpy.random.Generator` for
sampling and a `jax` key for the noise, so a run is reproducible from
two seeds. The output rule's index comes off the sampling generator
before the loop - it is host-side control flow, drawn from ``steps``
alone and so costing no budget, and a third stream would break the
two-seed contract.

No metrics; see ADR-0006.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from dimma.algorithms.spiderboost import step as _step
from dimma.core import gradients, updates
from dimma.core.sampling import poisson


def _check_batch_size(name: str, value: int, n: int) -> None:
    if not 0 < value <= n:
        raise ValueError(
            f"{name}={value} must be in (0, n] with n={n}; the sampling "
            f"rate q = {name} / n is a probability."
        )


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
    anchor_interval: int,
    anchor_expected_batch_size: int,
    variation_expected_batch_size: int,
    anchor_noise_scale: float,
    variation_noise_rate: float,
    variation_noise_cap: float,
    anchor_b_max: int | None = None,
    variation_b_max: int | None = None,
) -> tuple[Any, Any]:
    """Run Algorithm 2 for ``steps`` updates.

    Parameters
    ----------
    per_sample_loss_fn
        ``(params, x_single, y_single) -> scalar``. Vectorized here
        once, outside the loop, so each branch traces a single time.
        The same loss DP-SGD would be given, which is what lets one
        model be run under both.
    params
        ``w_0``: the initial parameters, a pytree of float arrays. Not
        mutated.
    optimizer
        Algorithm 2's line 16 is descent at a constant step size, which
        is ``updates.sgd(eta)``. Theorem B.3 fixes ``eta = 1/(2 L_1)``,
        which this loop cannot verify. Anything else departs from the
        paper: it leaves the privacy intact but drops the convergence
        guarantee, and it changes how far the parameters move per step,
        which feeds the variation branch's noise scale - so it changes
        the mechanism, not only what can be proved about it.
    x, y
        The training set, leading axis ``n`` on both. ``n`` is the
        denominator of both branches' sampling rates.
    key, rng
        The noise and sampling streams. Pass one ``rng`` for the whole
        run: independent draws are what the sampling assumption means.
    steps
        The number of optimizer updates, which for this algorithm is
        also the number of releases. Algorithm 2's ``T`` is
        ``steps - 1``; see the package docstring. Steps, not epochs -
        privacy composes over steps. Must be at least 2, or the output
        rule's support is empty.
    anchor_interval
        Algorithm 2's phase size ``q``: step ``t`` is an anchor step
        when ``t % anchor_interval == 0``, so step 0 always is. An
        interval of 1 makes every step an anchor and the algorithm
        degenerates to unclipped noisy SGD.
    anchor_expected_batch_size, variation_expected_batch_size
        Algorithm 2's ``b_1`` and ``b_2``. Each sets its branch's
        sampling rate ``q = b / len(x)`` and is the constant that
        branch's sum is divided by. Named for their branches because
        transposing them silently runs a different mechanism.
    anchor_noise_scale
        Algorithm 2's ``sigma_1``, the standard deviation added to the
        anchor branch's released mean.
    variation_noise_rate, variation_noise_cap
        Algorithm 2's ``sigma_2`` and ``sigma-hat_2``. The variation
        branch's standard deviation is
        ``min(rate * ||w_t - w_{t-1}||, cap)``.
    anchor_b_max, variation_b_max
        Padding caps for the two branches, defaulting per branch to
        `poisson.padded_batch_size(expected_batch_size, len(x))`.
        Separate, so a large anchor batch does not force a large
        variation batch. Not privacy parameters. Passing ``len(x)``
        makes a cap exact and unraisable at the cost of an ``O(n)``
        batch.

    Returns
    -------
    output_params : Any
        Algorithm 2's ``w-bar``: an iterate drawn uniformly from
        ``{w_1, ..., w_{steps-1}}``, every iterate the loop produced
        except the last. This is the algorithm's result — the one the
        convergence theorem bounds — and what a reported number should
        be computed from.
    final_params : Any
        The last iterate, ``w_steps``. Returned for comparison against
        the random one, and **carrying no stationarity bound**: no
        gradient estimate was ever taken at it, so the theorem says
        nothing about it. Do not report it as the algorithm's result.

    Raises
    ------
    ValueError
        If ``steps`` is below 2, if ``anchor_interval`` is below 1, or
        if either expected batch size is outside ``(0, len(x)]``.
    RuntimeError
        Propagated from :func:`poisson.subsample` if a draw exceeds its
        branch's cap. Raise the cap instead of catching it; ADR-0007
        records why.

    Notes
    -----
    No optimizer state is returned, as in
    `dimma.algorithms.dp_sgd.train`. Continuing a run here would
    further need the estimator's state and the phase position, so a
    subset would be worse than nothing.

    No privacy cost is returned, and none is claimed. The two branches
    are different mechanisms and compose separately: the anchor runs
    ``ceil(steps / anchor_interval)`` times at rate
    ``anchor_expected_batch_size / len(x)``, the variation the remaining
    times at ``variation_expected_batch_size / len(x)``. Those, with the
    three noise scales, are what an accountant takes — unconverted, so
    no translation step can silently disagree with what ran. What it
    cannot check is stated in the package docstring: the Lipschitz and
    smoothness constants the scales were calibrated against, and the
    step size.
    """
    n = x.shape[0]
    if steps < 2:
        raise ValueError(
            f"steps={steps} must be at least 2; the output rule draws "
            f"uniformly from the first steps - 1 iterates, and below 2 "
            f"that support is empty."
        )
    if anchor_interval < 1:
        raise ValueError(
            f"anchor_interval={anchor_interval} must be at least 1; it is "
            f"the phase length in steps."
        )
    _check_batch_size(
        "anchor_expected_batch_size", anchor_expected_batch_size, n
    )
    _check_batch_size(
        "variation_expected_batch_size", variation_expected_batch_size, n
    )
    if anchor_b_max is None:
        anchor_b_max = poisson.padded_batch_size(
            anchor_expected_batch_size, n
        )
    if variation_b_max is None:
        variation_b_max = poisson.padded_batch_size(
            variation_expected_batch_size, n
        )

    anchor_q = anchor_expected_batch_size / n
    variation_q = variation_expected_batch_size / n
    grad_fn = gradients.per_sample_grads(per_sample_loss_fn)
    anchor = jax.jit(partial(
        _step.anchor_step, grad_fn, optimizer,
        expected_batch_size=anchor_expected_batch_size,
        noise_scale=anchor_noise_scale,
    ))
    variation = jax.jit(partial(
        _step.variation_step, grad_fn, optimizer,
        expected_batch_size=variation_expected_batch_size,
        noise_rate=variation_noise_rate,
        noise_cap=variation_noise_cap,
    ))

    # Drawn before the loop, from the sampling stream: a host-side
    # integer for control flow, depending on `steps` and nothing in the
    # data, so it costs no budget. The support is w_1 .. w_{steps-1}.
    output_index = int(rng.integers(1, steps))

    opt_state = updates.init(optimizer, params)
    previous_params, estimate, output_params = None, None, None
    for t in range(steps):
        key, subkey = jax.random.split(key)
        if t % anchor_interval == 0:
            indices, mask = poisson.subsample(rng, n, anchor_q, anchor_b_max)
            new_params, estimate, opt_state = anchor(
                params, opt_state, x[indices], y[indices],
                jnp.asarray(mask), subkey,
            )
        else:
            indices, mask = poisson.subsample(
                rng, n, variation_q, variation_b_max
            )
            new_params, estimate, opt_state = variation(
                params, previous_params, estimate, opt_state,
                x[indices], y[indices], jnp.asarray(mask), subkey,
            )
        previous_params, params = params, new_params
        if t + 1 == output_index:
            output_params = params
    return output_params, params
