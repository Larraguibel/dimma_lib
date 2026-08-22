"""The bias-reduced training loop: Algorithm 4's randomly-stopped ``while``.

This module owns stage 1 and the stopping rule, and the two are the
same thing. Every other loop in dimma runs a number of steps the caller
chose; here the caller states a privacy budget and the accountant
decides, step by step, whether another one may run. ``T`` is an output.

The order inside an iteration is the whole point and is fixed:

1. draw the scale — the public coin, which touches no data;
2. price the step from the coin alone;
3. ask the filter, and stop if it refuses;
4. only then draw the batch, its halves and the single record;
5. release, combine, update;
6. charge the step.

Steps 1 to 3 are arithmetic on a coin, so the budget check can never
depend on what it is protecting. Nothing indexes ``x`` above the line;
a refused step costs one draw of a scale and no gradient at all.

Two random streams, as in the other loops. The
`numpy.random.Generator` drives the coin, the batch draws and the
output rule's reservoir; the `jax` key drives the Gaussian noise. A run
is reproducible from those two seeds, and the realized number of steps
is a function of the sampling seed alone.

`train` imports `dimma.accounting`, which no other loop does. The
filter *is* the termination condition, so the alternative was inlining
its closed form here — which would put a privacy claim outside
`accounting/`, and ADR-0003 forbids that. ADR-0018 records it.

Parameters are carried as a host `numpy` float64 pytree for the reason
:mod:`~dimma.algorithms.bias_reduced_sgd.step` gives: the debias
combine is a near-cancellation amplified by ``1 / p_N``. The releases
are float32 all the same, converted per call.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import numpy as np

from dimma.accounting import bias_reduced_sgd as accounting
from dimma.algorithms.bias_reduced_sgd import estimators
from dimma.algorithms.bias_reduced_sgd import step as _step
from dimma.core import gradients, pytree, updates
from dimma.core.sampling import dyadic

__all__ = ["Run", "train"]


class Run(NamedTuple):
    """What Algorithm 4 produced, and what its coin spent.

    Three parameter sets, because the paper's two output rules bound
    two different things and neither of them is the last iterate.
    """

    average_params: Any
    """``x-bar``, the mean of ``x^0 ... x^T``. The convex rule: it is
    the iterate Theorem 5.6's excess-risk bound is stated for."""

    random_params: Any
    """``x_that``, one iterate drawn uniformly from ``x^0 ... x^T``.
    The nonconvex rule, and what the stationarity bound covers. Drawn
    by reservoir sampling off the sampling stream, because ``T`` is not
    known when the run starts."""

    final_params: Any
    """``x^{T+1}``, the last iterate. No bound covers it: no gradient
    estimate was ever taken at it. Returned for comparison, not for
    reporting."""

    steps: int
    """``T + 1``, the number of updates the filter actually admitted.
    An output of this algorithm, not an input to it."""

    spent: accounting.Spent
    """The filter's final state. A function of the public coin and the
    budget, so reporting it is not a metric — see the Notes."""


def train(
    per_sample_loss_fn: Callable,
    params: Any,
    optimizer: updates.Optimizer,
    x: jax.Array,
    y: jax.Array,
    key: jax.Array,
    rng: np.random.Generator,
    *,
    target_epsilon: float,
    target_delta: float,
    clip_norm: float,
    radius: float,
    estimator: Callable[..., estimators.MeanEstimator] = (
        estimators.projection_estimator
    ),
    max_scale: int | None = None,
    max_steps: int | None = None,
) -> Run:
    """Run Algorithm 4 until the privacy filter stops it.

    The same first seven positional arguments as
    `dimma.algorithms.dp_sgd.train` and
    `dimma.algorithms.spiderboost.train`, so one model and one loss run
    under all three. Three deliberate differences after them:

    **It takes a budget, not noise scales.** Every other loop here
    takes scales because its accountant sits upstream (ADR-0011's
    closing line). This one cannot: the budget *is* the termination
    condition, and Algorithm 4's own signature is ``(x_0, S, eps,
    delta)``. `train` calls
    `dimma.accounting.bias_reduced_sgd.inner_noise_multiplier` and
    hands the result to ``estimator``. Taking a budget *and* a
    multiplier would let a caller pass a pair that disagrees, and the
    run would then be filtered against one number and perturbed
    according to another.

    **``estimator`` is a factory, not a scale.** It is the seam. Which
    estimator ran is visible to the accountant through the *type* of
    its `estimators.MeanEstimator.claim`, and this loop passes that
    claim to `dimma.accounting.bias_reduced_sgd.check_claim` before it
    takes a step, so no accountant's assumptions ever sit on another
    estimator's code.

    **There is no ``steps`` argument.** ``T`` is an output, in
    `Run.steps`. That is the point of the filter: the random stopping
    time the paper analyses is the one that actually runs.

    Parameters
    ----------
    per_sample_loss_fn
        ``(params, x_single, y_single) -> scalar``. Vectorized here
        once, outside the loop.
    params
        The initial ``x^0``. Converted once, on entry, to a host
        `numpy` float64 pytree; every returned parameter set is one
        too.
    optimizer
        Algorithm 4's line is ``x^{t+1} = Pi_X(x^t - eta G(x^t))``, so
        ``updates.sgd(eta)``. ``Pi_X`` is *not* an argument: a
        caller-side projection composes at the optimizer seam per
        ADR-0014, so wrap this in
        `dimma.transforms.projection.l1_projected` to run the paper's
        own containment. **Dtype caveat:** the float64 apply side
        survives only an optimizer whose arithmetic is `numpy`'s, such
        as `updates.sgd` at a constant learning rate. A
        `updates.Schedule` returns a `jax.numpy` scalar, and a stateful
        optax transformation carries `jax.numpy` state and operates on
        it; either pulls the update — and with it every iterate this
        loop returns — back down to float32. Nothing raises, so a
        caller who needs the precision has to check.
    x, y
        The training set, ``S``. ``len(x)`` is the ``n`` every
        amplification rate and every per-step price is stated over.
    key, rng
        The noise and sampling streams. Pass one ``rng`` for the whole
        run: it drives the scale coin, the batch draws and the output
        rule's reservoir, in that order within a step.
    target_epsilon, target_delta
        The run's whole budget, and the only thing that sets its
        length. ``target_epsilon`` must be at most 1: Lemma 5.3's
        amplification is stated there, and above it the per-step price
        would be an under-estimate. ``target_delta`` should further be
        below ``1 / n ** 2``, which Lemma 5.5's stopping-time bound
        assumes; above that the privacy still holds and the bound on
        how many steps to expect does not.
    clip_norm
        The paper's ``L``, enforced by stage 4 (ADR-0012) rather than
        assumed of the loss. It reaches the step through
        ``estimator``'s claim and appears nowhere else, so the bound
        and the noise calibrated against it cannot be given different
        numbers.
    radius
        The radius of ``K``, and the caller's number per ADR-0015. The
        paper's is ``clip_norm * sqrt(s)`` for ``s``-sparse per-sample
        gradients; the library states no sparsity it cannot check.
    estimator
        The inner mean estimator's *factory*, called once as
        ``estimator(clip_norm=..., radius=..., noise_multiplier=...)``.
        Defaults to `estimators.projection_estimator`, the paper's
        Algorithm 1. Algorithm 2 drops in here when it lands.
    max_scale
        ``M``. Defaults to `dimma.core.sampling.dyadic.max_scale` of
        ``len(x)``, which is the paper's. Lowering it is a *mechanism*
        change and not a memory cap: it runs ``TGeom(M')``, whose
        debias weights, per-step price and bias bound all follow along
        — the bias becomes that of a batch of ``2 ** (M' + 1)`` rather
        than of ``n``. It also bounds per-step device memory at exactly
        ``2 ** (M' + 1) * d`` floats, which is why it is the knob to
        reach for. Set it too high and XLA raises out of memory; the
        package docstring gives the float32 ceiling that bounds how
        high is worth setting in any case.
    max_steps
        A wall-clock guard, and **not** a privacy parameter.
        ``E[T] <= 64 n / (9 ln(4 / delta))`` — order ``10 ** 7`` at
        Criteo scale — so a run left to the filter alone can be
        compute-bound long before it is budget-bound. Stopping early is
        always sound: it is strictly less access to the data, so the
        ``(target_epsilon, target_delta)`` guarantee is untouched. What
        is lost is Section 5.3.1's *lower* bound on the stopping time,
        and with it the accuracy theorem — the privacy survives, the
        utility claim does not.

    Returns
    -------
    Run
        Both output rules, the last iterate, the realized step count
        and the filter's final state.

    Raises
    ------
    ValueError
        If ``target_epsilon`` is outside ``(0, 1]``, if
        ``target_delta`` is outside ``(0, 1)``, if ``len(x)`` is below
        2, if ``max_scale`` is outside ``0..dyadic.max_scale(len(x))``,
        if ``max_steps`` is negative, or if ``estimator`` returns a
        claim this algorithm's accountant does not price.

    Notes
    -----
    **What is reported, and why it is not a metric.** ADR-0006's rule
    against training loops reporting metrics is about evaluating the
    model on the training data, which is an unaccounted access.
    `Run.steps` and `Run.spent` are neither: both are deterministic
    functions of the public coin and the budget, and no data enters
    either. They are also necessary. ``T`` is an output of this
    algorithm, so a caller who did not receive it could not account for
    the run at all.

    No epsilon is computed here. Turning `Run.spent` into a number is
    ``accounting.bias_reduced_sgd.epsilon(run.spent,
    target_delta=target_delta)``, which keeps the claim in
    `accounting/` per ADR-0003.

    Because the filter is asked about the current step *before* it
    runs, the whole transcript is inside the filter: the realized
    ``epsilon(run.spent)`` is at or below ``target_epsilon / 2`` and
    the realized delta at or below ``target_delta / 4``. The run's
    guarantee is still ``(target_epsilon, target_delta)`` by Lemma 5.3,
    with the remaining ``eps/2, 3 delta/4`` of slack unspent. See the
    package docstring's third departure.

    No optimizer state is returned. `train` accepts none, so it cannot
    consume what it would hand back, and a caller who resumed from it
    would replay this run's noise stream and its filter from the start.
    """
    n = int(x.shape[0])
    if n < 2:
        raise ValueError(
            f"len(x)={n} must be at least 2; the smallest batch on the "
            f"dyadic ladder holds 2 ** (0 + 1) = 2 examples, so a "
            f"smaller training set admits no step at all."
        )
    if not 0.0 < target_epsilon <= 1.0:
        raise ValueError(
            f"target_epsilon={target_epsilon} must lie in (0, 1]; "
            f"Lemma 5.3's amplification is stated for eps <= 1, and a "
            f"larger budget would silently make every per-step price "
            f"an under-estimate."
        )
    if not 0.0 < target_delta < 1.0:
        raise ValueError(
            f"target_delta={target_delta} must lie in (0, 1); it is a "
            f"probability."
        )
    ceiling = dyadic.max_scale(n)
    if max_scale is None:
        max_scale = ceiling
    if not 0 <= max_scale <= ceiling:
        raise ValueError(
            f"max_scale={max_scale} must be in 0..{ceiling} for "
            f"len(x)={n}; above the ceiling the largest batch on the "
            f"ladder would not fit the training set."
        )
    if max_steps is not None and max_steps < 0:
        raise ValueError(
            f"max_steps={max_steps} must be non-negative; it is a "
            f"wall-clock guard on the number of updates."
        )

    mean_estimator = estimator(
        clip_norm=clip_norm,
        radius=radius,
        noise_multiplier=accounting.inner_noise_multiplier(
            target_epsilon=target_epsilon, target_delta=target_delta
        ),
    )
    accounting.check_claim(mean_estimator.claim)

    grad_fn = gradients.per_sample_grads(per_sample_loss_fn)
    batch = jax.jit(
        partial(_step.batch_release, grad_fn, mean_estimator),
        static_argnames=("batch_size",),
    )
    single = jax.jit(partial(_step.single_release, grad_fn, mean_estimator))
    releases = _step.Releases(batch=batch, single=single)
    probabilities = dyadic.scale_probabilities(max_scale)

    params = _step._as_host_float64(params)
    opt_state = updates.init(optimizer, params)
    average_params, random_params = params, params
    spent = accounting.NOTHING_SPENT
    steps = 0

    while max_steps is None or steps < max_steps:
        # The public coin, and the price it fixes. No data yet.
        scale = dyadic.draw_scale(rng, max_scale)
        cost = accounting.step_cost(
            scale=scale, n=n, target_epsilon=target_epsilon,
            target_delta=target_delta,
        )
        if not accounting.permits(
                spent, cost, target_epsilon=target_epsilon,
                target_delta=target_delta):
            break

        draw = dyadic.subsample(rng, n, scale)
        # The iterate this step is taken *from* joins both output
        # rules, so their support is x^0 .. x^T and excludes x^{T+1}.
        visited = steps + 1
        average_params = pytree.add(
            average_params,
            pytree.scale(
                pytree.sub(params, average_params), 1.0 / visited
            ),
        )
        if rng.random() < 1.0 / visited:
            random_params = params

        key, batch_key, single_key = jax.random.split(key, 3)
        params, opt_state = _step.step(
            releases, optimizer, params, opt_state,
            x[draw.whole], y[draw.whole], x[draw.single], y[draw.single],
            batch_key, single_key,
            batch_size=1 << (scale + 1),
            scale_probability=float(probabilities[scale]),
        )
        spent = accounting.spend(spent, cost)
        steps += 1

    return Run(
        average_params=average_params,
        random_params=random_params,
        final_params=params,
        steps=steps,
        spent=spent,
    )
