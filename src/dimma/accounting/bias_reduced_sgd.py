"""Bias-reduced SGD accounting: a privacy filter, in closed form.

Accounts `dimma.algorithms.bias_reduced_sgd`. This module travels with
that algorithm and is not general-purpose, per the membership rule in
`dimma.accounting`. What is bespoke is the *shape* of the accountant:
the schedule of per-step costs is chosen inside the run, so nothing is
fixed in advance to compose over, and Theorem A.4's
``(eps, delta)``-filter (Whitehouse, Ramdas, Rogers, Wu 2023, as the
paper cites it) is what covers that. ADR-0018 records the decision.

Assumed mechanism
-----------------
Lemma 5.3's, verbatim in its arithmetic. At step ``t`` the loop draws a
scale ``N_t`` from the public coin and then, conditionally on the
parameters ``x_t``, the scale and the batch ``B_t``:

- three inner Gaussian mean releases — over ``B_t``, over ``O_t`` and
  over ``E_t`` — each at ``(eps/32, delta/16)``, basic-composed to
  ``(3 eps/32, 3 delta/16)``;
- that composition amplified *once*, jointly, by the joint subsampling
  of ``B_t``, at rate ``2 ** (N_t + 1) / n``;
- one further Gaussian mean release over the single record ``I_t``, at
  ``(eps/32, delta/16)``, amplified at rate ``1 / n``.

which is::

    eps_t   = (3 * 2 ** (N_t + 1) + 1) * eps / (16 * n)
    delta_t = (3 * 2 ** (N_t + 1) + 1) * delta / (16 * n)

Both are functions of the coin alone. No data enters a cost, which is
what lets `permits` be consulted before the batch is drawn, and what
makes carrying a `Spent` out of a training loop a report of the coin
rather than a metric.

What this cannot check
----------------------
The numbers above are a guarantee only under statements this module is
in no position to verify — the ADR-0009 pattern, the assumption stated
where the number is made:

- **the amplification is the paper's.** Lemma 5.3 amplifies a
  fixed-size draw *without replacement* at rate ``2 ** (N + 1) / n``.
  That is not the subsampled-Gaussian statement `accounting.sampling`
  wraps, which is Poisson; substituting one for the other is an
  unproven analogy and `dimma.core.sampling.dyadic` deliberately makes
  no such claim. Proving the fixed-size statement is a research
  question, not a port;
- **the sensitivity is Theorem 3.3's.** An inner release perturbs an
  empirical mean whose ``l_2`` sensitivity is ``2 L / k`` at batch size
  ``k``, with ``L`` the clipping norm. The training loop enforces
  ``L`` by clipping, so that half is code; that the mean's sensitivity
  is ``2 L / k`` under add-or-remove-one adjacency is the paper's;
- **the budget is at most 1.** Lemma 5.3's amplification is stated for
  ``eps <= 1``. Above it the per-step cost here is an *under*-estimate,
  so the loop refuses such a budget rather than reporting from it;
- **the inner mechanism is Gaussian.** `check_claim` enforces exactly
  this much, by type, and refuses anything else.

No ``method`` argument
----------------------
No function here takes one, and there is no Rényi path:
`dp_accounting` exposes nothing filter-shaped, so the closed form
below is the whole accountant and an argument would advertise a choice
that does not exist. A Rényi filter would be tighter and is ticketed
(#35). ADR-0018 records the carve-out from ADR-0011.

Reference: B. Ghazi, C. Guzman, P. Kamath, R. Kumar, P. Manurangsi,
"Differentially Private Optimization with Sparse Gradients", NeurIPS
2024 — Algorithm 4, Lemma 5.3 and Theorem A.4.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

__all__ = [
    "GaussianMeanClaim",
    "Spent",
    "NOTHING_SPENT",
    "check_claim",
    "inner_noise_multiplier",
    "step_cost",
    "spend",
    "epsilon",
    "permits",
]


class Spent(NamedTuple):
    """The filter's state: everything Theorem A.4 needs, and nothing else.

    Two running sums and a count: `epsilon` needs the per-step costs
    only through their sum of squares, and the deltas add.

    Computed from the public coin alone — no data enters a cost — so
    carrying one out of a training loop reports the coin and not the
    training set, which is why
    `dimma.algorithms.bias_reduced_sgd.train` may return it without
    breaking ADR-0006's rule against loops reporting metrics.
    """

    steps: int
    """Releases the filter has been charged for, which for this
    algorithm is also the number of updates taken."""

    sum_squared_epsilon: float
    """``sum_s eps_s ** 2`` over the steps charged so far."""

    sum_delta: float
    """``sum_s delta_s`` over the steps charged so far."""


NOTHING_SPENT = Spent(0, 0.0, 0.0)
"""A run before its first step. The identity of `spend`."""


class GaussianMeanClaim(NamedTuple):
    """The privacy claim an inner mean release carries.

    A Gaussian mechanism over a batch mean of ``l_2`` sensitivity
    ``2 * clip_norm / batch_size`` — Theorem 3.3's ``Delta_2 = 2L/n``
    — perturbed at ``noise_multiplier`` times it, then post-processed.

    Its *type* is what the accountant checks: Lemma 5.3 composes four
    Gaussian releases and nothing else, so a different mechanism —
    Algorithm 2's second branch folds a random-matrix failure event
    into ``delta`` — must be refused rather than accounted by analogy.
    The type is defined here beside `check_claim` and the closed form
    that prices it, per ADR-0003, so that `dimma.accounting` prices a
    run without importing algorithm code;
    `dimma.algorithms.bias_reduced_sgd.estimators` re-exports it.

    The batch size is deliberately absent. It changes within a step and
    is a property of the slot, not of the estimator; what is fixed for
    the run, and what an accountant needs, is the dimensionless
    multiplier.
    """

    clip_norm: float
    """``L``. Enforced by stage 4 rather than assumed of the loss —
    ADR-0012's pattern, not ADR-0009's."""

    noise_multiplier: float
    """The standard deviation actually added, divided by the
    sensitivity it is calibrated against."""


def check_claim(claim: Any) -> None:
    """Refuse an inner mean estimator this analysis does not cover.

    The one thing the accountant checks about the code that ran; the
    error message below says what it rules out, and
    `GaussianMeanClaim` why.

    Parameters
    ----------
    claim
        `dimma.algorithms.bias_reduced_sgd.estimators.MeanEstimator`'s
        ``claim`` field, passed by whoever ran the estimator.

    Raises
    ------
    ValueError
        If ``claim`` is not a `GaussianMeanClaim`.
    """
    if not isinstance(claim, GaussianMeanClaim):
        raise ValueError(
            f"claim is a {type(claim).__name__}, and this accountant "
            f"prices only a GaussianMeanClaim. Lemma 5.3 composes four "
            f"Gaussian mean releases, each at (eps/32, delta/16) over a "
            f"sensitivity of 2 * clip_norm / batch_size. Algorithm 2's "
            f"Theorem 3.4 is the other mechanism the paper offers, and "
            f"it is not this one: it holds only in a regime condition "
            f"on the dimension and the batch size — which fails outright "
            f"at the batch of one that the G_0 slot needs — and it folds "
            f"a random-matrix failure event into delta rather than "
            f"perturbing a mean. Account it where it is analysed, not "
            f"here."
        )


def inner_noise_multiplier(*, target_epsilon: float,
                           target_delta: float) -> float:
    """The multiplier every inner mean release runs at::

        32 * sqrt(2 * ln(20 / delta)) / epsilon

    The budget's whole journey inward. Algorithm 4 hands Algorithm 3
    ``(eps/8, delta/4)``; Algorithm 3 splits that four ways to
    ``(eps/32, delta/16)`` a slot; Algorithm 1 answers a slot with
    ``sigma ** 2 = 8 L ** 2 ln(1.25 / delta_in) / (k eps_in) ** 2``
    over a sensitivity of ``2 L / k``. The ratio cancels ``L`` and
    ``k``, so one dimensionless number serves every slot of a run::

        sigma / (2 L / k) = sqrt(2 ln(1.25 / (delta / 16))) / (eps / 32)

    and ``1.25 * 16 = 20``.

    Parameters
    ----------
    target_epsilon, target_delta
        The run's whole budget, not a slot's share. The division is
        this function's.

    Returns
    -------
    float
        Pass to
        `dimma.algorithms.bias_reduced_sgd.estimators.projection_estimator`
        as ``noise_multiplier``.

    Raises
    ------
    ValueError
        If the budget is not one Lemma 5.3 covers.
    """
    _check_budget(target_epsilon, target_delta)
    return 32.0 * math.sqrt(2.0 * math.log(20.0 / target_delta)) / (
        target_epsilon
    )


def step_cost(*, scale: int, n: int, target_epsilon: float,
              target_delta: float) -> tuple[float, float]:
    """Lemma 5.3's price for one step at a drawn scale::

        (3 * 2 ** (scale + 1) + 1) * (epsilon, delta) / (16 * n)

    The module docstring's assumed mechanism, priced.

    Parameters
    ----------
    scale
        ``N_t``, from `dimma.core.sampling.dyadic.draw_scale`: the
        public coin, and the only thing the price depends on.
    n
        Training set size.
    target_epsilon, target_delta
        The run's whole budget.

    Returns
    -------
    tuple
        ``(epsilon, delta)`` for this one step.

    Raises
    ------
    ValueError
        On a negative scale, a dataset below two, or a budget outside
        what Lemma 5.3 covers.
    """
    _check_budget(target_epsilon, target_delta)
    if n < 2:
        raise ValueError(
            f"n={n} must be at least 2; the smallest batch on the "
            f"dyadic ladder holds two examples."
        )
    if scale < 0:
        raise ValueError(
            f"scale={scale} must be non-negative; it is the drawn "
            f"``N``, and the batch it sizes holds 2 ** (scale + 1) "
            f"examples."
        )
    releases = 3.0 * float(1 << (scale + 1)) + 1.0
    step_epsilon = releases * target_epsilon / (16.0 * n)
    step_delta = releases * target_delta / (16.0 * n)
    return (step_epsilon, step_delta)


def spend(spent: Spent, cost: tuple[float, float]) -> Spent:
    """Charge one step's cost against the filter's state.

    Parameters
    ----------
    spent
        The state before the step; `NOTHING_SPENT` at the start.
    cost
        ``(epsilon, delta)`` as `step_cost` returns it.

    Returns
    -------
    Spent
        A new state. Nothing is mutated: a filter's state is the
        transcript so far, and a caller holding an earlier one holds a
        true statement about an earlier prefix.
    """
    step_epsilon, step_delta = cost
    steps = spent.steps + 1
    sum_squared_epsilon = spent.sum_squared_epsilon + step_epsilon ** 2
    sum_delta = spent.sum_delta + step_delta
    return Spent(
        steps=steps,
        sum_squared_epsilon=sum_squared_epsilon,
        sum_delta=sum_delta,
    )


def epsilon(spent: Spent, *, target_delta: float) -> float:
    """Theorem A.4's ``eps[0:t]``, at the paper's ``delta' = delta/4``::

        sqrt(2 ln(4 / delta) * sum eps_s ** 2) + 0.5 * sum eps_s ** 2

    Advanced composition in the form that survives an adaptively chosen
    schedule. It is the ``eps`` half of the filter; the ``delta`` half
    is the plain sum `Spent.sum_delta` carries, tested against
    ``delta / 4``.

    Parameters
    ----------
    spent
        The state after the steps to be priced. Only its
        `Spent.sum_squared_epsilon` is read.
    target_delta
        The run's whole ``delta``, from which the theorem's ``delta'``
        is this module's quarter share. The same number the run was
        given, not one chosen at reporting time: the schedule was
        filtered against this one.

    Returns
    -------
    float
        The filter's epsilon for the transcript so far. Zero before the
        first step.

    Raises
    ------
    ValueError
        If ``target_delta`` is not in ``(0, 1)``, or if the state
        carries a negative sum of squares.
    """
    if not 0.0 < target_delta < 1.0:
        raise ValueError(
            f"target_delta={target_delta} must lie in (0, 1); it is a "
            f"probability, and the filter reads delta / 4 off it."
        )
    if spent.sum_squared_epsilon < 0.0:
        raise ValueError(
            f"sum_squared_epsilon={spent.sum_squared_epsilon} must be "
            f"non-negative; it is a sum of squares."
        )
    total = spent.sum_squared_epsilon
    return math.sqrt(2.0 * math.log(4.0 / target_delta) * total) + 0.5 * total


def permits(spent: Spent, cost: tuple[float, float], *,
            target_epsilon: float, target_delta: float) -> bool:
    """Whether the filter admits one more step at this cost.

    Theorem A.4's stopping time, at Lemma 5.3's half budget and with
    **this step's cost included**::

        eps[0:t + 1] <= target_epsilon / 2
        sum_{s <= t} delta_s <= target_delta / 4

    Both arms, and either one stops the run; ADR-0018 records the
    shift from Algorithm 4's printed ``while``.

    Parameters
    ----------
    spent
        The filter's state before the step.
    cost
        The step's ``(epsilon, delta)``, from `step_cost` and so from
        the coin.
    target_epsilon, target_delta
        The run's whole budget; the halves and quarters are the
        paper's, and are taken here so that no caller has to.

    Returns
    -------
    bool
        ``True`` if the step may run.

    Raises
    ------
    ValueError
        On a budget outside what Lemma 5.3 covers, or a negative cost.
    """
    _check_budget(target_epsilon, target_delta)
    step_epsilon, step_delta = cost
    if step_epsilon < 0.0 or step_delta < 0.0:
        raise ValueError(
            f"cost={cost} must be non-negative in both coordinates; it "
            f"is one step's (epsilon, delta)."
        )
    after = spend(spent, cost)
    return (epsilon(after, target_delta=target_delta)
            <= target_epsilon / 2.0
            and after.sum_delta <= target_delta / 4.0)


def _check_budget(target_epsilon: float, target_delta: float) -> None:
    """The budget Lemma 5.3 is stated for, and nothing wider."""
    if not 0.0 < target_epsilon <= 1.0:
        raise ValueError(
            f"target_epsilon={target_epsilon} must lie in (0, 1]; "
            f"Lemma 5.3's amplification is stated for eps <= 1, and "
            f"above it the per-step cost priced here is an "
            f"under-estimate rather than a bound."
        )
    if not 0.0 < target_delta < 1.0:
        raise ValueError(
            f"target_delta={target_delta} must lie in (0, 1); it is a "
            f"probability, and the paper's stopping-time bound further "
            f"assumes it below 1 / n ** 2."
        )
