"""The bias-reduced filter, against Lemma 5.3 and Theorem A.4.

Nothing here runs a loop. A filter's whole job is to answer one
question — may this step run? — from a cost that was computed before
the step existed, so it is testable on cost sequences built by hand,
and that is how the paper's stopping rule is pinned: on constructed
schedules, where what the filter should say is known independently of
what any training run did.

`tests/algorithms/bias_reduced_sgd/test_train.py` pins the other half,
that the loop asks the question in the right place.
"""

from __future__ import annotations

import inspect
import math
from typing import NamedTuple

import numpy as np
import pytest

from dimma.accounting import bias_reduced_sgd as accounting
from dimma.algorithms.bias_reduced_sgd import estimators
from dimma.core.sampling import dyadic

EPSILON, DELTA = 1.0, 1e-3
BUDGET = dict(target_epsilon=EPSILON, target_delta=DELTA)


def admitted_steps(scales, *, n, printed=False, **budget) -> int:
    """How many of ``scales`` the filter lets through, greedily.

    ``printed=True`` runs Algorithm 4's *printed* ``while`` instead,
    which sums the costs of steps ``s <= t - 1`` and so decides step
    ``t`` without pricing it.
    """
    budget = {**BUDGET, **budget}
    spent = accounting.NOTHING_SPENT
    for scale in scales:
        cost = accounting.step_cost(scale=int(scale), n=n, **budget)
        if printed:
            admitted = (
                accounting.epsilon(
                    spent, target_delta=budget["target_delta"])
                <= budget["target_epsilon"] / 2.0
                and spent.sum_delta <= budget["target_delta"] / 4.0
            )
        else:
            admitted = accounting.permits(spent, cost, **budget)
        if not admitted:
            return spent.steps
        spent = accounting.spend(spent, cost)
    raise AssertionError(
        "the sequence ran out before the filter stopped; lengthen it"
    )


# --- the per-step price ----------------------------------------------

@pytest.mark.parametrize("scale, n", [
    (0, 2), (0, 64), (1, 64), (5, 64), (0, 1024), (9, 1024),
    (3, 1_000_000), (24, 45_000_000),
])
def test_the_step_cost_is_the_papers_formula(scale, n):
    """Lemma 5.3, coefficient by coefficient::

        (3 * 2 ** (N + 1) + 1) * (eps, delta) / (16 * n)

    Three inner releases over one draw of ``B``, amplified together at
    ``2 ** (N + 1) / n``, plus ``G_0`` amplified at ``1 / n``.
    """
    factor = (3 * 2 ** (scale + 1) + 1) / (16 * n)
    step_epsilon, step_delta = accounting.step_cost(
        scale=scale, n=n, **BUDGET)
    assert math.isclose(step_epsilon, factor * EPSILON, rel_tol=1e-12)
    assert math.isclose(step_delta, factor * DELTA, rel_tol=1e-12)


def test_the_price_depends_on_the_coin_and_nothing_else():
    """Two arguments beyond the budget: the drawn scale and ``n``.

    This is what lets `permits` be consulted before the batch is drawn.
    A price that needed the batch, or the parameters, or a gradient,
    would make the budget check depend on what it is protecting.
    """
    parameters = inspect.signature(accounting.step_cost).parameters
    assert set(parameters) == {"scale", "n", "target_epsilon",
                               "target_delta"}


def test_no_step_costs_more_than_a_quarter_of_the_budget():
    """The worst rung of the ladder is still affordable.

    ``2 ** (M + 1) <= n``, so the largest price is at most
    ``(3 n + 1) / (16 n) < 1 / 4`` of the budget in each coordinate.
    A single step can therefore never overshoot on its own, which is
    what makes the filter a stopping rule rather than a refusal.
    """
    for n in [2, 3, 4, 7, 8, 64, 1000, 1024, 45_000_000]:
        for scale in range(dyadic.max_scale(n) + 1):
            step_epsilon, step_delta = accounting.step_cost(
                scale=scale, n=n, **BUDGET)
            assert step_epsilon <= EPSILON / 4.0
            assert step_delta <= DELTA / 4.0


# --- Theorem A.4's epsilon -------------------------------------------

def test_epsilon_is_theorem_a4s_closed_form():
    """``sqrt(2 ln(4/delta) * sum eps_s ** 2) + 0.5 * sum eps_s ** 2``,
    the theorem at the paper's ``delta' = delta / 4``."""
    spent = accounting.Spent(
        steps=3, sum_squared_epsilon=0.04, sum_delta=1e-5)
    assert math.isclose(
        accounting.epsilon(spent, target_delta=DELTA),
        math.sqrt(2.0 * math.log(4.0 / DELTA) * 0.04) + 0.5 * 0.04,
        rel_tol=1e-12,
    )


def test_a_run_that_has_not_started_has_spent_nothing():
    """`NOTHING_SPENT` is `spend`'s identity element and epsilon's
    zero."""
    assert accounting.NOTHING_SPENT == accounting.Spent(0, 0.0, 0.0)
    assert accounting.epsilon(
        accounting.NOTHING_SPENT, target_delta=DELTA) == 0.0


def test_spending_accumulates_squares_and_deltas():
    """The two running sums Theorem A.4 needs, and no transcript."""
    spent = accounting.spend(accounting.NOTHING_SPENT, (0.2, 1e-5))
    spent = accounting.spend(spent, (0.1, 2e-5))
    assert spent.steps == 2
    assert math.isclose(spent.sum_squared_epsilon, 0.05, rel_tol=1e-12)
    assert math.isclose(spent.sum_delta, 3e-5, rel_tol=1e-12)


# --- the filter ------------------------------------------------------

def test_the_filter_never_admits_a_step_that_would_exceed_the_budget():
    """The invariant, over random schedules: every admitted prefix is
    inside ``(eps/2, delta/4)``, and the step that was refused would
    have left it."""
    gen = np.random.default_rng(0)
    n = 4096
    for _ in range(20):
        scales = gen.integers(0, dyadic.max_scale(n) + 1, size=500)
        spent = accounting.NOTHING_SPENT
        for scale in scales:
            cost = accounting.step_cost(scale=int(scale), n=n, **BUDGET)
            if not accounting.permits(spent, cost, **BUDGET):
                would_be = accounting.spend(spent, cost)
                assert (accounting.epsilon(would_be, target_delta=DELTA)
                        > EPSILON / 2.0
                        or would_be.sum_delta > DELTA / 4.0)
                break
            spent = accounting.spend(spent, cost)
            assert (accounting.epsilon(spent, target_delta=DELTA)
                    <= EPSILON / 2.0)
            assert spent.sum_delta <= DELTA / 4.0
        else:
            raise AssertionError("the filter never stopped")


def test_the_filter_admits_fewer_steps_at_larger_scales():
    """A step's price is ``3 * 2 ** (N + 1) + 1``, so a run that keeps
    drawing the top of the ladder buys far fewer steps than one that
    keeps drawing the bottom. This is the whole reason `max_scale` is a
    knob worth having."""
    n = 4096
    counts = [admitted_steps([scale] * 20_000, n=n)
              for scale in range(dyadic.max_scale(n) + 1)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > 100 * counts[-1]


def test_the_filter_admits_more_steps_on_a_larger_dataset():
    """Every price carries ``1 / n``: the same schedule on twice the
    data costs half as much, which is the amplification doing its
    work."""
    counts = [admitted_steps([2] * 200_000, n=n)
              for n in [1024, 2048, 4096, 8192]]
    assert counts == sorted(counts)
    for smaller, larger in zip(counts, counts[1:]):
        assert math.isclose(larger, 2 * smaller, rel_tol=0.02)


def test_a_bigger_budget_buys_quieter_steps_and_not_more_of_them():
    """A property of this filter that is easy to expect the wrong way
    round, so it is pinned rather than left to be discovered.

    Both arms of the price are *proportional* to the budget —
    ``eps_t = c eps``, ``delta_t = c delta`` — so doubling the budget
    doubles the threshold and the cost together and the delta arm stops
    the run after the same number of steps. Under the epsilon arm a
    larger budget is very slightly *worse*, since Theorem A.4's linear
    term ``0.5 sum eps_s ** 2`` grows quadratically in the budget while
    the threshold grows linearly. What a bigger budget buys is a
    smaller `inner_noise_multiplier`: quieter steps, not more of them.
    """
    counts = [admitted_steps([2] * 100_000, n=4096, target_epsilon=eps)
              for eps in [0.1, 0.25, 0.5, 1.0]]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == counts[-1]

    multipliers = [
        accounting.inner_noise_multiplier(
            target_epsilon=eps, target_delta=DELTA)
        for eps in [0.1, 0.25, 0.5, 1.0]
    ]
    assert multipliers == sorted(multipliers, reverse=True)


def test_the_delta_arm_and_the_epsilon_arm_both_stop_the_run():
    """Two arms, and either one alone ends the run.

    Built as costs rather than drawn as scales, so each arm can be
    exercised with the other switched off: a schedule that spends only
    delta, and one that spends only epsilon.
    """
    delta_only = (0.0, DELTA / 40.0)
    spent = accounting.NOTHING_SPENT
    while accounting.permits(spent, delta_only, **BUDGET):
        spent = accounting.spend(spent, delta_only)
    assert spent.steps == 9
    assert accounting.epsilon(spent, target_delta=DELTA) == 0.0
    assert spent.sum_delta <= DELTA / 4.0
    assert accounting.spend(spent, delta_only).sum_delta > DELTA / 4.0

    epsilon_only = (0.02, 0.0)
    spent = accounting.NOTHING_SPENT
    while accounting.permits(spent, epsilon_only, **BUDGET):
        spent = accounting.spend(spent, epsilon_only)
    assert spent.sum_delta == 0.0
    assert accounting.epsilon(spent, target_delta=DELTA) <= EPSILON / 2.0
    assert (accounting.epsilon(
        accounting.spend(spent, epsilon_only), target_delta=DELTA)
        > EPSILON / 2.0)


def test_the_filter_stops_no_later_than_algorithm_4s_printed_condition():
    """The third departure, as a number.

    Algorithm 4's printed ``while`` tests the costs of steps
    ``s <= t - 1``, so it always takes one step whose price it never
    looked at. Theorem A.4's own stopping time prices step ``t`` too,
    which is what `permits` does. Both sums are monotone, so ours stops
    exactly one step earlier — never later, which is what makes the
    departure conservative.
    """
    gen = np.random.default_rng(3)
    n = 4096
    for _ in range(10):
        scales = list(gen.integers(0, dyadic.max_scale(n) + 1, size=500))
        ours = admitted_steps(scales, n=n)
        theirs = admitted_steps(scales, n=n, printed=True)
        assert ours <= theirs <= ours + 1


# --- the inner calibration -------------------------------------------

@pytest.mark.parametrize("target_epsilon", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("target_delta", [1e-6, 1e-3])
@pytest.mark.parametrize("clip_norm, batch_size", [
    (1.0, 1), (2.5, 8), (0.1, 1024),
])
def test_the_inner_noise_multiplier_is_the_gaussian_calibration(
        target_epsilon, target_delta, clip_norm, batch_size):
    """Reconstruct the standard deviation the multiplier implies and
    compare it against Algorithm 1's own::

        sigma = sqrt(8) L sqrt(ln(1.25 / delta_in)) / (k eps_in)

    at the slot's share ``(eps/32, delta/16)``. Swept over ``L`` and
    ``k`` because both cancel: one dimensionless number has to serve
    every slot of every step, including the batch of one.
    """
    multiplier = accounting.inner_noise_multiplier(
        target_epsilon=target_epsilon, target_delta=target_delta)
    sigma = multiplier * 2.0 * clip_norm / batch_size
    want = (
        math.sqrt(8.0) * clip_norm
        * math.sqrt(math.log(1.25 / (target_delta / 16.0)))
        / (batch_size * (target_epsilon / 32.0))
    )
    assert math.isclose(sigma, want, rel_tol=1e-12)


# --- the claim ------------------------------------------------------

class ForeignClaim(NamedTuple):
    """A stub standing in for a mechanism this analysis does not
    cover — Algorithm 2's, whose failure event lands in ``delta``."""

    failure_probability: float


def test_a_non_gaussian_claim_is_refused():
    """The seam is only as safe as the refusal at its far end. Lemma
    5.3 composes Gaussian mean releases, so anything else has to be
    turned away rather than priced by resemblance, and the message says
    which theorem the other mechanism belongs to."""
    with pytest.raises(ValueError, match="Theorem 3.4"):
        accounting.check_claim(ForeignClaim(1e-6))
    with pytest.raises(ValueError, match="GaussianMeanClaim"):
        accounting.check_claim((1.0, 3.0))


def test_the_gaussian_claim_is_the_one_this_module_prices():
    assert accounting.check_claim(
        estimators.GaussianMeanClaim(clip_norm=1.0, noise_multiplier=3.0)
    ) is None


# --- the carve-out ---------------------------------------------------

def test_there_is_no_method_argument():
    """ADR-0018's carve-out from ADR-0011, pinned where it would drift.

    Every other function in `accounting` takes ``method`` and defaults
    it to ``"rdp"``. None here does, and none can: a filter bounds an
    adaptively chosen composition and `dp_accounting` exposes nothing
    filter-shaped, so there is no second path for an argument to select
    between. Adding one would advertise a choice that does not exist.
    """
    public = [getattr(accounting, name) for name in accounting.__all__]
    functions = [obj for obj in public if inspect.isfunction(obj)]
    assert len(functions) == 6
    for function in functions:
        assert "method" not in inspect.signature(function).parameters
    assert not hasattr(accounting, "Method")


# --- the budget the analysis covers ----------------------------------

@pytest.mark.parametrize("target_epsilon", [0.0, -1.0, 1.5, 8.0])
def test_a_budget_above_one_is_refused(target_epsilon):
    """Lemma 5.3's amplification is stated for ``eps <= 1``; above it
    every price here is an under-estimate, so the module refuses rather
    than reporting a number that is not a bound."""
    for call in (
        lambda: accounting.inner_noise_multiplier(
            target_epsilon=target_epsilon, target_delta=DELTA),
        lambda: accounting.step_cost(
            scale=0, n=64, target_epsilon=target_epsilon,
            target_delta=DELTA),
        lambda: accounting.permits(
            accounting.NOTHING_SPENT, (0.0, 0.0),
            target_epsilon=target_epsilon, target_delta=DELTA),
    ):
        with pytest.raises(ValueError, match="target_epsilon="):
            call()


@pytest.mark.parametrize("target_delta", [0.0, -1e-6, 1.0, 2.0])
def test_a_delta_outside_zero_to_one_is_refused(target_delta):
    with pytest.raises(ValueError, match="target_delta="):
        accounting.step_cost(
            scale=0, n=64, target_epsilon=EPSILON, target_delta=target_delta)
    with pytest.raises(ValueError, match="target_delta="):
        accounting.epsilon(
            accounting.NOTHING_SPENT, target_delta=target_delta)


def test_a_dataset_below_two_has_no_step_to_price():
    with pytest.raises(ValueError, match="n="):
        accounting.step_cost(scale=0, n=1, **BUDGET)
