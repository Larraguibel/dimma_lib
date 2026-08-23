"""The bias-reduced loop: the coin, the filter, and the two output rules.

The accounting suite pins what the filter says; here, that the loop asks
it after the coin and before the data, and that every assertion about a
run's length reads the scales the loop actually drew.
"""

from __future__ import annotations

import inspect
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dimma.accounting import bias_reduced_sgd as accounting
from dimma.algorithms.bias_reduced_sgd import estimators
from dimma.algorithms.bias_reduced_sgd import step as brs_step
from dimma.algorithms.bias_reduced_sgd import train as brs_train
from dimma.core import updates
from dimma.core.sampling import dyadic

from ...helpers import tree_allclose, tree_equal
from .conftest import D, N, S, sparse_rows, squared_error

BUDGET = dict(target_epsilon=1.0, target_delta=1e-3)
GEOMETRY = dict(clip_norm=1.0, radius=float(math.sqrt(S)))
RUN = {**BUDGET, **GEOMETRY}


def run_train(x, y, params, rng, key, optimizer=None, **overrides):
    """`train` at this suite's budget, with the arguments spelled once."""
    return brs_train.train(
        squared_error, params, optimizer or updates.sgd(0.01), x, y, key,
        rng, **{**RUN, **overrides},
    )


def replay(scales, n, **budget):
    """The filter's verdict on a recorded sequence of scales."""
    budget = {**BUDGET, **budget}
    spent = accounting.NOTHING_SPENT
    admitted = []
    for scale in scales:
        cost = accounting.step_cost(scale=int(scale), n=n, **budget)
        if not accounting.permits(spent, cost, **budget):
            break
        spent = accounting.spend(spent, cost)
        admitted.append(int(scale))
    return admitted, spent


@pytest.fixture
def drawn_scales(monkeypatch):
    """Every scale the loop drew, in order: the run's public coin.

    Includes the scale of the step the filter refused, which is drawn
    and then thrown away — that it is drawn *before* the check, and
    that nothing else is, is the ordering these tests exist for.
    """
    scales: list[int] = []
    original = dyadic.draw_scale

    def recording(rng, max_scale):
        scale = original(rng, max_scale)
        scales.append(scale)
        return scale

    monkeypatch.setattr(dyadic, "draw_scale", recording)
    return scales


@pytest.fixture
def drawn_batches(monkeypatch):
    """Every batch the loop drew. One per step that actually ran."""
    draws: list = []
    original = dyadic.subsample

    def recording(rng, n, scale):
        draw = original(rng, n, scale)
        draws.append(draw)
        return draw

    monkeypatch.setattr(dyadic, "subsample", recording)
    return draws


@pytest.fixture
def visited_iterates(monkeypatch):
    """The parameters every step was taken *from*: ``x^0 ... x^T``."""
    iterates: list = []
    original = brs_step.step

    def recording(releases, optimizer, params, *args, **kwargs):
        iterates.append(params)
        return original(releases, optimizer, params, *args, **kwargs)

    monkeypatch.setattr(brs_step, "step", recording)
    return iterates


def full_loss(params, x, y) -> float:
    """Non-private evaluation, for the tests only."""
    return float(jnp.mean(
        jax.vmap(squared_error, in_axes=(None, 0, 0))(params, x, y)
    ))


# --- the coin, the filter, and where they sit ------------------------

def test_the_realized_step_count_matches_a_replay_of_the_coin(
        sparse_problem, zero_params, key, rng, drawn_scales):
    """``T`` is an output, and it is exactly what the filter says about
    the coin the run drew.

    The loop draws one more scale than it takes steps: the last one is
    priced, refused, and never turned into a batch.
    """
    x, y, _ = sparse_problem
    run = run_train(x, y, zero_params, rng, key, max_scale=3)

    assert run.steps > 0
    assert len(drawn_scales) == run.steps + 1
    admitted, spent = replay(drawn_scales, N)
    assert len(admitted) == run.steps
    assert spent == run.spent

    refused = accounting.step_cost(
        scale=drawn_scales[-1], n=N, **BUDGET)
    assert not accounting.permits(spent, refused, **BUDGET)


def test_no_gradient_is_taken_after_the_filter_refuses(
        sparse_problem, zero_params, key, rng, drawn_scales, drawn_batches):
    """Two gradient calls per admitted step and none at all after the
    filter refuses, counted eagerly: the refused step draws its scale
    and stops before any row of ``x`` is touched."""
    x, y, _ = sparse_problem
    small = 8
    x, y = x[:small], y[:small]
    calls = 0

    def counted_loss(params, x_single, y_single):
        nonlocal calls
        calls += 1
        return squared_error(params, x_single, y_single)

    with jax.disable_jit():
        run = brs_train.train(
            counted_loss, zero_params, updates.sgd(0.01), x, y, key, rng,
            **RUN,
        )

    assert run.steps > 0
    assert calls == 2 * run.steps
    assert len(drawn_batches) == run.steps
    assert len(drawn_scales) == run.steps + 1
    admitted, spent = replay(drawn_scales, small)
    assert len(admitted) == run.steps
    assert not accounting.permits(
        spent,
        accounting.step_cost(scale=drawn_scales[-1], n=small, **BUDGET),
        **BUDGET,
    )


def test_the_step_count_depends_on_the_sampling_seed_and_not_the_noise_seed(
        sparse_problem, zero_params):
    """The two streams, separated at the place it matters most.

    The stopping time is a function of the coin, which is the sampling
    stream; the noise stream cannot move it. If it could, the number of
    steps would be a function of the released noise, and the filter's
    schedule would no longer be predictable from public randomness.
    """
    x, y, _ = sparse_problem
    counts_by_noise = {
        run_train(x, y, zero_params, np.random.default_rng(0),
                  jax.random.key(seed), max_scale=1).steps
        for seed in range(3)
    }
    assert len(counts_by_noise) == 1

    counts_by_sampling = {
        run_train(x, y, zero_params, np.random.default_rng(seed),
                  jax.random.key(0), max_scale=1).steps
        for seed in range(4)
    }
    assert len(counts_by_sampling) > 1


def test_a_run_is_reproducible_from_its_two_seeds(sparse_problem,
                                                  zero_params):
    """Two seeds in, the same run out — iterates, count and filter
    state alike."""
    x, y, _ = sparse_problem
    runs = [
        run_train(x, y, zero_params, np.random.default_rng(5),
                  jax.random.key(9), max_scale=2)
        for _ in range(2)
    ]
    assert runs[0].steps == runs[1].steps
    assert runs[0].spent == runs[1].spent
    for field in ("average_params", "random_params", "final_params"):
        assert tree_equal(getattr(runs[0], field), getattr(runs[1], field))


# --- what the loop optimizes -----------------------------------------

def quieter_than_the_budget(*, clip_norm, radius, noise_multiplier):
    """Swap the calibrated multiplier for a small one, so that a run at
    test sizes is not measuring its own noise. Only the size of the
    noise changes; the mechanism around it is the shipped one."""
    del noise_multiplier
    return estimators.projection_estimator(
        clip_norm=clip_norm, radius=radius, noise_multiplier=0.05
    )


@pytest.fixture
def larger_sparse_problem():
    """The same sparse model at ``n = 1024``, where the filter admits
    enough steps for a trajectory to be visible at all."""
    gen = np.random.default_rng(2)
    n, d, s = 1024, 32, S
    x = sparse_rows(gen, n, d, s)
    w_true = np.zeros(d, dtype=np.float32)
    w_true[gen.choice(d, size=s, replace=False)] = gen.standard_normal(s)
    y = (x @ w_true + 0.05 * gen.standard_normal(n)).astype(np.float32)
    return jnp.asarray(x), jnp.asarray(y), {"w": jnp.zeros(d)}


def test_training_reduces_the_loss(larger_sparse_problem):
    """The loop descends, through both output rules.

    Small ``n``, a small step size, and the estimator seam carrying a
    multiplier a test can afford — see `quieter_than_the_budget` for
    why the calibrated one cannot be measured here.
    """
    x, y, start = larger_sparse_problem
    run = brs_train.train(
        squared_error, start, updates.sgd(0.1), x, y, jax.random.key(0),
        np.random.default_rng(0), estimator=quieter_than_the_budget,
        max_scale=4, **RUN,
    )
    assert run.steps > 100
    before = full_loss(start, x, y)
    assert full_loss(run.average_params, x, y) < 0.8 * before
    assert full_loss(run.final_params, x, y) < 0.8 * before


def test_the_output_iterates_come_from_the_trajectory(
        sparse_problem, zero_params, key, rng, visited_iterates):
    """Both output rules read ``x^0 ... x^T`` and neither reads
    ``x^{T+1}``.

    The average is the online mean of that support and the random
    iterate is one of its members, drawn by reservoir sampling because
    ``T`` is not known when the run starts. The last iterate is
    returned separately and carries no bound: no gradient estimate was
    ever taken at it.
    """
    x, y, _ = sparse_problem
    run = run_train(x, y, zero_params, rng, key, max_scale=2)

    assert len(visited_iterates) == run.steps
    assert tree_allclose(visited_iterates[0], zero_params, atol=0.0)
    assert any(tree_equal(run.random_params, iterate)
               for iterate in visited_iterates)
    assert not tree_equal(run.random_params, run.final_params)

    stacked = np.stack([np.asarray(iterate["w"])
                        for iterate in visited_iterates])
    assert np.allclose(run.average_params["w"], stacked.mean(axis=0),
                       atol=1e-12)


# --- the two knobs ---------------------------------------------------

def test_max_steps_stops_early_and_leaves_budget(sparse_problem,
                                                 zero_params, key):
    """A wall-clock guard, and nothing else.

    It cuts the run short, which is strictly less access to the data,
    so the guarantee is untouched and the unspent budget is visible in
    `Run.spent`. What is lost is the paper's *lower* bound on the
    stopping time and with it the accuracy theorem — the privacy
    survives, the utility claim does not.
    """
    x, y, _ = sparse_problem
    full = run_train(x, y, zero_params, np.random.default_rng(0), key,
                     max_scale=2)
    capped = run_train(x, y, zero_params, np.random.default_rng(0), key,
                       max_scale=2, max_steps=full.steps // 2)

    assert capped.steps == full.steps // 2
    assert capped.spent.sum_delta < full.spent.sum_delta
    assert (accounting.epsilon(capped.spent, target_delta=BUDGET[
        "target_delta"]) < accounting.epsilon(
            full.spent, target_delta=BUDGET["target_delta"]))
    assert accounting.permits(
        capped.spent,
        accounting.step_cost(scale=0, n=N, **BUDGET), **BUDGET)


def test_a_cap_of_zero_takes_no_step_at_all(sparse_problem, zero_params,
                                            key, rng, drawn_scales):
    """The degenerate guard: no coin, no draw, no update, and the
    output rules fall back on ``x^0``."""
    x, y, _ = sparse_problem
    run = run_train(x, y, zero_params, rng, key, max_steps=0)
    assert run.steps == 0
    assert run.spent == accounting.NOTHING_SPENT
    assert drawn_scales == []
    for field in ("average_params", "random_params", "final_params"):
        assert tree_allclose(getattr(run, field), zero_params, atol=0.0)


def test_max_scale_bounds_every_draw(sparse_problem, zero_params, key, rng,
                                     drawn_scales, drawn_batches):
    """``TGeom(M')`` is a law over ``0..M'``, so no draw leaves it and
    no batch is larger than ``2 ** (M' + 1)``."""
    x, y, _ = sparse_problem
    run = run_train(x, y, zero_params, rng, key, max_scale=2)
    assert run.steps > 0
    assert max(drawn_scales) <= 2
    assert max(draw.whole.size for draw in drawn_batches) <= 8


def test_a_lower_max_scale_buys_more_steps(sparse_problem, zero_params,
                                           key):
    """A shorter ladder means cheaper steps and so more of them. The
    asserted ratio comes from the expected prices, about 9/1024 of the
    budget a step at ``M' = 1`` against 16/1024 at ``M' = 4``, so it
    reads the prices rather than one realized sequence of draws."""
    x, y, _ = sparse_problem
    short_ladder, long_ladder = [
        float(np.mean([
            run_train(x, y, zero_params, np.random.default_rng(seed), key,
                      max_scale=ceiling).steps
            for seed in range(2)
        ]))
        for ceiling in (1, 4)
    ]
    assert short_ladder > 1.4 * long_ladder


# --- dtypes ----------------------------------------------------------

def test_the_parameters_are_carried_in_float64(sparse_problem, zero_params,
                                               key, rng):
    """Every iterate the loop returns is a host `numpy` float64 pytree,
    including the input converted on entry — the apply side runs where
    the ``1 / p_N`` amplification of a near-cancellation is safe."""
    x, y, _ = sparse_problem
    run = run_train(x, y, zero_params, rng, key, max_scale=1)
    for field in ("average_params", "random_params", "final_params"):
        for leaf in jax.tree_util.tree_leaves(getattr(run, field)):
            assert isinstance(leaf, np.ndarray)
            assert leaf.dtype == np.float64


def test_an_optax_optimizer_reverts_to_float32(sparse_problem, zero_params,
                                               key, rng):
    """The documented caveat: an optimizer carrying `jax.numpy` state or
    a `jax.numpy` rate pulls the float64 apply side back to float32,
    silently. Constant-rate ``optax.sgd`` happens not to, which is
    recorded as a boundary and not as a rule to lean on."""
    x, y, _ = sparse_problem
    for optimizer in (optax.adam(0.01),
                      updates.sgd(optax.linear_schedule(0.01, 0.0, 20))):
        run = run_train(x, y, zero_params, np.random.default_rng(0), key,
                        optimizer=optimizer, max_scale=1)
        leaves = jax.tree_util.tree_leaves(run.final_params)
        assert leaves and all(leaf.dtype == jnp.float32 for leaf in leaves)

    stayed = run_train(x, y, zero_params, np.random.default_rng(0), key,
                       optimizer=optax.sgd(0.01), max_scale=1)
    assert all(leaf.dtype == np.float64
               for leaf in jax.tree_util.tree_leaves(stayed.final_params))


# --- compilation -----------------------------------------------------

def test_the_loop_traces_once_per_scale_it_sees(sparse_problem, zero_params,
                                                key, rng, drawn_scales):
    """Per-shape compilation, over a whole run.

    The batch release is traced once per *distinct* scale the coin
    produced, however many steps each one served; the single release
    has one shape for the run and is traced once, which is why it is
    bound outside the per-scale table. A run therefore compiles at most
    ``max_scale + 2`` programs however long it is.
    """
    x, y, _ = sparse_problem
    traces = 0

    def counted_loss(params, x_single, y_single):
        nonlocal traces
        traces += 1
        return squared_error(params, x_single, y_single)

    run = brs_train.train(
        counted_loss, zero_params, updates.sgd(0.01), x, y, key, rng,
        max_scale=3, **RUN,
    )
    scales_that_ran = set(drawn_scales[:run.steps])
    assert len(scales_that_ran) > 1
    assert traces == len(scales_that_ran) + 1


# --- what the loop refuses -------------------------------------------

@pytest.mark.parametrize("target_epsilon", [1.5, 8.0, 0.0, -1.0])
def test_an_epsilon_above_one_is_rejected(sparse_problem, zero_params, key,
                                          rng, target_epsilon):
    """The budget must land in ``(0, 1]``. Above one, Lemma 5.3's
    amplification no longer holds and every per-step price would be an
    under-estimate; at or below zero there is nothing to spend. The run
    is refused rather than filtered against a bound that is not one."""
    x, y, _ = sparse_problem
    with pytest.raises(ValueError, match="target_epsilon="):
        run_train(x, y, zero_params, rng, key,
                  target_epsilon=target_epsilon)


@pytest.mark.parametrize("target_delta", [0.0, 1.0, -1e-6])
def test_a_delta_that_is_not_a_probability_is_rejected(
        sparse_problem, zero_params, key, rng, target_delta):
    x, y, _ = sparse_problem
    with pytest.raises(ValueError, match="target_delta="):
        run_train(x, y, zero_params, rng, key, target_delta=target_delta)


def test_a_dataset_below_two_is_rejected(zero_params, key, rng):
    """The ladder's bottom rung holds two examples, so a single-example
    training set admits no step at all."""
    x = jnp.zeros((1, D))
    y = jnp.zeros((1,))
    with pytest.raises(ValueError, match="len\\(x\\)="):
        run_train(x, y, zero_params, rng, key)


@pytest.mark.parametrize("max_scale", [6, 12, -1])
def test_a_max_scale_above_the_ceiling_is_rejected(sparse_problem,
                                                   zero_params, key, rng,
                                                   max_scale):
    """``dyadic.max_scale(64) == 5``: above it the largest batch on the
    ladder would not fit the training set, and below zero the ladder has
    no rung to draw at."""
    x, y, _ = sparse_problem
    with pytest.raises(ValueError, match="max_scale="):
        run_train(x, y, zero_params, rng, key, max_scale=max_scale)


def test_an_estimator_this_accountant_cannot_price_is_rejected(
        sparse_problem, zero_params, key, rng):
    """The seam's far end. `train` hands the estimator's claim to the
    accountant *before* it takes a step, so no run is ever filtered
    under assumptions belonging to another mechanism."""
    x, y, _ = sparse_problem

    def foreign(*, clip_norm, radius, noise_multiplier):
        del clip_norm, radius, noise_multiplier
        return estimators.MeanEstimator(
            name="stub", claim=("not", "a", "claim"),
            estimate=lambda mean, key, batch_size: mean,
        )

    with pytest.raises(ValueError, match="Theorem 3.4"):
        run_train(x, y, zero_params, rng, key, estimator=foreign)


# --- the contract ----------------------------------------------------

def test_the_loop_returns_no_metric():
    """ADR-0006's rule, and the two things that are not exceptions to
    it.

    `Run` carries three parameter sets, the realized step count and the
    filter's state — no loss, no evaluation, no callback to hide one
    in. ``steps`` and ``spent`` are deterministic functions of the
    public coin and the budget, and ``steps`` is *this algorithm's
    output*: a caller who did not receive it could not account for the
    run at all. Converting `Run.spent` to an epsilon stays in
    `accounting`, per ADR-0003, so no epsilon is computed here either.
    """
    assert brs_train.Run._fields == (
        "average_params", "random_params", "final_params", "steps", "spent",
    )
    parameters = inspect.signature(brs_train.train).parameters
    assert set(parameters) == {
        "per_sample_loss_fn", "params", "optimizer", "x", "y", "key", "rng",
        "target_epsilon", "target_delta", "clip_norm", "radius",
        "estimator", "max_scale", "max_steps",
    }
    assert "steps" not in parameters
    assert not any("epsilon" in name for name in brs_train.Run._fields)
