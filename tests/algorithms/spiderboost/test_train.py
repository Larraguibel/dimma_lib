"""The SpiderBoost loop: stage 1, the two branches, and the output rule."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dimma.algorithms.spiderboost import step as spider_step
from dimma.algorithms.spiderboost import train as spider_train
from dimma.core import gradients, updates
from dimma.core.sampling import poisson

from .conftest import squared_error

QUIET = dict(anchor_expected_batch_size=100, variation_expected_batch_size=60,
             anchor_noise_scale=0.05, variation_noise_rate=0.5,
             variation_noise_cap=0.2)


def full_loss(params, x, y):
    """Non-private evaluation, for the tests only."""
    return float(jnp.mean(
        jax.vmap(squared_error, in_axes=(None, 0, 0))(params, x, y)
    ))


@pytest.fixture
def drawn_rates(monkeypatch):
    """Every sampling rate the loop drew at, in the order it drew them.

    The two branches sample at different rates, so the loop's
    consumption of the sampling stream *is* its branch sequence — which
    is the sequence an accountant composes over.
    """
    rates: list[float] = []
    original = poisson.subsample

    def recording(rng, n, p, b_max):
        rates.append(p)
        return original(rng, n, p, b_max)

    monkeypatch.setattr(poisson, "subsample", recording)
    return rates


def test_training_reduces_the_loss(problem, zero_params, key, rng):
    x, y, _ = problem
    output, _ = spider_train.train(
        squared_error, zero_params, updates.sgd(0.3), x, y, key, rng,
        steps=60, anchor_interval=5, **QUIET,
    )
    assert full_loss(output, x, y) < full_loss(zero_params, x, y)


def test_training_moves_toward_the_true_parameters(problem, zero_params, key,
                                                   rng):
    """Not a strawman: at a modest scale it recovers the signal."""
    x, y, w_true = problem
    output, _ = spider_train.train(
        squared_error, zero_params, updates.sgd(0.3), x, y, key, rng,
        steps=60, anchor_interval=5, **QUIET,
    )
    before = jnp.linalg.norm(w_true - zero_params["w"])
    assert jnp.linalg.norm(w_true - output["w"]) < 0.5 * before


def test_more_noise_hurts(problem, zero_params, key):
    """The privacy-utility tradeoff is visible in the loop's output."""
    x, y, _ = problem
    args = dict(steps=60, anchor_interval=5, anchor_expected_batch_size=100,
                variation_expected_batch_size=60)
    quiet, _ = spider_train.train(
        squared_error, zero_params, updates.sgd(0.3), x, y, key,
        np.random.default_rng(0), anchor_noise_scale=0.05,
        variation_noise_rate=0.5, variation_noise_cap=0.2, **args)
    loud, _ = spider_train.train(
        squared_error, zero_params, updates.sgd(0.3), x, y, key,
        np.random.default_rng(0), anchor_noise_scale=20.0,
        variation_noise_rate=50.0, variation_noise_cap=20.0, **args)
    assert full_loss(quiet, x, y) < full_loss(loud, x, y)


def test_a_run_is_reproducible_from_its_two_seeds(problem, zero_params):
    """One key and one generator determine the run, both returns included."""
    x, y, _ = problem

    def run():
        return spider_train.train(
            squared_error, zero_params, updates.sgd(0.3), x, y,
            jax.random.key(3), np.random.default_rng(3),
            steps=15, anchor_interval=4, **QUIET)

    (output_a, final_a), (output_b, final_b) = run(), run()
    assert jnp.array_equal(output_a["w"], output_b["w"])
    assert jnp.array_equal(final_a["w"], final_b["w"])


def test_the_noise_stream_changes_the_run(problem, zero_params):
    x, y, _ = problem
    args = dict(steps=15, anchor_interval=4, **QUIET)
    a, _ = spider_train.train(squared_error, zero_params, updates.sgd(0.3),
                              x, y, jax.random.key(0),
                              np.random.default_rng(3), **args)
    b, _ = spider_train.train(squared_error, zero_params, updates.sgd(0.3),
                              x, y, jax.random.key(1),
                              np.random.default_rng(3), **args)
    assert not jnp.allclose(a["w"], b["w"])


def test_the_sampling_stream_changes_the_run(problem, zero_params, key):
    x, y, _ = problem
    args = dict(steps=15, anchor_interval=4, **QUIET)
    a, _ = spider_train.train(squared_error, zero_params, updates.sgd(0.3),
                              x, y, key, np.random.default_rng(0), **args)
    b, _ = spider_train.train(squared_error, zero_params, updates.sgd(0.3),
                              x, y, key, np.random.default_rng(1), **args)
    assert not jnp.allclose(a["w"], b["w"])


def test_anchor_steps_occur_at_multiples_of_the_anchor_interval(
        problem, zero_params, key, rng, drawn_rates):
    """``mod(t, q) = 0``, with step 0 an anchor - not one step later."""
    x, y, _ = problem
    n = x.shape[0]
    steps, interval = 13, 4
    spider_train.train(
        squared_error, zero_params, updates.sgd(0.3), x, y, key, rng,
        steps=steps, anchor_interval=interval, **QUIET)

    anchor_q = QUIET["anchor_expected_batch_size"] / n
    variation_q = QUIET["variation_expected_batch_size"] / n
    assert drawn_rates == [
        anchor_q if t % interval == 0 else variation_q for t in range(steps)
    ]


def test_an_interval_of_one_makes_every_step_an_anchor(problem, zero_params,
                                                       key, rng, drawn_rates):
    """The degenerate phase length, where no variation release is taken."""
    x, y, _ = problem
    n = x.shape[0]
    spider_train.train(
        squared_error, zero_params, updates.sgd(0.3), x, y, key, rng,
        steps=7, anchor_interval=1, **QUIET)
    anchor_q = QUIET["anchor_expected_batch_size"] / n
    assert drawn_rates == [anchor_q] * 7


def test_the_output_iterate_is_one_the_loop_produced(problem, zero_params):
    """With two steps the support is ``{w_1}``, so the answer is pinned.

    Rebuilt from the primitives against the two streams consumed in the
    loop's documented order - the output index first, off the sampling
    generator, then the first step's key split and its draw.
    """
    x, y, _ = problem
    n = x.shape[0]
    opt = updates.sgd(0.3)
    output, final = spider_train.train(
        squared_error, zero_params, opt, x, y, jax.random.key(7),
        np.random.default_rng(7), steps=2, anchor_interval=4, **QUIET)

    rng = np.random.default_rng(7)
    rng.integers(1, 2)
    _, subkey = jax.random.split(jax.random.key(7))
    b_expected = QUIET["anchor_expected_batch_size"]
    b_max = poisson.padded_batch_size(b_expected, n)
    indices, mask = poisson.subsample(rng, n, b_expected / n, b_max)
    w_1, _, _ = spider_step.anchor_step(
        gradients.per_sample_grads(squared_error), opt, zero_params,
        updates.init(opt, zero_params), x[indices], y[indices],
        jnp.asarray(mask), subkey,
        expected_batch_size=b_expected,
        noise_scale=QUIET["anchor_noise_scale"])

    assert jnp.allclose(output["w"], w_1["w"], atol=1e-6)
    assert not jnp.allclose(final["w"], output["w"])


def test_the_output_iterate_is_never_the_final_one(problem, zero_params):
    """``w-bar`` is drawn from ``{w_1, .., w_{steps-1}}``; the last iterate
    is the one at which no estimate was ever taken, so no bound covers it."""
    x, y, _ = problem
    for seed in range(6):
        output, final = spider_train.train(
            squared_error, zero_params, updates.sgd(0.3), x, y,
            jax.random.key(seed), np.random.default_rng(seed),
            steps=8, anchor_interval=3, **QUIET)
        assert not jnp.allclose(output["w"], final["w"])


@pytest.mark.parametrize("steps", [0, 1])
def test_a_run_shorter_than_two_steps_is_rejected(problem, zero_params, key,
                                                  rng, steps):
    """Below two the output rule's support is empty; fail before spending."""
    x, y, _ = problem
    with pytest.raises(ValueError, match="steps"):
        spider_train.train(
            squared_error, zero_params, updates.sgd(0.3), x, y, key, rng,
            steps=steps, anchor_interval=4, **QUIET)


def test_a_non_positive_anchor_interval_is_rejected(problem, zero_params, key,
                                                    rng):
    x, y, _ = problem
    with pytest.raises(ValueError, match="anchor_interval"):
        spider_train.train(
            squared_error, zero_params, updates.sgd(0.3), x, y, key, rng,
            steps=10, anchor_interval=0, **QUIET)


@pytest.mark.parametrize(
    "name",
    ["anchor_expected_batch_size", "variation_expected_batch_size"],
)
def test_a_batch_larger_than_the_dataset_is_rejected(problem, zero_params,
                                                     key, rng, name):
    """q = b / n is a probability, per branch."""
    x, y, _ = problem
    args = {**QUIET, name: x.shape[0] + 1}
    with pytest.raises(ValueError, match=name):
        spider_train.train(
            squared_error, zero_params, updates.sgd(0.3), x, y, key, rng,
            steps=4, anchor_interval=4, **args)


def test_an_oversize_draw_propagates(problem, zero_params, key, rng):
    """Catching it would mean truncating or redrawing; both change the
    mechanism the accounting assumes."""
    x, y, _ = problem
    with pytest.raises(RuntimeError, match="b_max"):
        spider_train.train(
            squared_error, zero_params, updates.sgd(0.3), x, y, key, rng,
            steps=50, anchor_interval=1, anchor_expected_batch_size=300,
            variation_expected_batch_size=60, anchor_noise_scale=0.05,
            variation_noise_rate=0.5, variation_noise_cap=0.2,
            anchor_b_max=305)


def test_a_cap_of_n_never_raises(problem, zero_params, key, rng):
    """The exact, unraisable setting, at the cost of an O(n) batch."""
    x, y, _ = problem
    n = x.shape[0]
    output, final = spider_train.train(
        squared_error, zero_params, updates.sgd(0.3), x, y, key, rng,
        steps=6, anchor_interval=3, anchor_expected_batch_size=n // 2,
        variation_expected_batch_size=n // 2, anchor_noise_scale=0.05,
        variation_noise_rate=0.5, variation_noise_cap=0.2,
        anchor_b_max=n, variation_b_max=n)
    assert jnp.all(jnp.isfinite(output["w"]))
    assert jnp.all(jnp.isfinite(final["w"]))


def test_the_padding_caps_are_independent_per_branch(problem, zero_params,
                                                     key, rng):
    """A large anchor batch must not force a large variation batch."""
    x, y, _ = problem
    n = x.shape[0]
    output, _ = spider_train.train(
        squared_error, zero_params, updates.sgd(0.3), x, y, key, rng,
        steps=8, anchor_interval=4, anchor_expected_batch_size=n // 2,
        variation_expected_batch_size=20, anchor_noise_scale=0.05,
        variation_noise_rate=0.5, variation_noise_cap=0.2,
        anchor_b_max=n, variation_b_max=40)
    assert jnp.all(jnp.isfinite(output["w"]))


def test_the_loops_parameters_are_the_accountants_parameters(
        problem, zero_params, key, rng, drawn_rates):
    """The run and its epsilon must describe the same mechanism.

    The two branches are separate mechanisms and compose separately, so
    an accountant needs a rate and a composition count for each. Both
    are functions of `train`'s signature alone - the interval and the
    two expected batch sizes - and the three noise scales reach the
    releases unconverted. Nothing translates, so nothing can silently
    disagree with what ran.
    """
    x, y, _ = problem
    n, steps, interval = x.shape[0], 17, 5
    spider_train.train(
        squared_error, zero_params, updates.sgd(0.3), x, y, key, rng,
        steps=steps, anchor_interval=interval, **QUIET)

    anchor_q = QUIET["anchor_expected_batch_size"] / n
    variation_q = QUIET["variation_expected_batch_size"] / n
    anchor_compositions = math.ceil(steps / interval)

    assert len(drawn_rates) == steps
    assert drawn_rates.count(anchor_q) == anchor_compositions
    assert drawn_rates.count(variation_q) == steps - anchor_compositions


def test_a_whole_run_compiles_once_per_branch(problem, zero_params, key, rng,
                                              monkeypatch):
    """Two branches, two compilations, however many steps the run takes."""
    compilations = {"anchor": 0, "variation": 0}
    anchor_step, variation_step = (spider_step.anchor_step,
                                   spider_step.variation_step)

    def counted_anchor(*args, **kwargs):
        compilations["anchor"] += 1
        return anchor_step(*args, **kwargs)

    def counted_variation(*args, **kwargs):
        compilations["variation"] += 1
        return variation_step(*args, **kwargs)

    monkeypatch.setattr(spider_step, "anchor_step", counted_anchor)
    monkeypatch.setattr(spider_step, "variation_step", counted_variation)

    x, y, _ = problem
    spider_train.train(
        squared_error, zero_params, updates.sgd(0.3), x, y, key, rng,
        steps=40, anchor_interval=5, **QUIET)
    assert compilations == {"anchor": 1, "variation": 1}
