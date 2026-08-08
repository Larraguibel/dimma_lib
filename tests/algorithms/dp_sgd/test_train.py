"""The DP-SGD loop: stage 1, state threading, and what it refuses to do."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dimma.accounting.sampling import poisson_gaussian_epsilon
from dimma.algorithms.dp_sgd import train as dp_train
from dimma.core import updates

from .conftest import squared_error


def full_loss(params, x, y):
    """Non-private evaluation, for the tests only."""
    return float(jnp.mean(
        jax.vmap(squared_error, in_axes=(None, 0, 0))(params, x, y)
    ))


def test_training_reduces_the_loss(problem, zero_params, key, rng):
    x, y, _ = problem
    params = dp_train.train(
        squared_error, zero_params, updates.sgd(0.5), x, y, key, rng,
        steps=60, expected_batch_size=100, clip_norm=1.0,
        noise_multiplier=0.5,
    )
    assert full_loss(params, x, y) < full_loss(zero_params, x, y)


def test_training_moves_toward_the_true_parameters(problem, zero_params, key,
                                                   rng):
    """Not a strawman: with a modest sigma it recovers the signal."""
    x, y, w_true = problem
    params = dp_train.train(
        squared_error, zero_params, updates.sgd(0.5), x, y, key, rng,
        steps=200, expected_batch_size=100, clip_norm=2.0,
        noise_multiplier=0.5,
    )
    before = jnp.linalg.norm(w_true - zero_params["w"])
    assert jnp.linalg.norm(w_true - params["w"]) < 0.5 * before


def test_more_noise_hurts(problem, zero_params, key, rng):
    """The privacy-utility tradeoff is visible in the loop's output."""
    x, y, _ = problem
    args = dict(steps=100, expected_batch_size=100, clip_norm=2.0)
    quiet = dp_train.train(
        squared_error, zero_params, updates.sgd(0.5), x, y, key,
        np.random.default_rng(0), noise_multiplier=0.5, **args)
    loud = dp_train.train(
        squared_error, zero_params, updates.sgd(0.5), x, y, key,
        np.random.default_rng(0), noise_multiplier=50.0, **args)
    assert full_loss(quiet, x, y) < full_loss(loud, x, y)


def test_a_run_is_reproducible_from_its_two_seeds(problem, zero_params):
    """One key and one generator determine the run."""
    x, y, _ = problem
    args = dict(steps=15, expected_batch_size=100, clip_norm=1.0,
                noise_multiplier=1.0)

    def run():
        return dp_train.train(
            squared_error, zero_params, updates.sgd(0.2), x, y,
            jax.random.key(3), np.random.default_rng(3), **args)

    assert jnp.array_equal(run()["w"], run()["w"])


def test_the_noise_stream_changes_the_run(problem, zero_params, rng):
    x, y, _ = problem
    args = dict(steps=15, expected_batch_size=100, clip_norm=1.0,
                noise_multiplier=1.0)
    a = dp_train.train(squared_error, zero_params, updates.sgd(0.2), x, y,
                       jax.random.key(0), np.random.default_rng(3), **args)
    b = dp_train.train(squared_error, zero_params, updates.sgd(0.2), x, y,
                       jax.random.key(1), np.random.default_rng(3), **args)
    assert not jnp.allclose(a["w"], b["w"])


def test_zero_steps_returns_the_initial_parameters(problem, zero_params, key,
                                                   rng):
    x, y, _ = problem
    params = dp_train.train(
        squared_error, zero_params, updates.sgd(0.5), x, y, key, rng,
        steps=0, expected_batch_size=100, clip_norm=1.0, noise_multiplier=1.0,
    )
    assert jnp.array_equal(params["w"], zero_params["w"])


def test_a_batch_larger_than_the_dataset_is_rejected(problem, zero_params,
                                                     key, rng):
    """q = L / n is a probability."""
    x, y, _ = problem
    with pytest.raises(ValueError, match="expected_batch_size"):
        dp_train.train(
            squared_error, zero_params, updates.sgd(0.5), x, y, key, rng,
            steps=1, expected_batch_size=x.shape[0] + 1, clip_norm=1.0,
            noise_multiplier=1.0,
        )


def test_an_oversize_draw_propagates(problem, zero_params, key, rng):
    """Catching it would mean truncating or redrawing; both change the
    mechanism the accounting assumes."""
    x, y, _ = problem
    with pytest.raises(RuntimeError, match="b_max"):
        dp_train.train(
            squared_error, zero_params, updates.sgd(0.5), x, y, key, rng,
            steps=50, expected_batch_size=300, clip_norm=1.0,
            noise_multiplier=1.0,
            b_max=305,
        )


def test_a_cap_of_n_never_raises(problem, zero_params, key, rng):
    """The exact, unraisable setting, at the cost of an O(n) batch."""
    x, y, _ = problem
    n = x.shape[0]
    params = dp_train.train(
        squared_error, zero_params, updates.sgd(0.5), x, y, key, rng,
        steps=5, expected_batch_size=n // 2, clip_norm=1.0,
        noise_multiplier=1.0,
        b_max=n,
    )
    assert jnp.all(jnp.isfinite(params["w"]))


def test_the_loops_parameters_are_the_accountants_parameters(problem,
                                                             zero_params, key,
                                                             rng):
    """The run and its epsilon must describe the same mechanism.

    `train` takes ``expected_batch_size``, ``steps`` and
    ``noise_multiplier``; the accountant takes
    ``q = expected_batch_size / n``, ``num_compositions = steps`` and the
    same ``noise_multiplier``, unconverted. Nothing else is
    needed, which is what makes DP-SGD the case the standard accountant
    covers exactly.
    """
    x, y, _ = problem
    n, expected_batch_size, steps, sigma = x.shape[0], 100, 50, 1.1

    dp_train.train(
        squared_error, zero_params, updates.sgd(0.5), x, y, key, rng,
        steps=steps, expected_batch_size=expected_batch_size, clip_norm=1.0,
        noise_multiplier=sigma,
    )
    epsilon = poisson_gaussian_epsilon(
        sampling_probability=expected_batch_size / n,
        noise_multiplier=sigma,
        num_compositions=steps,
        target_delta=1e-5,
    )
    assert epsilon > 0.0


def test_a_schedule_supplies_the_eta_t_subscript(problem, zero_params, key,
                                                 rng):
    """Algorithm 1 writes eta_t; a schedule indexes on update calls.

    An optax schedule, to pin that `updates.sgd` takes any
    ``count -> rate`` callable rather than only dimma's own.
    """
    x, y, _ = problem
    args = dict(steps=30, expected_batch_size=100, clip_norm=1.0,
                noise_multiplier=0.5)
    decayed = dp_train.train(
        squared_error, zero_params,
        updates.sgd(optax.cosine_decay_schedule(0.5, decay_steps=30)),
        x, y, key, np.random.default_rng(1), **args)
    constant = dp_train.train(
        squared_error, zero_params, updates.sgd(0.5), x, y, key,
        np.random.default_rng(1), **args)
    assert not jnp.allclose(decayed["w"], constant["w"])
