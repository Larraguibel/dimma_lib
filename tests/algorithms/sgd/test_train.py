"""The SGD loop: stage 1, one random stream, and the arm it is half of."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dimma.algorithms.dp_sgd import train as dp_train
from dimma.algorithms.sgd import train as sgd_train
from dimma.core import updates

from .conftest import squared_error


def full_loss(params, x, y):
    """Evaluation on the training data. Free here; in the private arm it
    is an unaccounted access, which is why neither loop reports it."""
    return float(jnp.mean(
        jax.vmap(squared_error, in_axes=(None, 0, 0))(params, x, y)
    ))


def test_training_reduces_the_loss(problem, zero_params, rng):
    x, y, _ = problem
    params = sgd_train.train(
        squared_error, zero_params, updates.sgd(0.5), x, y, rng,
        steps=60, batch_size=100,
    )
    assert full_loss(params, x, y) < full_loss(zero_params, x, y)


def test_training_recovers_the_true_parameters(problem, zero_params, rng):
    """A baseline written to lose makes every comparison against it
    meaningless (ADR-0005). This one converges."""
    x, y, w_true = problem
    params = sgd_train.train(
        squared_error, zero_params, updates.sgd(0.2), x, y, rng,
        steps=300, batch_size=100,
    )
    assert jnp.linalg.norm(w_true - params["w"]) < 0.05


def test_zero_steps_returns_the_initial_parameters(problem, zero_params, rng):
    x, y, _ = problem
    params = sgd_train.train(
        squared_error, zero_params, updates.sgd(0.5), x, y, rng,
        steps=0, batch_size=100,
    )
    assert jnp.array_equal(params["w"], zero_params["w"])


def test_a_run_is_reproducible_from_its_one_seed(problem, zero_params):
    x, y, _ = problem

    def run():
        return sgd_train.train(
            squared_error, zero_params, updates.sgd(0.2), x, y,
            np.random.default_rng(3), steps=15, batch_size=100)

    assert jnp.array_equal(run()["w"], run()["w"])


def test_the_sampling_stream_changes_the_run(problem, zero_params):
    """The only randomness there is, so it has to be visible."""
    x, y, _ = problem
    args = dict(steps=15, batch_size=100)
    a = sgd_train.train(squared_error, zero_params, updates.sgd(0.2), x, y,
                        np.random.default_rng(0), **args)
    b = sgd_train.train(squared_error, zero_params, updates.sgd(0.2), x, y,
                        np.random.default_rng(1), **args)
    assert not jnp.allclose(a["w"], b["w"])


def test_the_loop_takes_no_key(problem, zero_params):
    """One stream, not two: `train`'s signature is DP-SGD's minus the
    noise key and the three privacy parameters."""
    names = set(inspect.signature(sgd_train.train).parameters)
    assert "key" not in names
    assert not names & {"clip_norm", "noise_multiplier", "b_max",
                        "expected_batch_size"}


def test_the_run_crosses_epoch_boundaries(problem, zero_params, rng):
    """600 examples at 100 a batch is six steps to an epoch; a 60-step
    run reshuffles ten times and must not run out of batches."""
    x, y, _ = problem
    params = sgd_train.train(
        squared_error, zero_params, updates.sgd(0.1), x, y, rng,
        steps=60, batch_size=100,
    )
    assert jnp.all(jnp.isfinite(params["w"]))


def test_a_batch_larger_than_the_dataset_is_rejected(problem, zero_params,
                                                     rng):
    x, y, _ = problem
    with pytest.raises(ValueError, match="batch_size"):
        sgd_train.train(
            squared_error, zero_params, updates.sgd(0.5), x, y, rng,
            steps=1, batch_size=x.shape[0] + 1,
        )


def test_the_batch_size_is_checked_before_the_first_step(problem, zero_params,
                                                         rng):
    """Even at zero steps, so a misconfigured run fails at the call."""
    x, y, _ = problem
    with pytest.raises(ValueError, match="batch_size"):
        sgd_train.train(
            squared_error, zero_params, updates.sgd(0.5), x, y, rng,
            steps=0, batch_size=x.shape[0] + 1,
        )


def test_a_schedule_threads_through(problem, zero_params):
    """The same seam DP-SGD's eta_t uses, and an optax schedule to pin
    that `updates.sgd` takes any ``count -> rate`` callable."""
    x, y, _ = problem
    args = dict(steps=30, batch_size=100)
    decayed = sgd_train.train(
        squared_error, zero_params,
        updates.sgd(optax.cosine_decay_schedule(0.5, decay_steps=30)),
        x, y, np.random.default_rng(1), **args)
    constant = sgd_train.train(
        squared_error, zero_params, updates.sgd(0.5), x, y,
        np.random.default_rng(1), **args)
    assert not jnp.allclose(decayed["w"], constant["w"])


def test_the_baseline_beats_its_private_counterpart(problem, zero_params, key):
    """The comparison ADR-0005 exists to make possible, run once here.

    Both arms take the same loss, the same initial parameters, the same
    optimizer, the same step count and a batch of the same size — the
    private one in expectation, since Poisson cardinality is random.
    What is left between them is the privacy, and at this noise level it
    costs visibly.
    """
    x, y, _ = problem
    optimizer, steps = updates.sgd(0.5), 100
    private = dp_train.train(
        squared_error, zero_params, optimizer, x, y, key,
        np.random.default_rng(0), steps=steps,
        expected_batch_size=100, clip_norm=1.0, noise_multiplier=20.0,
    )
    baseline = sgd_train.train(
        squared_error, zero_params, optimizer, x, y,
        np.random.default_rng(0), steps=steps, batch_size=100,
    )
    assert full_loss(baseline, x, y) < full_loss(private, x, y)


def test_one_optimizer_object_drives_both_arms(problem, zero_params, key):
    """ADR-0002 has both sides name the same rule, so the same instance
    is handed to both — which is sound only if it carries nothing
    between runs. A baseline run is unchanged by the private arm having
    used the optimizer first."""
    x, y, _ = problem
    optimizer = updates.sgd(0.3)
    args = dict(steps=10, batch_size=100)
    before = sgd_train.train(squared_error, zero_params, optimizer, x, y,
                             np.random.default_rng(0), **args)
    dp_train.train(squared_error, zero_params, optimizer, x, y, key,
                   np.random.default_rng(0), steps=10,
                   expected_batch_size=100, clip_norm=1.0,
                   noise_multiplier=1.0)
    after = sgd_train.train(squared_error, zero_params, optimizer, x, y,
                            np.random.default_rng(0), **args)
    assert jnp.array_equal(before["w"], after["w"])
