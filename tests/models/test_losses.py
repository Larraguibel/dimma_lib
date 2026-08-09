"""Sigmoid BCE: the stable form, and the contract an algorithm takes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dimma.core import gradients, updates
from dimma.models import logreg, losses


D = 4


def naive_bce(logit, y):
    """The textbook form, written the way it invites being written."""
    s = jax.nn.sigmoid(logit)
    return -(y * jnp.log(s) + (1.0 - y) * jnp.log(1.0 - s))


@pytest.fixture
def trained() -> dict:
    return {"w": jnp.array([1.0, -2.0, 0.5, 3.0]), "b": jnp.array(-0.25)}


@pytest.fixture
def batch():
    x = jax.random.normal(jax.random.key(3), (8, D))
    y = jnp.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    return x, y


# ---------------------------------------------------------------------------
# _stable_bce
# ---------------------------------------------------------------------------

def reference_bce(z, y):
    """The same loss in float64, where neither form is under strain."""
    z, y = np.float64(z), np.float64(y)
    return np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))


@pytest.mark.parametrize("z", [-5.0, -3.0, -0.5, 0.0, 0.5, 3.0, 5.0])
@pytest.mark.parametrize("y", [0.0, 1.0])
def test_stable_bce_is_the_textbook_loss_rearranged(z, y):
    """Where float32 puts the direct form under no strain, they agree."""
    got = losses._stable_bce(jnp.array(z), jnp.array(y))
    assert jnp.allclose(got, naive_bce(jnp.array(z), jnp.array(y)), atol=1e-5)
    assert jnp.allclose(got, reference_bce(z, y), atol=1e-5)


@pytest.mark.parametrize("z", [-40.0, -20.0, -8.0, 8.0, 20.0, 40.0])
@pytest.mark.parametrize("y", [0.0, 1.0])
def test_stable_bce_stays_accurate_where_the_direct_form_does_not(z, y):
    """The saturated tail, against a float64 reference."""
    got = losses._stable_bce(jnp.array(z), jnp.array(y))
    assert jnp.allclose(got, reference_bce(z, y), rtol=1e-6)


def relative_error(z: float) -> float:
    """How far the direct form has drifted from the true loss at ``y = 0``."""
    got = float(naive_bce(jnp.array(z), jnp.array(0.0)))
    reference = reference_bce(z, 0.0)
    return abs(got - reference) / reference


def test_the_direct_form_holds_up_to_a_logit_of_fourteen():
    """The lower end of the window, so the docstring's ``15`` is pinned.

    Both ends matter: quoting the threshold too low would make the
    rearrangement look more urgent than it is, and too high would leave
    a band where the direct form is quietly wrong and nobody looks.
    """
    assert relative_error(14.0) < 1e-3
    assert relative_error(15.0) > 1e-2


def test_the_direct_form_goes_wrong_before_it_goes_non_finite():
    """Why the rearrangement is not merely defensive.

    In float32 ``1 - sigmoid(z)`` loses its leading digits from around
    ``|z| = 15`` and hits exactly 0 around 17, so the direct form
    returns a plausible wrong number first and ``inf`` / ``nan`` only
    afterwards. A silently wrong loss is the worse of the two.
    """
    y = jnp.array(0.0)

    quietly_wrong = naive_bce(jnp.array(16.0), y)
    assert jnp.isfinite(quietly_wrong)
    assert abs(float(quietly_wrong) - reference_bce(16.0, 0.0)) > 1e-2
    assert jnp.allclose(losses._stable_bce(jnp.array(16.0), y),
                        reference_bce(16.0, 0.0), rtol=1e-6)

    assert not jnp.isfinite(naive_bce(jnp.array(17.0), y))
    assert jnp.isfinite(losses._stable_bce(jnp.array(17.0), y))


@pytest.mark.parametrize("z", [17.0, 40.0, 1e4])
@pytest.mark.parametrize("y", [0.0, 1.0])
def test_stable_bce_is_finite_where_the_textbook_form_is_not(z, y):
    z, y = jnp.array(z), jnp.array(y)
    assert not jnp.isfinite(naive_bce(z, y)) or not jnp.isfinite(naive_bce(-z, y))
    assert jnp.isfinite(losses._stable_bce(z, y))
    assert jnp.isfinite(losses._stable_bce(-z, y))


@pytest.mark.parametrize("z", [-1e4, -200.0, 200.0, 1e4])
@pytest.mark.parametrize("y", [0.0, 1.0])
def test_stable_bce_tends_to_the_hinge_asymptote(z, y):
    z, y = jnp.array(z), jnp.array(y)
    assert jnp.allclose(losses._stable_bce(z, y), jnp.maximum(z, 0.0) - z * y)


def test_stable_bce_is_never_negative():
    z = jnp.linspace(-30.0, 30.0, 201)
    for y in (0.0, 1.0):
        assert jnp.all(losses._stable_bce(z, jnp.array(y)) >= 0.0)


def test_stable_bce_broadcasts_over_a_batch():
    z = jnp.array([-1.0, 0.0, 2.0])
    y = jnp.array([0.0, 1.0, 1.0])
    out = losses._stable_bce(z, y)
    assert out.shape == (3,)
    assert jnp.allclose(out, jnp.stack([losses._stable_bce(z[i], y[i])
                                        for i in range(3)]))


# ---------------------------------------------------------------------------
# per_sample_bce_loss - the shape an algorithm is handed
# ---------------------------------------------------------------------------

def test_per_sample_loss_is_a_scalar(trained):
    x = jnp.ones((D,))
    loss = losses.per_sample_bce_loss(trained, x, jnp.array(1.0))
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_per_sample_gradient_matches_the_closed_form(trained):
    """``(sigmoid(z) - y) * x`` in ``w``, ``sigmoid(z) - y`` in ``b``."""
    x = jnp.array([1.0, -0.5, 2.0, 4.0])
    y = jnp.array(1.0)
    grad = jax.grad(losses.per_sample_bce_loss)(trained, x, y)
    residual = jax.nn.sigmoid(logreg.forward(trained, x)) - y
    assert jnp.allclose(grad["w"], residual * x, atol=1e-6)
    assert jnp.allclose(grad["b"], residual, atol=1e-6)


def test_per_sample_gradient_is_dense(trained):
    """Unlike the hashed model it came from, every entry is touched."""
    x = jnp.array([1.0, -0.5, 2.0, 4.0])
    grad = jax.grad(losses.per_sample_bce_loss)(trained, x, jnp.array(0.0))
    assert jnp.count_nonzero(grad["w"]) == D


def test_it_satisfies_the_stage_3_contract(trained, batch):
    """``core.gradients`` vectorizes it without adaptation."""
    x, y = batch
    grads = gradients.per_sample_grads(losses.per_sample_bce_loss)(trained, x, y)
    assert grads["w"].shape == (x.shape[0], D)
    assert grads["b"].shape == (x.shape[0],)
    assert jnp.all(jnp.isfinite(grads["w"]))


def test_batch_grads_also_takes_it(trained, batch):
    """The baselines differentiate the same per-sample loss."""
    x, y = batch
    grad = gradients.batch_grads(losses.per_sample_bce_loss)(trained, x, y)
    assert grad["w"].shape == (D,)
    assert grad["b"].shape == ()


def test_a_saturated_example_stays_finite_through_the_gradient(trained):
    """Where the direct form would have produced ``nan``."""
    x = jnp.array([100.0, 0.0, 0.0, 0.0])
    grad = jax.grad(losses.per_sample_bce_loss)(trained, x, jnp.array(0.0))
    assert jnp.all(jnp.isfinite(grad["w"]))
    assert jnp.isfinite(grad["b"])


# ---------------------------------------------------------------------------
# batch_bce_loss
# ---------------------------------------------------------------------------

def test_batch_loss_is_the_mean_of_the_per_sample_losses(trained, batch):
    x, y = batch
    per_sample = jnp.stack([
        losses.per_sample_bce_loss(trained, x[i], y[i]) for i in range(x.shape[0])
    ])
    assert jnp.allclose(losses.batch_bce_loss(trained, x, y),
                        jnp.mean(per_sample), atol=1e-6)


def test_batch_loss_is_a_finite_scalar(trained, batch):
    x, y = batch
    loss = losses.batch_bce_loss(trained, x, y)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


# ---------------------------------------------------------------------------
# End to end - the reason this ticket blocks the Criteo work
# ---------------------------------------------------------------------------

def test_dp_sgd_trains_the_shipped_model(key):
    """An algorithm runs on the model and loss with no glue in between."""
    from dimma.algorithms.dp_sgd import train as dp_sgd_train

    n = 64
    x = jax.random.normal(jax.random.key(4), (n, D))
    y = (jax.random.uniform(jax.random.key(5), (n,)) < 0.5).astype(jnp.float32)
    params = logreg.init_params(key, D)

    trained = dp_sgd_train.train(
        losses.per_sample_bce_loss,
        params,
        updates.sgd(0.1),
        x, y,
        jax.random.key(6),
        np.random.default_rng(0),
        steps=3,
        expected_batch_size=16,
        clip_norm=1.0,
        noise_multiplier=1.0,
        b_max=n,
    )

    assert trained["w"].shape == (D,)
    assert jnp.all(jnp.isfinite(trained["w"]))
    assert jnp.isfinite(trained["b"])
    assert not jnp.array_equal(trained["w"], params["w"])
