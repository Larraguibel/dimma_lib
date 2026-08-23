"""The inner mean estimator, against Section 3's Algorithm 1.

The seam is the last place one call's noise scale is observable: `step`
combines four of them.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dimma.algorithms.bias_reduced_sgd import estimators
from dimma.core import noise, projection

from ...helpers import tree_equal


def l1_norm(tree) -> float:
    return float(sum(jnp.sum(jnp.abs(leaf))
                     for leaf in jax.tree_util.tree_leaves(tree)))


def sparse_mean(gen: np.random.Generator, d: int, s: int, norm: float):
    """An ``s``-sparse vector of a given ``l_2`` norm, as (A.7) supplies."""
    z = np.zeros(d, dtype=np.float32)
    z[gen.choice(d, size=s, replace=False)] = gen.standard_normal(s)
    z *= norm / np.linalg.norm(z)
    return z


def test_at_zero_noise_the_estimator_is_the_projection(key):
    """Algorithm 1 without its perturbation is its second line alone."""
    estimator = estimators.projection_estimator(
        clip_norm=1.0, radius=0.75, noise_multiplier=0.0
    )
    mean = {"w": jnp.array([0.6, -0.4, 0.2]), "b": jnp.array(0.3)}
    assert tree_equal(
        estimator.estimate(mean, key, 8),
        projection.project_l1_ball_pytree(mean, 0.75),
    )


@pytest.mark.parametrize("batch_size", [1, 2, 8, 1024])
def test_the_noise_scale_is_the_multiplier_times_the_mean_sensitivity(
        batch_size):
    """``z * 2L/k``: the multiplier is fixed for the run and the scale
    tracks the slot's own cardinality, which is what lets one number
    serve four slots whose sizes differ by three orders of magnitude.

    Measured through a ball wide enough that the projection is the
    identity, so what is measured is the perturbation and not the
    geometry.
    """
    clip_norm, multiplier, d = 1.5, 2.0, 200
    estimator = estimators.projection_estimator(
        clip_norm=clip_norm, radius=1e6, noise_multiplier=multiplier
    )
    mean = {"w": jnp.zeros(d)}
    keys = jax.random.split(jax.random.key(0), 2000)
    draws = jax.vmap(
        lambda k: estimator.estimate(mean, k, batch_size)["w"]
    )(keys)
    expected = multiplier * 2.0 * clip_norm / batch_size
    assert np.allclose(np.std(np.asarray(draws), axis=0), expected, rtol=0.12)


def test_the_estimate_lies_in_the_ball(key):
    """``K`` is the constraint set, so every release is in it whatever
    the noise did."""
    radius = 2.0
    estimator = estimators.projection_estimator(
        clip_norm=1.0, radius=radius, noise_multiplier=5.0
    )
    mean = {"w": jnp.zeros(120)}
    for k in jax.random.split(key, 20):
        assert l1_norm(estimator.estimate(mean, k, 4)) <= radius + 1e-4


def test_a_batch_of_one_is_not_special_cased(key):
    """The ``G_0`` slot goes through the same call as the others.

    Its noise is simply the largest, at ``2L/1``, and the estimate is
    the same two lines it is everywhere else — no branch, and none
    needed: Algorithm 1 carries no regime condition, while the paper's
    own Algorithm 2 fails at a batch of one.
    """
    clip_norm, multiplier, radius = 1.0, 3.0, 4.0
    estimator = estimators.projection_estimator(
        clip_norm=clip_norm, radius=radius, noise_multiplier=multiplier
    )
    mean = {"w": jnp.array([0.3, -0.2, 0.1])}
    for batch_size in (1, 16):
        scale = multiplier * 2.0 * clip_norm / batch_size
        want = projection.project_l1_ball_pytree(
            noise.add_gaussian(mean, key, scale), radius
        )
        assert tree_equal(estimator.estimate(mean, key, batch_size), want)


def test_the_claim_carries_the_clip_norm_and_the_multiplier():
    """What the accountant reads, and the name of what ran."""
    estimator = estimators.projection_estimator(
        clip_norm=1.25, radius=3.0, noise_multiplier=4.5
    )
    assert estimator.name == "projection"
    assert isinstance(estimator.claim, estimators.GaussianMeanClaim)
    assert estimator.claim.clip_norm == 1.25
    assert estimator.claim.noise_multiplier == 4.5
    assert not hasattr(estimator.claim, "batch_size")


def test_the_second_moment_bound_of_the_research_note():
    """`docs/research/algorithm-1-carries-algorithm-3.md`, point (b)+(c),
    swept over ``d`` so that each branch of the minimum binds somewhere
    and neither is asserted vacuously."""
    clip_norm, s, k, epsilon, delta = 1.0, 5, 64, 1.0, 1e-5
    multiplier = math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    sigma = multiplier * 2.0 * clip_norm / k
    estimator = estimators.projection_estimator(
        clip_norm=clip_norm,
        radius=clip_norm * math.sqrt(s),
        noise_multiplier=multiplier,
    )

    gen = np.random.default_rng(3)
    binding = []
    for d in (50, 200, 800):
        zbar = sparse_mean(gen, d, s, 0.8 * clip_norm)
        mean = {"w": jnp.asarray(zbar)}
        keys = jax.random.split(jax.random.key(d), 400)
        draws = np.asarray(
            jax.vmap(lambda key: estimator.estimate(mean, key, k)["w"])(keys)
        )
        second_moment = float(np.mean(np.sum((draws - zbar) ** 2, axis=1)))

        dense = d * sigma ** 2
        sparse = (8.0 * clip_norm ** 2
                  * math.sqrt(s * math.log(2 * d) * math.log(1.25 / delta))
                  / (k * epsilon))
        assert second_moment <= sparse
        assert second_moment <= dense
        binding.append("sparse" if sparse < dense else "dense")

    assert binding == ["dense", "sparse", "sparse"]


@pytest.mark.parametrize("bad", [
    dict(clip_norm=0.0), dict(radius=-1.0), dict(noise_multiplier=-0.5),
])
def test_the_factory_rejects_a_number_that_describes_no_mechanism(bad):
    """A non-positive bound calibrates against nothing, and a negative
    ball or scale is not a mechanism at all."""
    arguments = dict(clip_norm=1.0, radius=1.0, noise_multiplier=1.0) | bad
    with pytest.raises(ValueError, match=f"{next(iter(bad))}="):
        estimators.projection_estimator(**arguments)
