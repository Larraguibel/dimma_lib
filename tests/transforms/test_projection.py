"""The l1 projection transform, at the optimizer seam."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from jax.flatten_util import ravel_pytree

from dimma.core import projection as geometry
from dimma.core import updates
from dimma.transforms import projection

from tests.helpers import tree_allclose


def global_l1(tree) -> float:
    flat, _ = ravel_pytree(tree)
    return float(jnp.sum(jnp.abs(flat)))


@pytest.fixture
def grad(params):
    return jax.tree.map(lambda leaf: jnp.full_like(leaf, 0.5), params)


def test_every_update_lands_inside_the_ball(params, grad):
    """The transform's whole claim, over a loop rather than one call."""
    radius = 2.0
    opt = projection.l1_projected(updates.sgd(1.0), radius)
    current, state = params, updates.init(opt, params)
    assert global_l1(current) > radius  # or the loop below tests nothing
    for _ in range(3):
        current, state = updates.apply(opt, current, grad, state)
        assert global_l1(current) <= radius + 1e-4


def test_inside_the_ball_the_wrapped_rule_matches_the_bare_one(params, grad):
    """A ball the trajectory never touches leaves the rule as it was."""
    bare = updates.sgd(0.1)
    wrapped = projection.l1_projected(updates.sgd(0.1), radius=1e6)
    a, _ = updates.apply(bare, params, grad, updates.init(bare, params))
    b, _ = updates.apply(wrapped, params, grad, updates.init(wrapped, params))
    assert tree_allclose(a, b)


def test_the_ball_is_global_across_leaves_not_per_leaf(grad):
    """Each leaf alone fits the ball; only their concatenation exceeds
    it. A per-leaf projection would leave this pytree unchanged."""
    radius = 2.0
    inside_per_leaf = {"a": jnp.array([1.5, 0.0]), "b": jnp.array([0.0, 1.5])}
    zero = jax.tree.map(jnp.zeros_like, inside_per_leaf)
    opt = projection.l1_projected(updates.sgd(1.0), radius)
    out, _ = updates.apply(
        opt, inside_per_leaf, zero, updates.init(opt, inside_per_leaf)
    )
    assert global_l1(out) <= radius + 1e-5
    assert not tree_allclose(out, inside_per_leaf)


def test_the_wrapper_agrees_with_the_geometry_it_applies(params, grad):
    """One step of the wrapped rule is one bare step, projected."""
    radius = 2.0
    bare = updates.sgd(1.0)
    wrapped = projection.l1_projected(updates.sgd(1.0), radius)
    stepped, _ = updates.apply(bare, params, grad, updates.init(bare, params))
    projected, _ = updates.apply(
        wrapped, params, grad, updates.init(wrapped, params)
    )
    assert tree_allclose(
        projected, geometry.project_l1_ball_pytree(stepped, radius),
        atol=1e-6,
    )


def test_the_inner_state_threads_unchanged(params, grad):
    """The wrapper adds no state: `init` and the count are the wrapped
    rule's, so a schedule keeps indexing the same count."""
    opt = projection.l1_projected(updates.sgd(0.1), 1.0)
    current, state = params, updates.init(opt, params)
    assert isinstance(state, updates.SgdState)
    assert int(state.count) == 0
    for _ in range(3):
        current, state = updates.apply(opt, current, grad, state)
    assert int(state.count) == 3


def test_an_optax_optimizer_wraps_the_same_way(params, grad):
    """The seam is structural, so the wrapper is too — a baseline's Adam
    can be constrained to the same ball as an algorithm's sgd."""
    radius = 1.0
    opt = projection.l1_projected(optax.adam(0.5), radius)
    current, state = params, updates.init(opt, params)
    for _ in range(2):
        current, state = updates.apply(opt, current, grad, state)
        assert global_l1(current) <= radius + 1e-4
    assert jax.tree_util.tree_structure(current) == \
        jax.tree_util.tree_structure(params)


def test_radius_may_be_traced(params, grad):
    """Callers deriving the radius at runtime depend on this, as they do
    for the geometry underneath."""

    def run(current, estimate, state, radius):
        opt = projection.l1_projected(updates.sgd(1.0), radius)
        return updates.apply(opt, current, estimate, state)

    state = updates.init(updates.sgd(1.0), params)
    eager, _ = run(params, grad, state, 2.0)
    jitted, _ = jax.jit(run)(params, grad, state, jnp.float32(2.0))
    assert tree_allclose(eager, jitted)


def test_zero_radius_projects_to_the_origin(params, grad):
    opt = projection.l1_projected(updates.sgd(0.1), 0.0)
    out, _ = updates.apply(opt, params, grad, updates.init(opt, params))
    assert all(
        jnp.all(leaf == 0.0) for leaf in jax.tree_util.tree_leaves(out)
    )


def test_update_without_params_raises(params, grad):
    """Silently skipping the projection would be an unconstrained run
    reported as a constrained one."""
    opt = projection.l1_projected(updates.sgd(0.1), 1.0)
    with pytest.raises(ValueError, match="params"):
        opt.update(grad, updates.init(opt, params))


def test_a_negative_concrete_radius_is_rejected():
    with pytest.raises(ValueError, match="radius"):
        projection.l1_projected(updates.sgd(0.1), -1.0)


def test_the_guard_does_not_reject_array_radii(params, grad):
    """The check inspects concrete Python numbers, not arrays."""
    for radius in (jnp.float32(2.0),):
        opt = projection.l1_projected(updates.sgd(1.0), radius)
        out, _ = updates.apply(opt, params, grad, updates.init(opt, params))
        assert global_l1(out) <= 2.0 + 1e-4


# --- the estimate-side wrapper ------------------------------------------------


def probe() -> updates.Optimizer:
    """A rule whose increment is its estimate, so a test can read what
    the wrapper fed it."""
    return updates.Optimizer(
        lambda params: None,
        lambda estimate, state, params=None: (estimate, state),
    )


def test_the_rule_consumes_the_projected_estimate(params, grad):
    """One wrapped step is one bare step on the projected estimate."""
    radius = 1.0
    bare = updates.sgd(1.0)
    wrapped = projection.l1_projected_estimate(updates.sgd(1.0), radius)
    a, _ = updates.apply(
        bare, params, geometry.project_l1_ball_pytree(grad, radius),
        updates.init(bare, params),
    )
    b, _ = updates.apply(wrapped, params, grad, updates.init(wrapped, params))
    assert tree_allclose(a, b, atol=1e-7)


def test_an_estimate_inside_the_ball_passes_through_unchanged(params, grad):
    """A ball the estimate never leaves leaves the rule as it was."""
    bare = updates.sgd(0.1)
    wrapped = projection.l1_projected_estimate(updates.sgd(0.1), radius=1e6)
    a, _ = updates.apply(bare, params, grad, updates.init(bare, params))
    b, _ = updates.apply(wrapped, params, grad, updates.init(wrapped, params))
    assert tree_allclose(a, b)


def test_the_iterates_are_not_constrained(params):
    """The dual of `l1_projected`: the estimate lands in the ball, the
    parameters go wherever the rule sends them."""
    radius = 0.5
    assert global_l1(params) > radius
    zero = jax.tree.map(jnp.zeros_like, params)
    opt = projection.l1_projected_estimate(updates.sgd(1.0), radius)
    out, _ = updates.apply(opt, params, zero, updates.init(opt, params))
    assert tree_allclose(out, params)
    assert global_l1(out) > radius


def test_the_estimate_ball_is_global_across_leaves_not_per_leaf(params):
    """Each leaf alone fits the ball; only their concatenation exceeds
    it. A per-leaf projection would pass this estimate through."""
    radius = 2.0
    estimate = {"a": jnp.array([1.5, 0.0]), "b": jnp.array([0.0, 1.5])}
    zero = jax.tree.map(jnp.zeros_like, estimate)
    opt = projection.l1_projected_estimate(probe(), radius)
    fed, _ = updates.apply(opt, zero, estimate, updates.init(opt, zero))
    assert global_l1(fed) <= radius + 1e-5
    assert not tree_allclose(fed, estimate)


def test_the_denoising_bound_of_lemma_31():
    """Ghazi et al. 2024, Lemma 3.1: for an s-sparse mean with l_2 <= L
    and any dense noise, projecting the noisy estimate onto the ball of
    radius L*sqrt(s) satisfies |zhat - mean|_2 <= sqrt(2 L |xi|_inf sqrt(s)).
    The bound is the geometry's, so it holds through the wrapper for any
    noise the mechanism upstream added."""
    import math

    import numpy as np

    L, s, d = 1.0, 5, 300
    radius = L * math.sqrt(s)
    gen = np.random.default_rng(4)
    zero = jnp.zeros(d)
    opt = projection.l1_projected_estimate(probe(), radius)
    state = updates.init(opt, zero)
    for t in range(50):
        mean = np.zeros(d, dtype=np.float32)
        idx = gen.choice(d, size=s, replace=False)
        mean[idx] = gen.standard_normal(s).astype(np.float32)
        mean *= 0.8 * L / np.linalg.norm(mean)
        xi = gen.laplace(scale=0.05, size=d).astype(np.float32)
        estimate = jnp.asarray(mean + xi)
        zhat, _ = updates.apply(opt, zero, estimate, state)
        lhs = float(jnp.linalg.norm(zhat - mean))
        rhs = math.sqrt(2.0 * L * float(np.max(np.abs(xi))) * math.sqrt(s))
        assert lhs <= rhs + 1e-4
        # And the projection denoises: the fed estimate sits closer to
        # the mean than the noisy one the mechanism released.
        assert lhs < float(jnp.linalg.norm(estimate - jnp.asarray(mean)))


def test_the_estimate_wrapper_threads_the_inner_state_unchanged(params, grad):
    opt = projection.l1_projected_estimate(updates.sgd(0.1), 1.0)
    current, state = params, updates.init(opt, params)
    assert isinstance(state, updates.SgdState)
    assert int(state.count) == 0
    for _ in range(3):
        current, state = updates.apply(opt, current, grad, state)
    assert int(state.count) == 3


def test_an_optax_optimizer_wraps_the_estimate_side_the_same_way(params, grad):
    opt = projection.l1_projected_estimate(optax.adam(0.5), 1.0)
    current, state = params, updates.init(opt, params)
    for _ in range(2):
        current, state = updates.apply(opt, current, grad, state)
    assert jax.tree_util.tree_structure(current) == \
        jax.tree_util.tree_structure(params)


def test_the_estimate_radius_may_be_traced(params, grad):
    def run(current, estimate, state, radius):
        opt = projection.l1_projected_estimate(updates.sgd(1.0), radius)
        return updates.apply(opt, current, estimate, state)

    state = updates.init(updates.sgd(1.0), params)
    eager, _ = run(params, grad, state, 2.0)
    jitted, _ = jax.jit(run)(params, grad, state, jnp.float32(2.0))
    assert tree_allclose(eager, jitted)


def test_zero_radius_zeroes_the_estimate(params, grad):
    """A zero ball projects every estimate to the origin, so the rule
    descends along nothing and the parameters stay put."""
    opt = projection.l1_projected_estimate(updates.sgd(0.1), 0.0)
    out, _ = updates.apply(opt, params, grad, updates.init(opt, params))
    assert tree_allclose(out, params)


def test_the_estimate_wrapper_rejects_a_negative_concrete_radius():
    with pytest.raises(ValueError, match="radius"):
        projection.l1_projected_estimate(updates.sgd(0.1), -1.0)


def test_the_estimate_wrapper_needs_no_params(params, grad):
    """Projecting the estimate reads nothing from the parameters, so a
    bare ``update(estimate, state)`` call works — unlike `l1_projected`,
    which raises."""
    opt = projection.l1_projected_estimate(updates.sgd(1.0), 1.0)
    increment, _ = opt.update(grad, updates.init(opt, params))
    assert tree_allclose(
        increment,
        jax.tree.map(
            lambda leaf: -leaf, geometry.project_l1_ball_pytree(grad, 1.0)
        ),
        atol=1e-7,
    )
