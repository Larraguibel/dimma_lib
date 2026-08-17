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
