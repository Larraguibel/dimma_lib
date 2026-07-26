"""Stage 7 - optimization, dimma's seam onto optax."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

from dimma.core import updates

from tests.helpers import tree_allclose


@pytest.fixture
def grad(params):
    return jax.tree.map(lambda leaf: jnp.full_like(leaf, 0.5), params)


def test_apply_returns_new_params_and_new_state(params, grad):
    opt = optax.sgd(0.1)
    state = updates.init(opt, params)
    new_params, new_state = updates.apply(opt, params, grad, state)
    assert jax.tree_util.tree_structure(new_params) == \
        jax.tree_util.tree_structure(params)
    assert new_state is not None


def test_sgd_reproduces_the_hand_rolled_update(params, grad):
    """The claim the optax migration rests on: `p - lr * g`, bit-identical.

    If this ever fails, ported SpiderBoost results stop matching the
    original implementation.
    """
    lr = 0.1
    opt = optax.sgd(lr)
    state = updates.init(opt, params)
    new_params, _ = updates.apply(opt, params, grad, state)
    expected = jax.tree.map(lambda p, g: p - lr * g, params, grad)
    leaves_a = jax.tree_util.tree_leaves(new_params)
    leaves_b = jax.tree_util.tree_leaves(expected)
    assert all(jnp.array_equal(x, y) for x, y in zip(leaves_a, leaves_b))


def test_state_threads_through_a_loop(params, grad):
    """Momentum only accumulates if the caller carries the state forward."""
    opt = optax.sgd(0.1, momentum=0.9)
    state = updates.init(opt, params)
    first, state = updates.apply(opt, params, grad, state)
    second, state = updates.apply(opt, first, grad, state)
    step_one = jax.tree_util.tree_leaves(
        jax.tree.map(lambda a, b: a - b, params, first)
    )[0]
    step_two = jax.tree_util.tree_leaves(
        jax.tree.map(lambda a, b: a - b, first, second)
    )[0]
    assert not jnp.allclose(step_one, step_two)


def test_schedules_advance_per_apply_call_not_per_epoch(params, grad):
    """Steps, not epochs — the privacy horizon and the schedule share a count."""
    schedule = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=3
    )
    opt = optax.sgd(schedule)
    state = updates.init(opt, params)

    sizes = []
    current = params
    for _ in range(3):
        nxt, state = updates.apply(opt, current, grad, state)
        delta = jax.tree.map(lambda a, b: a - b, current, nxt)
        sizes.append(float(jnp.abs(jax.tree_util.tree_leaves(delta)[0]).sum()))
        current = nxt

    assert sizes[0] > sizes[1] > sizes[2]


def test_adam_runs_through_the_same_seam(params, grad):
    """Baselines and private methods share one optimizer implementation."""
    opt = optax.adam(1e-3)
    state = updates.init(opt, params)
    new_params, _ = updates.apply(opt, params, grad, state)
    assert not tree_allclose(new_params, params)


def test_apply_is_jittable(params, grad):
    opt = optax.sgd(0.1)
    state = updates.init(opt, params)
    jitted = jax.jit(updates.apply, static_argnums=0)
    a, _ = jitted(opt, params, grad, state)
    b, _ = updates.apply(opt, params, grad, state)
    assert tree_allclose(a, b)


def test_optax_is_not_re_exported():
    """A deliberate boundary: callers name their optimizer via optax itself.

    Proxying optax would make dimma's public surface change with every
    optax release, invisibly.
    """
    assert not hasattr(updates, "adam")
    assert not hasattr(updates, "cosine_decay_schedule")
    assert set(updates.__all__) == {
        "GradientTransformation",
        "OptState",
        "Schedule",
        "apply_updates",
        "init",
        "apply",
    }
