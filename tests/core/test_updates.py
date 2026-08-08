"""Stage 7 - optimization, the rule dimma's own algorithms descend with."""

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
    opt = updates.sgd(0.1)
    state = updates.init(opt, params)
    new_params, new_state = updates.apply(opt, params, grad, state)
    assert jax.tree_util.tree_structure(new_params) == \
        jax.tree_util.tree_structure(params)
    assert isinstance(new_state, updates.SgdState)


def test_sgd_is_the_hand_rolled_update(params, grad):
    """Algorithm 1's descent line, bit-identical to writing it out."""
    lr = 0.1
    opt = updates.sgd(lr)
    new_params, _ = updates.apply(opt, params, grad, updates.init(opt, params))
    expected = jax.tree.map(lambda p, g: p - lr * g, params, grad)
    leaves_a = jax.tree_util.tree_leaves(new_params)
    leaves_b = jax.tree_util.tree_leaves(expected)
    assert all(jnp.array_equal(x, y) for x, y in zip(leaves_a, leaves_b))


def test_sgd_agrees_with_optax_sgd_bit_for_bit(params, grad):
    """What the migration off optax rests on.

    If this ever fails, a run recorded before dimma implemented stage 7
    stops being comparable to one recorded after it, and the difference
    between them is no longer the thing under study.
    """
    ours = updates.sgd(0.1)
    theirs = optax.sgd(0.1)
    a, _ = updates.apply(ours, params, grad, updates.init(ours, params))
    b, _ = updates.apply(theirs, params, grad, updates.init(theirs, params))
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    assert all(jnp.array_equal(x, y) for x, y in zip(leaves_a, leaves_b))


def test_apply_adds_the_optimizers_increment(params):
    """The sign lives in the optimizer, which is what optax assumes too."""
    increment = jax.tree.map(lambda leaf: jnp.full_like(leaf, 2.0), params)
    stub = updates.Optimizer(lambda p: None,
                             lambda g, s, p=None: (increment, s))
    got, _ = updates.apply(stub, params, params, None)
    assert tree_allclose(got, jax.tree.map(lambda p: p + 2.0, params))


def test_the_count_advances_once_per_apply(params, grad):
    """Steps, not epochs - the unit privacy composes over."""
    opt = updates.sgd(0.1)
    current, state = params, updates.init(opt, params)
    assert int(state.count) == 0
    for _ in range(4):
        current, state = updates.apply(opt, current, grad, state)
    assert int(state.count) == 4


@pytest.mark.parametrize("schedule", [
    pytest.param(lambda t: 1.0 / (1.0 + t), id="plain_callable"),
    pytest.param(optax.linear_schedule(1.0, 0.0, 3), id="optax"),
])
def test_a_schedule_indexes_on_the_update_count(schedule, params, grad):
    """`eta_t`, with t the count the accountant composes over.

    Any ``count -> rate`` callable satisfies `Schedule`, so optax's
    schedules are usable without dimma reimplementing them.
    """
    opt = updates.sgd(schedule)
    sizes = []
    current, state = params, updates.init(opt, params)
    for _ in range(3):
        nxt, state = updates.apply(opt, current, grad, state)
        delta = jax.tree.map(lambda a, b: a - b, current, nxt)
        sizes.append(float(jnp.abs(jax.tree_util.tree_leaves(delta)[0]).sum()))
        current = nxt
    assert sizes[0] > sizes[1] > sizes[2]


def test_an_optax_optimizer_passes_through_the_same_seam(params, grad):
    """A baseline's Adam and an algorithm's sgd share one stage 7.

    `Optimizer` is optax's signature deliberately. If this fails, the
    baselines can no longer be pinned to the same optimizer as the
    method they are the counterpart of.
    """
    opt = optax.adam(1e-3)
    state = updates.init(opt, params)
    new_params, state = updates.apply(opt, params, grad, state)
    assert not tree_allclose(new_params, params)
    again, _ = updates.apply(opt, new_params, grad, state)
    assert not tree_allclose(again, new_params)


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


def test_apply_is_jittable(params, grad):
    opt = updates.sgd(0.1)
    state = updates.init(opt, params)
    jitted = jax.jit(updates.apply, static_argnums=0)
    a, _ = jitted(opt, params, grad, state)
    b, _ = updates.apply(opt, params, grad, state)
    assert tree_allclose(a, b)


def test_the_state_shape_is_invariant_across_steps(params, grad):
    """What lets a loop thread it through `jax.jit` without retracing."""
    opt = updates.sgd(0.1)
    state = updates.init(opt, params)
    before = jax.tree_util.tree_structure(state)
    _, state = updates.apply(opt, params, grad, state)
    assert jax.tree_util.tree_structure(state) == before


@pytest.mark.parametrize("bad", [0.0, -0.1, jnp.array(-0.1)])
def test_a_non_positive_learning_rate_is_rejected(bad):
    """Whatever its type: only a schedule is exempt, being uninspectable."""
    with pytest.raises(ValueError, match="learning_rate"):
        updates.sgd(bad)


def test_updates_carries_only_the_rule_the_algorithms_state():
    """Momentum, Adam and weight decay are absent on purpose; ADR-0002.

    An unused option at stage 7 is a way for two runs to differ without
    the difference being reported.
    """
    assert not hasattr(updates, "momentum")
    assert set(updates.__all__) == {
        "Optimizer", "OptState", "Schedule", "SgdState",
        "sgd", "init", "apply",
    }
