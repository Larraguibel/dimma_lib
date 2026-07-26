"""Stage 6 - perturbation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from dimma.core import noise

from tests.helpers import tree_allclose, tree_equal

ADDERS = [noise.add_gaussian, noise.add_laplace]


@pytest.fixture
def big_leaf():
    """One large leaf, so the statistical assertions are not flaky."""
    return {"w": jnp.zeros((200_000,))}


@pytest.mark.parametrize("add", ADDERS)
def test_structure_shapes_and_dtypes_survive(add, params, key):
    out = add(params, key, 1.0)
    assert jax.tree_util.tree_structure(out) == \
        jax.tree_util.tree_structure(params)
    for noisy, clean in zip(
        jax.tree_util.tree_leaves(out), jax.tree_util.tree_leaves(params)
    ):
        assert noisy.shape == clean.shape
        assert noisy.dtype == clean.dtype


@pytest.mark.parametrize("add", ADDERS)
def test_zero_scale_leaves_the_input_untouched(add, params, key):
    assert tree_equal(add(params, key, 0.0), params)


@pytest.mark.parametrize("add", ADDERS)
def test_same_key_gives_the_same_draw(add, params, key):
    assert tree_equal(add(params, key, 1.0), add(params, key, 1.0))


@pytest.mark.parametrize("add", ADDERS)
def test_different_keys_give_different_draws(add, params):
    a = add(params, jax.random.key(0), 1.0)
    b = add(params, jax.random.key(1), 1.0)
    assert not tree_equal(a, b)


@pytest.mark.parametrize("add", ADDERS)
def test_leaves_are_perturbed_independently(add, key):
    """Each leaf draws from its own split, not a shared one."""
    tree = {"a": jnp.zeros((64,)), "b": jnp.zeros((64,))}
    out = add(tree, key, 1.0)
    assert not jnp.allclose(out["a"], out["b"])


@pytest.mark.parametrize("add", ADDERS)
def test_scale_may_be_traced(add, params, key):
    """Close, not bit-equal: XLA fuses the multiply-add under `jit`.

    Bit-identity holds between two jitted runs of the same program, which
    is what an algorithm's reproducibility rests on, not between a jitted
    and an eager one.
    """
    jitted = jax.jit(add)
    assert tree_allclose(jitted(params, key, jnp.float32(2.0)),
                         add(params, key, 2.0), rtol=1e-6)


@pytest.mark.parametrize("add", ADDERS)
def test_jitted_draws_are_bit_reproducible(add, params, key):
    """Two runs of the same compiled program agree exactly."""
    jitted = jax.jit(add)
    assert tree_equal(jitted(params, key, jnp.float32(2.0)),
                      jitted(params, key, jnp.float32(2.0)))


def test_gaussian_scale_is_a_standard_deviation(big_leaf, key):
    std = 3.0
    out = noise.add_gaussian(big_leaf, key, std)
    assert jnp.isclose(jnp.std(out["w"]), std, rtol=0.02)
    assert jnp.isclose(jnp.mean(out["w"]), 0.0, atol=0.05)


def test_laplace_scale_is_b_not_a_standard_deviation(big_leaf, key):
    """The documented trap: variance is 2b^2, so std is b*sqrt(2)."""
    b = 3.0
    out = noise.add_laplace(big_leaf, key, b)
    assert jnp.isclose(jnp.std(out["w"]), b * jnp.sqrt(2.0), rtol=0.02)
    assert not jnp.isclose(jnp.std(out["w"]), b, rtol=0.02)


def test_noise_is_added_to_the_input_not_replacing_it(key):
    tree = {"w": jnp.full((100_000,), 5.0)}
    out = noise.add_gaussian(tree, key, 0.1)
    assert jnp.isclose(jnp.mean(out["w"]), 5.0, atol=0.01)
