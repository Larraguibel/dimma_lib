"""Lipschitz and smoothness constants for logistic regression.

Expected numbers worked by hand from ADR-0012's closed forms rather
than recomputed the way the code computes them.
"""

from __future__ import annotations

import math

import pytest

from dimma.accounting.lipschitz import logreg_bce_constants


def test_the_unit_ball_with_a_bias():
    """R = 1, augmented norm sqrt(1 + 1): L0 = sqrt(2), L1 = 2/4,
    step = 1/(2 * 0.5)."""
    constants = logreg_bce_constants(1.0, has_bias=True)
    assert constants.lipschitz_constant == pytest.approx(math.sqrt(2.0))
    assert constants.smoothness_constant == pytest.approx(0.5)
    assert constants.step_size == pytest.approx(1.0)


def test_the_unit_ball_without_a_bias():
    """The flag is worth a factor of two in L1 and in the step size."""
    constants = logreg_bce_constants(1.0, has_bias=False)
    assert constants.lipschitz_constant == pytest.approx(1.0)
    assert constants.smoothness_constant == pytest.approx(0.25)
    assert constants.step_size == pytest.approx(2.0)


def test_a_wider_bound_with_a_bias():
    """R = 2: augmented norm sqrt(5), L1 = 5/4, step = 1/2.5."""
    constants = logreg_bce_constants(2.0, has_bias=True)
    assert constants.lipschitz_constant == pytest.approx(math.sqrt(5.0))
    assert constants.smoothness_constant == pytest.approx(1.25)
    assert constants.step_size == pytest.approx(0.4)


# --------------------------------------------------------------------
# The relations the triple must satisfy to be one triple
# --------------------------------------------------------------------


@pytest.mark.parametrize("bound", [0.25, 1.0, 3.0, 40.0])
@pytest.mark.parametrize("has_bias", [True, False])
def test_smoothness_is_a_quarter_of_the_lipschitz_constant_squared(
    bound, has_bias
):
    """`sup sigma(1 - sigma) = 1/4` against `sup |sigma - y| = 1`."""
    constants = logreg_bce_constants(bound, has_bias=has_bias)
    assert constants.smoothness_constant == pytest.approx(
        constants.lipschitz_constant**2 / 4.0
    )


@pytest.mark.parametrize("bound", [0.25, 1.0, 3.0, 40.0])
@pytest.mark.parametrize("has_bias", [True, False])
def test_the_step_size_is_theorem_b3s(bound, has_bias):
    """Returned together because B.3 fixes it, not as a convenience."""
    constants = logreg_bce_constants(bound, has_bias=has_bias)
    assert constants.step_size == pytest.approx(
        1.0 / (2.0 * constants.smoothness_constant)
    )


def test_the_bias_flag_is_worth_two_in_the_smoothness_at_the_unit_ball():
    """Silently defaulting it would halve or double every noise scale."""
    with_bias = logreg_bce_constants(1.0, has_bias=True)
    without = logreg_bce_constants(1.0, has_bias=False)
    assert with_bias.smoothness_constant == pytest.approx(
        2.0 * without.smoothness_constant
    )
    assert with_bias.step_size == pytest.approx(without.step_size / 2.0)


def test_the_bias_flag_is_not_worth_two_in_the_lipschitz_constant():
    """The two constants do not move together: L0 goes up by sqrt(2)
    where L1 goes up by 2. Pinned because it is an easy thing to say
    loosely and a false thing to have said."""
    with_bias = logreg_bce_constants(1.0, has_bias=True)
    without = logreg_bce_constants(1.0, has_bias=False)
    assert with_bias.lipschitz_constant == pytest.approx(
        math.sqrt(2.0) * without.lipschitz_constant
    )


def test_the_bias_flag_matters_less_as_the_bound_widens():
    """Nor is the factor of two a constant of the flag: the augmenting
    1 is a smaller share of R^2 + 1 the larger R is."""
    narrow = logreg_bce_constants(1.0, has_bias=True).smoothness_constant / (
        logreg_bce_constants(1.0, has_bias=False).smoothness_constant)
    wide = logreg_bce_constants(10.0, has_bias=True).smoothness_constant / (
        logreg_bce_constants(10.0, has_bias=False).smoothness_constant)
    assert narrow == pytest.approx(2.0)
    assert wide < 1.02


def test_the_bias_flag_has_to_be_said():
    with pytest.raises(TypeError):
        logreg_bce_constants(1.0)


def test_the_bias_flag_cannot_be_passed_positionally():
    """A positional bool next to a float is the transposition ADR-0006
    keeps hyperparameters keyword-only to prevent."""
    with pytest.raises(TypeError):
        logreg_bce_constants(1.0, True)


# --------------------------------------------------------------------
# What a wider bound costs
# --------------------------------------------------------------------


def test_doubling_the_bound_more_than_doubles_the_smoothness_constant():
    """`R` is not a free knob: L1 grows like R^2, and the variation
    branch's noise grows with it."""
    narrow = logreg_bce_constants(4.0, has_bias=True)
    wide = logreg_bce_constants(8.0, has_bias=True)
    assert wide.smoothness_constant > 2.0 * narrow.smoothness_constant


def test_doubling_the_bound_more_than_halves_the_step_size():
    """And the run gets slower at the same time as it gets noisier."""
    narrow = logreg_bce_constants(4.0, has_bias=True)
    wide = logreg_bce_constants(8.0, has_bias=True)
    assert wide.step_size < narrow.step_size / 2.0


# --------------------------------------------------------------------
# There is no route from the data to a constant
# --------------------------------------------------------------------


def test_the_function_takes_a_bound_and_a_flag_and_nothing_else():
    """ADR-0012's enforcement: a contributor who thinks measuring
    `max ||x||` would be convenient has nowhere to put it."""
    import inspect

    parameters = inspect.signature(logreg_bce_constants).parameters
    assert list(parameters) == ["feature_norm_bound", "has_bias"]


def test_the_module_has_no_array_library_to_measure_with():
    from dimma.accounting import lipschitz

    assert not hasattr(lipschitz, "np")
    assert not hasattr(lipschitz, "jnp")
    assert not hasattr(lipschitz, "jax")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_bound_that_bounds_nothing_is_rejected(bad):
    with pytest.raises(ValueError, match="feature_norm_bound"):
        logreg_bce_constants(bad, has_bias=True)


# --------------------------------------------------------------------
# What the constants are for
# --------------------------------------------------------------------

# ADR-0010's worked configuration, as `test_spiderboost` uses it.
RUN = dict(steps=2000, anchor_interval=20, anchor_expected_batch_size=2000,
           variation_expected_batch_size=500, dataset_size=100_000)


def test_the_two_constants_are_the_accountants_keywords():
    """The docstring says keyword-for-keyword, so pin it: a rename on
    either side should fail here rather than at a call site."""
    import inspect

    from dimma.accounting import spiderboost

    parameters = set(inspect.signature(spiderboost.noise_scales).parameters)
    assert {"lipschitz_constant", "smoothness_constant"} <= parameters
    assert "step_size" not in parameters


def test_the_triple_calibrates_a_run():
    constants = logreg_bce_constants(1.0, has_bias=True)

    from dimma.accounting import spiderboost

    scales = spiderboost.noise_scales(
        lipschitz_constant=constants.lipschitz_constant,
        smoothness_constant=constants.smoothness_constant,
        target_epsilon=1.0, target_delta=1e-6, **RUN,
    )
    assert all(scale > 0.0 for scale in scales)


def test_a_wider_bound_buys_more_noise_at_the_same_budget():
    """The cost of raising R, at the place it is actually paid."""
    from dimma.accounting import spiderboost

    def rate(bound):
        constants = logreg_bce_constants(bound, has_bias=True)
        return spiderboost.noise_scales(
            lipschitz_constant=constants.lipschitz_constant,
            smoothness_constant=constants.smoothness_constant,
            target_epsilon=1.0, target_delta=1e-6, **RUN,
        ).variation_noise_rate

    assert rate(2.0) > rate(1.0)
