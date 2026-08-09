"""Poisson-subsampled Gaussian accounting.

The wrapped library is Google's and is not re-tested here. These pin the
wrapper: the parameterisation, the monotonicities a privacy bound must
have, and the claims the two function names make.
"""

from __future__ import annotations

import logging
import math

import pytest

from dimma.accounting.sampling import (
    PoissonGaussianSchedule,
    calibrate_noise_multiplier,
    composed_poisson_gaussian_epsilon,
    poisson_gaussian_epsilon,
    poisson_gaussian_truncated_epsilon,
)

# A realistic DP-SGD run: q = 1e-3, sigma = 1.1, T = 10k, delta = 1e-5.
RUN = dict(sampling_probability=1e-3, noise_multiplier=1.1,
           num_compositions=10_000, target_delta=1e-5)
# The same run as a schedule, for the composing entry point.
RUN_RELEASE = {k: v for k, v in RUN.items() if k != "target_delta"}


def test_a_realistic_run_lands_in_a_sane_range():
    """Abadi et al. report single-digit epsilon at these settings."""
    assert 0.0 < poisson_gaussian_epsilon(**RUN) < 10.0


def test_more_noise_lowers_epsilon():
    quiet = poisson_gaussian_epsilon(**{**RUN, "noise_multiplier": 0.6})
    loud = poisson_gaussian_epsilon(**{**RUN, "noise_multiplier": 4.0})
    assert loud < quiet


def test_more_steps_raise_epsilon():
    """Privacy composes over optimizer steps; it cannot be free."""
    short = poisson_gaussian_epsilon(**{**RUN, "num_compositions": 100})
    long = poisson_gaussian_epsilon(**{**RUN, "num_compositions": 10_000})
    assert short < long


def test_a_higher_sampling_rate_raises_epsilon():
    """Subsampling amplifies privacy; less of it amplifies less."""
    rare = poisson_gaussian_epsilon(**{**RUN, "sampling_probability": 1e-4})
    often = poisson_gaussian_epsilon(**{**RUN, "sampling_probability": 1e-2})
    assert rare < often


def test_a_looser_delta_lowers_epsilon():
    tight = poisson_gaussian_epsilon(**{**RUN, "target_delta": 1e-9})
    loose = poisson_gaussian_epsilon(**{**RUN, "target_delta": 1e-3})
    assert loose < tight


def test_zero_steps_costs_nothing():
    assert poisson_gaussian_epsilon(**{**RUN, "num_compositions": 0}) == 0.0


def test_pld_is_tighter_than_rdp():
    """Why the method is a documented choice and not an implementation
    detail: the same run reports a materially different epsilon."""
    as_rdp = poisson_gaussian_epsilon(**RUN, method="rdp")
    as_pld = poisson_gaussian_epsilon(**RUN, method="pld")
    assert as_pld < as_rdp
    assert as_pld < 0.8 * as_rdp


def test_the_default_method_is_rdp():
    """The moments accountant's descendant, faithful to the paper."""
    assert poisson_gaussian_epsilon(**RUN) == \
        poisson_gaussian_epsilon(**RUN, method="rdp")


def test_the_truncated_bound_equals_the_standard_one():
    """Same number, weaker claim. The name carries the difference."""
    assert poisson_gaussian_truncated_epsilon(**RUN) == \
        poisson_gaussian_epsilon(**RUN)


def test_the_noise_multiplier_is_not_the_standard_deviation():
    """Passing ``z * C`` instead of ``z`` overstates the guarantee.

    Pins that the argument is the ratio: at C = 2 the mistake reports a
    smaller epsilon than the truth, which is the dangerous direction.
    """
    z, clip = 1.1, 2.0
    honest = poisson_gaussian_epsilon(**{**RUN, "noise_multiplier": z})
    mistaken = poisson_gaussian_epsilon(**{**RUN, "noise_multiplier": z * clip})
    assert mistaken < honest


@pytest.mark.parametrize("bad", [
    {"sampling_probability": 0.0},
    {"sampling_probability": 1.5},
    {"sampling_probability": -0.1},
    {"noise_multiplier": 0.0},
    {"noise_multiplier": -1.0},
    {"num_compositions": -1},
    {"target_delta": 0.0},
    {"target_delta": 1.0},
])
def test_invalid_parameters_are_rejected(bad):
    """An accountant that silently accepts nonsense reports nonsense."""
    with pytest.raises(ValueError):
        poisson_gaussian_epsilon(**{**RUN, **bad})


def test_an_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method"):
        poisson_gaussian_epsilon(**RUN, method="moments")


def test_an_unknown_method_is_rejected_before_the_zero_step_shortcut():
    """Validation must not be skippable by an edge-case return path."""
    with pytest.raises(ValueError, match="method"):
        poisson_gaussian_epsilon(**{**RUN, "num_compositions": 0},
                                 method="moments")


def test_epsilon_matches_the_dp_accounting_reference():
    """Guards the wrapper's composition, not the library's math."""
    from dp_accounting import dp_event, rdp

    accountant = rdp.RdpAccountant()
    accountant.compose(
        dp_event.PoissonSampledDpEvent(
            sampling_probability=RUN["sampling_probability"],
            event=dp_event.GaussianDpEvent(
                noise_multiplier=RUN["noise_multiplier"]),
        ),
        count=RUN["num_compositions"],
    )
    expected = accountant.get_epsilon(target_delta=RUN["target_delta"])
    assert poisson_gaussian_epsilon(**RUN) == pytest.approx(expected)


# --- composing more than one schedule --------------------------------

# Two schedules at different rates and counts, as a variance-reduced
# method produces. ADR-0010's configuration.
TWO = [
    PoissonGaussianSchedule(sampling_probability=0.02,
                            noise_multiplier=1.5, num_compositions=100),
    PoissonGaussianSchedule(sampling_probability=0.005,
                            noise_multiplier=1.5, num_compositions=1900),
]


def test_one_schedule_is_the_single_schedule_function():
    """The general form has to agree with the special case exactly, or
    the two entry points describe different mechanisms."""
    assert composed_poisson_gaussian_epsilon(
        [PoissonGaussianSchedule(**RUN_RELEASE)], RUN["target_delta"]
    ) == poisson_gaussian_epsilon(**RUN)


def test_composing_beats_summing_the_schedules_separately():
    """Why this exists. Summing per-schedule epsilons is sound - basic
    composition - but pays for converting to epsilon twice."""
    composed = composed_poisson_gaussian_epsilon(TWO, 1e-6)
    summed = sum(
        poisson_gaussian_epsilon(r.sampling_probability, r.noise_multiplier,
                                 r.num_compositions, 1e-6 / len(TWO))
        for r in TWO
    )
    assert composed < summed


def test_the_order_of_the_schedules_does_not_matter():
    """Each schedule's curve is fixed before the run, so composition is
    adaptive but order-independent."""
    assert composed_poisson_gaussian_epsilon(TWO, 1e-6) == \
        composed_poisson_gaussian_epsilon(list(reversed(TWO)), 1e-6)


def test_a_schedule_that_never_runs_costs_nothing():
    idle = PoissonGaussianSchedule(0.5, 1.0, 0)
    assert composed_poisson_gaussian_epsilon(TWO + [idle], 1e-6) == \
        composed_poisson_gaussian_epsilon(TWO, 1e-6)


def test_no_releases_at_all_costs_nothing():
    assert composed_poisson_gaussian_epsilon([], 1e-6) == 0.0


def test_more_releases_on_either_schedule_raise_epsilon():
    for index in (0, 1):
        longer = list(TWO)
        longer[index] = longer[index]._replace(
            num_compositions=longer[index].num_compositions * 2)
        assert composed_poisson_gaussian_epsilon(longer, 1e-6) > \
            composed_poisson_gaussian_epsilon(TWO, 1e-6)


def test_a_bad_schedule_is_named_by_its_position():
    """With several schedules, 'noise_multiplier must be positive' does
    not say which one."""
    bad = [TWO[0], TWO[1]._replace(noise_multiplier=0.0)]
    with pytest.raises(ValueError, match=r"releases\[1\]"):
        composed_poisson_gaussian_epsilon(bad, 1e-6)


def test_the_single_schedule_function_does_not_name_a_position():
    """Its caller passed bare parameters and never saw a sequence."""
    with pytest.raises(ValueError) as raised:
        poisson_gaussian_epsilon(**{**RUN, "noise_multiplier": 0.0})
    assert "releases[" not in str(raised.value)


def test_a_dropped_renyi_order_is_warned_about():
    """`RdpAccountant` reports these through `absl` logging, where
    `warnings.catch_warnings` cannot see them and a caller who turned
    warnings into errors is unprotected. They must reach the caller's
    channel."""
    with pytest.warns(UserWarning, match="dropped"):
        composed_poisson_gaussian_epsilon(
            [r._replace(noise_multiplier=0.42) for r in TWO], 1e-6)


def test_a_dropped_order_still_returns_a_sound_bound():
    """Epsilon is a minimum over orders, so dropping one raises it."""
    with pytest.warns(UserWarning):
        loose = composed_poisson_gaussian_epsilon(
            [r._replace(noise_multiplier=0.42) for r in TWO], 1e-6)
    assert math.isfinite(loose)
    assert loose > composed_poisson_gaussian_epsilon(TWO, 1e-6)


def test_the_absl_logger_is_left_as_it_was_found():
    """The capture displaces `absl`'s handlers; a caller's logging
    configuration has to survive that."""
    logger = logging.getLogger("absl")
    before = (list(logger.handlers), logger.propagate)
    composed_poisson_gaussian_epsilon(TWO, 1e-6)
    assert (list(logger.handlers), logger.propagate) == before


# --- calibrating -----------------------------------------------------

def one_schedule(multiplier):
    return [PoissonGaussianSchedule(1e-3, multiplier, 10_000)]


def two_schedules(multiplier):
    return [r._replace(noise_multiplier=multiplier) for r in TWO]


@pytest.mark.parametrize("build", [one_schedule, two_schedules])
@pytest.mark.parametrize("target", [0.5, 2.0])
def test_calibrating_hits_the_budget(build, target):
    multiplier = calibrate_noise_multiplier(build, target, 1e-6)
    assert composed_poisson_gaussian_epsilon(build(multiplier), 1e-6) == \
        pytest.approx(target, rel=1e-5)


def test_calibration_meets_the_budget_rather_than_exceeding_it():
    for target in (0.5, 2.0):
        multiplier = calibrate_noise_multiplier(two_schedules, target, 1e-6)
        spent = composed_poisson_gaussian_epsilon(
            two_schedules(multiplier), 1e-6)
        assert spent <= target * (1 + 1e-9)


def test_a_tighter_budget_needs_more_noise():
    assert calibrate_noise_multiplier(two_schedules, 0.5, 1e-6) > \
        calibrate_noise_multiplier(two_schedules, 3.0, 1e-6)


def test_rdp_needs_more_noise_than_pld():
    """ADR-0011's cost, in the direction the ADR states it."""
    as_rdp = calibrate_noise_multiplier(one_schedule, 1.0, 1e-5, method="rdp")
    as_pld = calibrate_noise_multiplier(one_schedule, 1.0, 1e-5, method="pld")
    assert as_rdp > as_pld


def test_calibration_does_not_warn_about_orders_it_only_probed():
    """The search passes through small multipliers where the accountant
    drops orders. Warning from a probe would report a defect the
    returned answer does not have."""
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        calibrate_noise_multiplier(two_schedules, 3.0, 1e-6)
    assert [w for w in caught if "dropped" in str(w.message)] == []


def test_an_unreachable_budget_is_refused():
    """Under RDP epsilon is a minimum over a finite grid of orders, so
    it floors above zero however much noise is added. A target below
    that floor has no answer and must not come back as a large scale
    that quietly misses it."""
    with pytest.raises(ValueError, match="No noise multiplier up to"):
        calibrate_noise_multiplier(two_schedules, 1e-9, 1e-6)


def test_a_budget_met_without_searching_is_refused():
    with pytest.raises(ValueError, match="meaningless"):
        calibrate_noise_multiplier(two_schedules, 1e8, 1e-6)


def test_calibrating_against_no_releases_is_refused():
    with pytest.raises(ValueError, match="no releases"):
        calibrate_noise_multiplier(lambda m: [], 1.0, 1e-6)


@pytest.mark.parametrize("bad", [
    {"target_epsilon": 0.0},
    {"target_epsilon": -1.0},
    {"target_delta": 0.0},
    {"target_delta": 1.0},
])
def test_invalid_calibration_parameters_are_rejected(bad):
    kwargs = {"target_epsilon": 1.0, "target_delta": 1e-6, **bad}
    with pytest.raises(ValueError):
        calibrate_noise_multiplier(two_schedules, **kwargs)


def test_calibration_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="method"):
        calibrate_noise_multiplier(two_schedules, 1.0, 1e-6, method="moments")
