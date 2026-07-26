"""Poisson-subsampled Gaussian accounting.

The wrapped library is Google's and is not re-tested here. These pin the
wrapper: the parameterisation, the monotonicities a privacy bound must
have, and the claims the two function names make.
"""

from __future__ import annotations

import pytest

from dimma.accounting.sampling import (
    poisson_gaussian_epsilon,
    poisson_gaussian_truncated_epsilon,
)

# A realistic DP-SGD run: q = 1e-3, sigma = 1.1, T = 10k, delta = 1e-5.
RUN = dict(sampling_probability=1e-3, noise_multiplier=1.1,
           num_compositions=10_000, target_delta=1e-5)


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
