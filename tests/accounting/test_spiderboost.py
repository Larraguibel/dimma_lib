"""Private SpiderBoost accounting.

The composition and the accountant underneath it are tested in
`test_sampling`; these pin what this module adds. Three things: that
the two schedules it derives are Algorithm 2's, that calibrating and
reporting are inverse, and that the alignment the fixed multiplier
rests on is enforced rather than assumed.
"""

from __future__ import annotations

import math

import pytest

from dimma.accounting import spiderboost
from dimma.accounting.sampling import (
    PoissonGaussianSchedule,
    composed_poisson_gaussian_epsilon,
    poisson_gaussian_epsilon,
)

# ADR-0010's worked configuration, so the numbers here are comparable
# with the ones recorded there.
RUN = dict(steps=2000, anchor_interval=20, anchor_expected_batch_size=2000,
           variation_expected_batch_size=500, dataset_size=100_000)
CONSTANTS = dict(lipschitz_constant=1.0, smoothness_constant=5.0)
DELTA = 1e-6


def scales(target_epsilon=1.0, **overrides):
    return spiderboost.noise_scales(
        **{**CONSTANTS, **RUN, **overrides},
        target_epsilon=target_epsilon, target_delta=DELTA,
    )


def spent(scale_set, **overrides):
    return spiderboost.epsilon(
        **{**CONSTANTS, **RUN, **overrides}, **scale_set._asdict(),
        target_delta=DELTA,
    )


# --- the schedules ---------------------------------------------------

def test_the_branches_split_every_step_between_them():
    """Every step releases, on exactly one branch."""
    anchors, variations = spiderboost.release_counts(2000, 20)
    assert anchors + variations == 2000


def test_step_zero_is_an_anchor_step():
    """``t % anchor_interval`` makes it one, so a 1-step run is 1-0."""
    assert spiderboost.release_counts(2, 1000) == (1, 1)


def test_a_partial_final_phase_still_costs_its_anchor():
    """21 steps at interval 20 anchors twice, not once. Rounding down
    would under-count a release and understate epsilon."""
    assert spiderboost.release_counts(21, 20) == (2, 19)


def test_an_interval_of_one_makes_every_step_an_anchor():
    assert spiderboost.release_counts(50, 1) == (50, 0)


def test_a_branch_that_never_runs_is_priced_as_absent():
    """At interval 1 the variation branch makes no release, so the cost
    is the anchor branch's alone - not an error about a zero count."""
    only_anchors = spiderboost.epsilon(
        **CONSTANTS, **{**RUN, "anchor_interval": 1, "steps": 50},
        anchor_noise_scale=1e-3, variation_noise_rate=5e-3,
        variation_noise_cap=2e-3, target_delta=DELTA,
    )
    assert only_anchors == poisson_gaussian_epsilon(
        sampling_probability=RUN["anchor_expected_batch_size"]
        / RUN["dataset_size"],
        noise_multiplier=1e-3 * RUN["anchor_expected_batch_size"] / 1.0,
        num_compositions=50, target_delta=DELTA,
    )


# --- the two directions are inverse ----------------------------------

@pytest.mark.parametrize("target", [0.5, 1.0, 3.0])
def test_calibrating_then_reporting_returns_the_budget(target):
    """The property that makes the pair trustworthy: what a budget buys
    costs that budget."""
    assert spent(scales(target)) == pytest.approx(target, rel=1e-5)


def test_a_tighter_budget_buys_more_noise():
    assert scales(0.5).anchor_noise_scale > scales(3.0).anchor_noise_scale


def test_the_budget_is_met_rather_than_approached():
    """Calibration must not overshoot: epsilon at the returned scales
    is at or under the target, never over."""
    for target in (0.5, 1.0, 3.0):
        assert spent(scales(target)) <= target * (1 + 1e-9)


# --- the scales the paper's constants imply --------------------------

def test_the_calibrated_scales_sit_at_the_saturation_ratio():
    """``cap / rate == 2 L0 / L1`` is what makes the variation branch a
    fixed-multiplier mechanism; calibration has to produce it."""
    s = scales()
    expected = 2 * CONSTANTS["lipschitz_constant"] \
        / CONSTANTS["smoothness_constant"]
    assert s.variation_noise_cap / s.variation_noise_rate == \
        pytest.approx(expected)


def test_the_three_scales_share_one_multiplier():
    """Recovering the multiplier from each scale gives the same number,
    which is the claim ADR-0010 rests the composition on."""
    s = scales()
    b1 = RUN["anchor_expected_batch_size"]
    b2 = RUN["variation_expected_batch_size"]
    l0, l1 = CONSTANTS["lipschitz_constant"], CONSTANTS["smoothness_constant"]
    from_anchor = s.anchor_noise_scale * b1 / l0
    from_rate = s.variation_noise_rate * b2 / l1
    from_cap = s.variation_noise_cap * b2 / (2 * l0)
    assert from_anchor == pytest.approx(from_rate)
    assert from_anchor == pytest.approx(from_cap)


def test_the_scales_are_named_as_train_takes_them():
    """They are passed straight through, so a rename here is a silent
    mis-parameterisation there."""
    from dimma.algorithms.spiderboost import train

    parameters = train.train.__code__.co_varnames
    assert set(scales()._asdict()) <= set(parameters)


# --- what the composition buys ---------------------------------------

def test_composing_the_branches_beats_summing_them():
    """The reason this module exists rather than two calls to
    `poisson_gaussian_epsilon`. Summing is sound but is basic
    composition; ADR-0010's configuration pays tens of percent for it.
    """
    s = scales()
    b1, b2 = RUN["anchor_expected_batch_size"], \
        RUN["variation_expected_batch_size"]
    n = RUN["dataset_size"]
    anchors, variations = spiderboost.release_counts(
        RUN["steps"], RUN["anchor_interval"])
    multiplier = s.anchor_noise_scale * b1 / CONSTANTS["lipschitz_constant"]

    composed = spent(s)
    summed = (
        poisson_gaussian_epsilon(b1 / n, multiplier, anchors, DELTA / 2)
        + poisson_gaussian_epsilon(b2 / n, multiplier, variations, DELTA / 2)
    )
    assert composed < summed
    assert summed > 1.4 * composed


def test_epsilon_matches_composing_the_two_schedules_by_hand():
    """Guards the mapping from scales to schedules, which is the whole
    of what this module contributes."""
    s = scales()
    b1, b2 = RUN["anchor_expected_batch_size"], \
        RUN["variation_expected_batch_size"]
    n = RUN["dataset_size"]
    anchors, variations = spiderboost.release_counts(
        RUN["steps"], RUN["anchor_interval"])
    multiplier = s.anchor_noise_scale * b1 / CONSTANTS["lipschitz_constant"]

    assert spent(s) == composed_poisson_gaussian_epsilon(
        [PoissonGaussianSchedule(b1 / n, multiplier, anchors),
         PoissonGaussianSchedule(b2 / n, multiplier, variations)],
        DELTA,
    )


def test_the_method_moves_the_answer():
    """Why it is reported alongside the number rather than assumed."""
    as_rdp = scales(1.0)
    as_pld = spiderboost.noise_scales(
        **CONSTANTS, **RUN, target_epsilon=1.0, target_delta=DELTA,
        method="pld")
    assert as_pld.anchor_noise_scale < as_rdp.anchor_noise_scale


# --- misalignment ----------------------------------------------------

def misaligned():
    s = scales()
    return s._replace(variation_noise_rate=s.variation_noise_rate * 0.5)


def test_scales_off_the_ratio_are_refused():
    """The misalignment is invisible in the output, so it cannot be a
    silent number."""
    with pytest.raises(ValueError, match="2 \\* L0 / L1"):
        spent(misaligned())


def test_the_refusal_names_the_sound_multiplier():
    """A caller told only 'no' cannot act on it."""
    with pytest.raises(ValueError, match="accept_misaligned_scales"):
        spent(misaligned())


def test_misalignment_can_be_accepted_but_not_quietly():
    with pytest.warns(UserWarning, match="conservative multiplier"):
        spiderboost.epsilon(**CONSTANTS, **RUN, **misaligned()._asdict(),
                            target_delta=DELTA,
                            accept_misaligned_scales=True)


def test_the_conservative_multiplier_is_the_pessimistic_endpoint():
    """Halving the rate halves one endpoint and leaves the other, so
    the reported epsilon is the one for the noisier-is-not-assumed
    branch: strictly worse than the aligned run it came from."""
    with pytest.warns(UserWarning):
        loose = spiderboost.epsilon(**CONSTANTS, **RUN,
                                    **misaligned()._asdict(),
                                    target_delta=DELTA,
                                    accept_misaligned_scales=True)
    assert loose > spent(scales())


def test_a_cap_raised_alone_does_not_change_the_price():
    """The cap is not the binding endpoint when the rate is unchanged,
    so a caller cannot buy a smaller epsilon by raising it."""
    s = scales()
    generous = s._replace(variation_noise_cap=s.variation_noise_cap * 10)
    with pytest.warns(UserWarning):
        raised = spiderboost.epsilon(**CONSTANTS, **RUN,
                                     **generous._asdict(),
                                     target_delta=DELTA,
                                     accept_misaligned_scales=True)
    assert raised == pytest.approx(spent(s))


# --- validation ------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {"steps": 1},
    {"steps": 0},
    {"anchor_interval": 0},
    {"dataset_size": 0},
    {"anchor_expected_batch_size": 0},
    {"variation_expected_batch_size": -1},
    {"lipschitz_constant": 0.0},
    {"smoothness_constant": -1.0},
])
def test_invalid_parameters_are_rejected_when_calibrating(bad):
    with pytest.raises(ValueError):
        scales(**bad)


def test_a_batch_larger_than_the_dataset_is_rejected():
    """It would mean a sampling rate above 1, which is not a Poisson
    draw at all."""
    with pytest.raises(ValueError, match="sampling rate"):
        scales(anchor_expected_batch_size=RUN["dataset_size"] + 1)


@pytest.mark.parametrize("bad", [
    {"anchor_noise_scale": 0.0},
    {"variation_noise_rate": -1.0},
    {"variation_noise_cap": 0.0},
])
def test_non_positive_scales_are_rejected_when_reporting(bad):
    with pytest.raises(ValueError):
        spent(scales()._replace(**bad))


def test_an_unreachable_budget_is_refused_rather_than_approximated():
    """A budget no multiplier reaches must not come back as a very
    large scale that quietly misses it."""
    with pytest.raises(ValueError, match="No noise multiplier up to"):
        scales(target_epsilon=1e-9)


def test_a_budget_too_loose_to_calibrate_is_refused():
    """Above the search range every multiplier fits, so no returned
    number would be the smallest sufficient one."""
    with pytest.raises(ValueError, match="meaningless"):
        scales(target_epsilon=1e8)


def test_reporting_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="method"):
        spiderboost.epsilon(**CONSTANTS, **RUN, **scales()._asdict(),
                            target_delta=DELTA, method="moments")


# --- the constants are premises, not decoration ----------------------

def test_the_constants_rescale_the_noise_and_not_the_budget():
    """The multiplier depends only on the rates, the counts and the
    budget - never on L0 or L1. So assuming a function class twice as
    steep buys exactly twice the noise for the same epsilon, which is
    what makes the constants premises rather than tuning knobs."""
    base, steeper = scales(), scales(lipschitz_constant=2.0)
    assert steeper.anchor_noise_scale == \
        pytest.approx(2 * base.anchor_noise_scale)
    assert steeper.variation_noise_cap == \
        pytest.approx(2 * base.variation_noise_cap)
    assert steeper.variation_noise_rate == \
        pytest.approx(base.variation_noise_rate)


def test_reporting_under_a_constant_the_scales_were_not_built_for_is_refused():
    """The three scales encode L0 and L1 in their ratio, so a caller
    who reports a run under different constants is describing a
    mechanism that did not run. It is caught rather than repriced."""
    for wrong in ({"smoothness_constant": 10.0}, {"lipschitz_constant": 2.0}):
        with pytest.raises(ValueError, match="2 \\* L0 / L1"):
            spent(scales(), **wrong)


def test_epsilon_grows_with_the_number_of_steps():
    """Privacy composes over releases; a longer run cannot be cheaper."""
    s = scales()
    assert spent(s, steps=4000) > spent(s)


def test_epsilon_is_finite_at_the_calibrated_scales():
    assert math.isfinite(spent(scales()))
