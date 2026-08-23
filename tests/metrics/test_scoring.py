"""The proper scores, and the property that makes them proper."""

from __future__ import annotations

import numpy as np
import pytest

from dimma.metrics.scoring import brier_score, log_loss, normalized_entropy


def test_perfect_predictions_score_zero():
    y = np.array([0.0, 1.0, 1.0, 0.0])
    assert log_loss(y, y) == pytest.approx(0.0, abs=1e-12)
    assert brier_score(y, y) == pytest.approx(0.0, abs=1e-12)


def test_base_rate_predictor_scores_the_base_rate_entropy(uninformative):
    """The anchor every other log loss is read against."""
    probs, labels = uninformative
    rate = float(labels.mean())
    expected = -(rate * np.log(rate) + (1 - rate) * np.log1p(-rate))
    assert log_loss(probs, labels) == pytest.approx(expected, rel=1e-12)
    assert brier_score(probs, labels) == pytest.approx(
        rate * (1 - rate), rel=1e-12
    )


def test_normalized_entropy_is_one_for_a_model_that_learned_nothing(
    uninformative,
):
    probs, labels = uninformative
    assert normalized_entropy(probs, labels) == pytest.approx(1.0, rel=1e-12)


def test_normalized_entropy_is_below_one_for_a_model_that_learned(calibrated):
    probs, labels = calibrated
    assert 0.0 < normalized_entropy(probs, labels) < 1.0


@pytest.mark.parametrize("score", [log_loss, brier_score])
def test_the_scores_are_proper(score, calibrated):
    """Reporting the true probability beats reporting anything else.

    This is the whole reason to select on one of these. A shift and a
    sharpening move the predictions away from the truth while reordering
    nothing — the clips only tie records at a bound — so a ranking
    metric would call these variants all but identical; a proper score
    has to prefer the truth.
    """
    probs, labels = calibrated
    truth = score(probs, labels)
    for distorted in (
        np.clip(probs + 0.05, 0.0, 1.0),
        np.clip(probs - 0.05, 0.0, 1.0),
        np.clip(probs * 1.5, 0.0, 1.0),
        probs**0.5,
    ):
        assert score(distorted, labels) > truth


def test_a_monotone_transform_that_ranking_ignores_is_penalized(calibrated):
    """The failure a private run reaches first, and AUC's blind spot.

    Doubling is increasing below 0.5 and ties the 6% above it at 1.0, so
    the order check spot-checks the hundred smallest, where no tie
    reaches.
    """
    probs, labels = calibrated
    scaled = np.clip(probs * 2.0, 0.0, 1.0)
    order = np.argsort(probs)
    assert np.array_equal(np.argsort(scaled, kind="stable")[:100], order[:100])
    assert log_loss(scaled, labels) > log_loss(probs, labels)


def test_certainty_that_was_wrong_is_clipped_rather_than_infinite():
    """A floor on how bad one record may look, not a fact about the model.

    ``1 - 1e-15`` is not itself a float64: floats there are 1.11e-16
    apart, so the nearest one sits 9.99e-16 below 1.0 and the loss comes
    out at 34.5396 rather than ``-log(1e-15)``'s 34.5388. The bound is
    written as the expression the code evaluates for that reason.
    """
    assert np.isfinite(log_loss([1.0, 0.0], [0.0, 1.0]))
    assert log_loss([1.0], [0.0]) == pytest.approx(
        -np.log1p(-(1.0 - 1e-15)), rel=1e-12
    )
    assert log_loss([1.0], [0.0]) < 35.0


def test_logits_are_refused_rather_than_scored():
    with pytest.raises(ValueError, match="probabilities"):
        log_loss([-3.0, 2.5], [0.0, 1.0])


def test_constant_labels_have_no_entropy_to_normalize_against():
    with pytest.raises(ValueError, match="constant"):
        normalized_entropy([0.3, 0.4], [1.0, 1.0])


@pytest.mark.parametrize(
    "probs,labels,match",
    [
        ([0.5, 0.5], [1.0], "same length"),
        ([[0.5]], [[1.0]], "one-dimensional"),
        ([], [], "empty"),
        ([np.nan, 0.5], [1.0, 0.0], "non-finite"),
        ([0.5, 0.5], [0.0, 2.0], "labels must be"),
    ],
)
def test_bad_input_is_named(probs, labels, match):
    with pytest.raises(ValueError, match=match):
        log_loss(probs, labels)
