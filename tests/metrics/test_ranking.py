"""The ranking read: what an ordering scores, and what it ignores."""

from __future__ import annotations

import numpy as np
import pytest

from dimma.metrics.ranking import pr_curve


@pytest.fixture
def separable(rng):
    """``(probs, labels)`` whose order is perfect and whose scores are not.

    Positives are drawn above every negative, with no tie anywhere, so
    the ordering is exactly right while the numbers are nowhere near the
    conditional probability they claim.
    """
    labels = (rng.random(20_000) < 0.25).astype(np.float64)
    probs = 0.5 * labels + 0.4 * rng.random(labels.size)
    return probs, labels


class TestAveragePrecision:
    def test_an_uninformative_ranking_scores_the_base_rate(
        self, uninformative
    ):
        """The floor every average precision is read against."""
        probs, labels = uninformative
        curve = pr_curve(probs, labels)
        assert curve.average_precision == pytest.approx(
            float(labels.mean()), abs=0.01
        )

    def test_a_ranking_uncorrelated_with_the_labels_scores_the_same(self, rng):
        """The floor is the base rate whatever the scores look like."""
        labels = (rng.random(200_000) < 0.25).astype(np.float64)
        noise = rng.random(labels.size)
        assert pr_curve(noise, labels).average_precision == pytest.approx(
            float(labels.mean()), abs=0.01
        )

    def test_a_perfect_ranking_scores_one(self, separable):
        probs, labels = separable
        assert pr_curve(probs, labels).average_precision == pytest.approx(
            1.0, abs=1e-12
        )

    def test_an_informative_ranking_scores_between_the_two(self, calibrated):
        probs, labels = calibrated
        curve = pr_curve(probs, labels)
        assert float(labels.mean()) < curve.average_precision < 1.0

    @pytest.mark.parametrize(
        "transform",
        [
            lambda p: p / 2.0,
            lambda p: p**2,
            lambda p: np.sqrt(p),
            lambda p: np.log(p / (1.0 - p)) / 40.0 + 0.5,
        ],
        ids=["halved", "squared", "rooted", "logit"],
    )
    def test_it_is_invariant_under_any_increasing_transform(
        self, transform, calibrated
    ):
        """The property that makes this a ranking read and not a score.

        Every transform here moves the stated probabilities without
        touching their order. Average precision cannot tell them apart,
        which is why `dimma.metrics.scoring` is the one selection runs
        on and this is the one it is reported beside.
        """
        probs, labels = calibrated
        truth = pr_curve(probs, labels).average_precision
        moved = pr_curve(np.clip(transform(probs), 0.0, 1.0), labels)
        assert moved.average_precision == pytest.approx(truth, rel=1e-12)


class TestCurve:
    def test_there_is_one_point_per_record(self, calibrated):
        probs, labels = calibrated
        curve = pr_curve(probs, labels)
        assert curve.precision.shape == probs.shape
        assert curve.recall.shape == probs.shape

    def test_recall_is_monotone_along_the_curve(self, calibrated):
        """Reading down the sorted list can only find more positives."""
        probs, labels = calibrated
        curve = pr_curve(probs, labels)
        assert np.all(np.diff(curve.recall) >= 0.0)

    def test_the_curve_ends_at_full_recall(self, calibrated):
        probs, labels = calibrated
        curve = pr_curve(probs, labels)
        assert curve.recall[-1] == pytest.approx(1.0, abs=1e-12)
        assert curve.precision[-1] == pytest.approx(
            float(labels.mean()), rel=1e-12
        )

    def test_precision_is_not_monotone(self, calibrated):
        """The sawtooth is the curve, not noise in it."""
        probs, labels = calibrated
        curve = pr_curve(probs, labels)
        assert np.any(np.diff(curve.precision) > 0.0)
        assert np.any(np.diff(curve.precision) < 0.0)


class TestTies:
    def test_ties_resolve_deterministically(self):
        """Equal scores get some order; it is the same order every call."""
        probs = np.full(1_000, 0.3)
        probs[::3] = 0.7
        labels = (np.arange(probs.size) % 4 == 0).astype(np.float64)
        first = pr_curve(probs, labels)
        for _ in range(5):
            again = pr_curve(probs, labels)
            assert np.array_equal(again.precision, first.precision)
            assert np.array_equal(again.recall, first.recall)
            assert again.average_precision == first.average_precision

    def test_a_wholly_tied_ranking_still_scores_the_base_rate(self):
        """No information in the scores, so no credit above the floor."""
        labels = (np.arange(1_000) % 4 == 0).astype(np.float64)
        curve = pr_curve(np.full(labels.size, 0.25), labels)
        assert curve.average_precision == pytest.approx(0.25, abs=0.05)


class TestNotebookParity:
    """The promoted function reproduces the notebooks' code exactly.

    The expected numbers below were computed by running the
    DP-SGD-vs-SGD-baseline comparison's `pr_curve` on this input, which
    includes a four-way tie at 0.5 with positives and negatives inside it.
    They therefore pin the tie order NumPy's default (non-stable) sort
    produces as well as the arithmetic.
    """

    PROBS = np.array([0.9, 0.1, 0.5, 0.5, 0.5, 0.8, 0.2, 0.5])
    LABELS = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def test_the_pinned_case_matches_the_notebooks(self):
        curve = pr_curve(self.PROBS, self.LABELS)
        assert curve.precision.tolist() == pytest.approx(
            [
                1.0,
                0.5,
                0.3333333333333333,
                0.5,
                0.4,
                0.5,
                0.42857142857142855,
                0.375,
            ],
            rel=1e-15,
        )
        assert curve.recall.tolist() == pytest.approx(
            [
                0.3333333333333333,
                0.3333333333333333,
                0.3333333333333333,
                0.6666666666666666,
                0.6666666666666666,
                1.0,
                1.0,
                1.0,
            ],
            rel=1e-15,
        )
        assert curve.average_precision == pytest.approx(
            0.6666666666666666, rel=1e-15
        )

    def test_the_fields_unpack_in_the_documented_order(self):
        precision, recall, average_precision = pr_curve(
            self.PROBS, self.LABELS
        )
        curve = pr_curve(self.PROBS, self.LABELS)
        assert np.array_equal(precision, curve.precision)
        assert np.array_equal(recall, curve.recall)
        assert average_precision == curve.average_precision


class TestRefusals:
    def test_labels_with_no_positives_are_refused(self):
        with pytest.raises(ValueError, match="no positives"):
            pr_curve([0.9, 0.2, 0.1], [0.0, 0.0, 0.0])

    def test_the_shared_validation_sits_in_front(self):
        with pytest.raises(ValueError, match="probabilities"):
            pr_curve([-3.0, 2.5], [0.0, 1.0])
        with pytest.raises(ValueError, match="same length"):
            pr_curve([0.5, 0.5], [1.0])
