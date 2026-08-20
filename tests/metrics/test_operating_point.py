"""The F1-maximising cut, the counts at a cut, and what ties do to both."""

from __future__ import annotations

import numpy as np
import pytest

from dimma.metrics.operating_point import best_f1_threshold, confusion_at


def f1_of(confusion):
    """F1 from the four counts, as a caller would read it off the table."""
    denominator = (
        2 * confusion.true_positives
        + confusion.false_positives
        + confusion.false_negatives
    )
    if denominator == 0:
        return 0.0
    return 2 * confusion.true_positives / denominator


def slow_best_f1(probs, labels):
    """Brute force: every distinct observed score, tried as a threshold.

    Walks the candidates descending so that a tie in F1 resolves to the
    same cut the fast path's `argmax` takes, which is the first one it
    meets in the score-sorted order.
    """
    best_threshold, best_f1 = float("nan"), -1.0
    for candidate in np.unique(probs)[::-1]:
        f1 = f1_of(confusion_at(probs, labels, float(candidate)))
        if f1 > best_f1:
            best_threshold, best_f1 = float(candidate), f1
    return best_threshold, best_f1


class TestBestF1Threshold:
    def test_the_threshold_is_the_exact_maximiser_over_the_observed_cuts(
        self, rng
    ):
        """Against brute force, not against an approximation of it."""
        probs = rng.random(40)
        labels = (rng.random(40) < probs).astype(np.float64)
        point = best_f1_threshold(probs, labels)
        reference_threshold, reference_f1 = slow_best_f1(probs, labels)
        assert point.threshold == pytest.approx(reference_threshold, rel=1e-12)
        assert point.f1 == pytest.approx(reference_f1, rel=1e-12)

    def test_it_never_scores_below_the_naive_half(self, calibrated):
        probs, labels = calibrated
        point = best_f1_threshold(probs, labels)
        achieved = f1_of(confusion_at(probs, labels, point.threshold))
        assert point.f1 == pytest.approx(achieved, rel=1e-9)
        assert achieved >= f1_of(confusion_at(probs, labels, 0.5))

    def test_a_constant_model_returns_the_one_score_it_states(
        self, uninformative
    ):
        """Which predicts every record positive, the cut degenerating.

        There is one candidate cut, the model's single probability, and
        an inclusive threshold at it admits the whole set. F1 collapses
        to the value predicting positive everywhere reaches.
        """
        probs, labels = uninformative
        rate = float(labels.mean())
        point = best_f1_threshold(probs, labels)
        assert point.threshold == probs[0]

        confusion = confusion_at(probs, labels, point.threshold)
        assert confusion.true_negatives == 0
        assert confusion.false_negatives == 0
        assert confusion.true_positives + confusion.false_positives == (
            probs.size
        )
        assert f1_of(confusion) == pytest.approx(2 * rate / (1 + rate))

    def test_labels_with_no_positive_have_no_cut_to_choose(self):
        with pytest.raises(ValueError, match="no positive"):
            best_f1_threshold([0.2, 0.7, 0.9], [0.0, 0.0, 0.0])

    def test_logits_are_refused_rather_than_cut(self):
        """The shared coercion, reached through this module too."""
        with pytest.raises(ValueError, match="probabilities"):
            best_f1_threshold([-3.0, 2.5], [0.0, 1.0])


class TestConfusionAt:
    @pytest.mark.parametrize("threshold", [0.0, 0.1, 0.25, 0.5, 0.9, 1.0])
    def test_the_counts_sum_to_the_number_of_records(
        self, calibrated, threshold
    ):
        probs, labels = calibrated
        confusion = confusion_at(probs, labels, threshold)
        assert sum(confusion) == probs.size

    def test_a_threshold_below_every_score_predicts_all_positive(self):
        confusion = confusion_at([0.2, 0.6, 0.9], [1.0, 0.0, 1.0], 0.1)
        assert confusion.true_positives == 2
        assert confusion.false_positives == 1
        assert confusion.true_negatives == 0
        assert confusion.false_negatives == 0

    def test_a_threshold_above_every_score_predicts_none(self):
        confusion = confusion_at([0.2, 0.6, 0.9], [1.0, 0.0, 1.0], 0.95)
        assert confusion.true_positives == 0
        assert confusion.false_positives == 0
        assert confusion.true_negatives == 1
        assert confusion.false_negatives == 2

    def test_a_record_exactly_at_the_threshold_is_predicted_positive(self):
        """The cut is ``>=``; a strict ``>`` would report the other row."""
        confusion = confusion_at([0.4, 0.4], [1.0, 0.0], 0.4)
        assert confusion.true_positives == 1
        assert confusion.false_positives == 1
        assert confusion.false_negatives == 0

    def test_bad_input_is_named(self):
        with pytest.raises(ValueError, match="same length"):
            confusion_at([0.5, 0.5], [1.0], 0.5)


class TestNotebook02Parity:
    """Values computed from notebook 02's own code, hardcoded here.

    The notebook is not imported or executed; these are the numbers its
    `best_f1_threshold` and `confusion_at` produce on the fixed inputs
    below, so a change to the rule shows up as a failure here.
    """

    PROBS = [0.90, 0.80, 0.80, 0.60, 0.40, 0.40, 0.35, 0.10]
    LABELS = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

    def test_the_pinned_threshold_and_f1(self):
        point = best_f1_threshold(self.PROBS, self.LABELS)
        assert point.threshold == 0.35
        assert point.f1 == pytest.approx(0.7272727272727273, rel=1e-12)

    @pytest.mark.parametrize(
        "threshold,expected",
        [
            (0.0, (0, 4, 0, 4)),
            (0.35, (1, 3, 0, 4)),
            (0.40, (1, 3, 1, 3)),
            (0.50, (2, 2, 2, 2)),
            (0.80, (3, 1, 2, 2)),
            (0.90, (4, 0, 3, 1)),
            (1.00, (4, 0, 4, 0)),
        ],
    )
    def test_the_pinned_confusion_counts(self, threshold, expected):
        confusion = confusion_at(self.PROBS, self.LABELS, threshold)
        assert tuple(confusion) == expected
        assert confusion.true_negatives == expected[0]
        assert confusion.true_positives == expected[3]

    def test_a_cut_inside_a_tied_run_reports_an_F1_no_threshold_reaches(self):
        """The notebook's behaviour on ties, pinned rather than fixed.

        Every score here is tied but one, so the prefix maximising F1
        ends half-way through the tied run: it counts two of the four
        tied records. The threshold that prefix names admits all four,
        which is a worse table than the reported F1 of 1.0 describes.
        """
        probs = [0.6, 0.6, 0.6, 0.6, 0.1]
        labels = [1.0, 1.0, 0.0, 0.0, 0.0]
        point = best_f1_threshold(probs, labels)
        assert point.threshold == 0.6
        assert point.f1 == pytest.approx(1.0, rel=1e-12)

        confusion = confusion_at(probs, labels, point.threshold)
        assert tuple(confusion) == (1, 2, 0, 2)
        assert f1_of(confusion) == pytest.approx(2 / 3)
