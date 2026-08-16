"""Binning, the reliability curve, and what each calibration number misses."""

from __future__ import annotations

import numpy as np
import pytest

from dimma.metrics._binning import bin_predictions
from dimma.metrics.calibration import (
    calibration_ratio,
    expected_calibration_error,
    reliability_curve,
)


class TestBinning:
    def test_every_record_lands_in_exactly_one_bin(self, calibrated):
        probs, labels = calibrated
        bins = bin_predictions(probs, labels, n_bins=15)
        assert bins.count.sum() == probs.size
        assert bins.index.shape == probs.shape
        assert bins.index.max() == bins.count.size - 1

    def test_equal_mass_fills_its_bins_evenly(self, calibrated):
        probs, labels = calibrated
        bins = bin_predictions(probs, labels, n_bins=20, strategy="equal_mass")
        assert bins.count.size == 20
        assert bins.count.max() / bins.count.min() < 1.05

    def test_equal_width_does_not(self, calibrated):
        """Which is the reason equal-mass is the default."""
        probs, labels = calibrated
        bins = bin_predictions(probs, labels, n_bins=20, strategy="equal_width")
        assert bins.count.max() / bins.count.min() > 10

    def test_empty_bins_are_dropped(self):
        probs = np.array([0.01, 0.02, 0.98, 0.99])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        bins = bin_predictions(probs, labels, n_bins=10, strategy="equal_width")
        assert np.all(bins.count > 0)
        assert bins.count.size == 2

    def test_ties_stay_in_one_bin(self):
        """A repeated prediction is one statement; it gets one bin."""
        probs = np.full(1000, 0.3)
        labels = (np.arange(1000) % 4 == 0).astype(np.float64)
        bins = bin_predictions(probs, labels, n_bins=10, strategy="equal_mass")
        assert bins.count.size == 1
        assert bins.mean_observed[0] == pytest.approx(0.25)

    def test_the_edges_are_refused(self, calibrated):
        probs, labels = calibrated
        with pytest.raises(ValueError, match="n_bins"):
            bin_predictions(probs, labels, n_bins=0)
        with pytest.raises(ValueError, match="strategy"):
            bin_predictions(probs, labels, strategy="quantile")


class TestReliabilityCurve:
    def test_a_calibrated_model_sits_on_the_diagonal(self, calibrated):
        probs, labels = calibrated
        curve = reliability_curve(probs, labels, n_bins=15)
        assert np.max(np.abs(curve.gap)) < 0.01

    def test_over_prediction_shows_as_a_positive_gap(self, calibrated):
        probs, labels = calibrated
        curve = reliability_curve(np.clip(probs + 0.1, 0, 1), labels, n_bins=15)
        assert np.all(curve.gap > 0.0)

    def test_the_bins_are_ordered_and_cover_the_predictions(self, calibrated):
        probs, labels = calibrated
        curve = reliability_curve(probs, labels, n_bins=15)
        assert np.all(np.diff(curve.mean_predicted) > 0)
        assert curve.lower[0] <= probs.min()
        assert curve.upper[-1] >= probs.max()


class TestExpectedCalibrationError:
    def test_a_calibrated_model_scores_near_zero(self, calibrated):
        probs, labels = calibrated
        assert expected_calibration_error(probs, labels, n_bins=15) < 0.005

    def test_a_shifted_model_scores_the_shift(self, calibrated):
        probs, labels = calibrated
        shifted = np.clip(probs + 0.1, 0.0, 1.0)
        assert expected_calibration_error(shifted, labels) == pytest.approx(
            0.1, abs=0.02
        )

    def test_more_bins_never_scores_a_calibrated_model_better(self, calibrated):
        """The documented bias: noise per bin can only add to the gap."""
        probs, labels = calibrated
        coarse = expected_calibration_error(probs, labels, n_bins=5)
        fine = expected_calibration_error(probs, labels, n_bins=500)
        assert fine > coarse


class TestCalibrationRatio:
    def test_a_calibrated_model_scores_one(self, calibrated):
        probs, labels = calibrated
        assert calibration_ratio(probs, labels) == pytest.approx(1.0, abs=0.01)

    def test_it_reads_as_the_factor_the_bid_is_wrong_by(self, calibrated):
        probs, labels = calibrated
        doubled = np.clip(probs * 2.0, 0.0, 1.0)
        assert calibration_ratio(doubled, labels) == pytest.approx(0.5, abs=0.02)

    def test_it_is_blind_to_errors_that_cancel(self, calibrated):
        """Why the curve is here and not only this ratio.

        Push the low half up and the high half down by the same total
        mass: every bin is now wrong, the aggregate is untouched, and
        only the binned numbers notice.
        """
        probs, labels = calibrated
        median = np.median(probs)
        low, high = probs < median, probs >= median
        broken = probs.copy()
        broken[low] += 0.05
        broken[high] -= 0.05
        assert calibration_ratio(broken, labels) == pytest.approx(1.0, abs=0.01)
        assert expected_calibration_error(broken, labels, n_bins=15) > 0.02
        assert expected_calibration_error(probs, labels, n_bins=15) < 0.005

    def test_nothing_to_divide_by_is_refused(self):
        with pytest.raises(ValueError, match="sum to zero"):
            calibration_ratio([0.0, 0.0], [1.0, 0.0])
