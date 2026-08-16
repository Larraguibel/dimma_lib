"""The identity, and the reading it is there to support.

Two kinds of claim. That the four terms reconstruct the score is
arithmetic and is held to floating point. That each term moves when and
only when the thing it names moves is the reason to compute them at all,
and is what the rest of this file pins.
"""

from __future__ import annotations

import numpy as np
import pytest

from dimma.metrics.decomposition import (
    brier_decomposition,
    log_loss_decomposition,
)
from dimma.metrics.scoring import brier_score, log_loss

DECOMPOSITIONS = [
    (brier_decomposition, brier_score),
    (log_loss_decomposition, log_loss),
]


@pytest.mark.parametrize("decompose,score", DECOMPOSITIONS)
@pytest.mark.parametrize("n_bins", [1, 5, 15, 100])
def test_the_terms_reconstruct_the_score(decompose, score, n_bins, calibrated):
    """``calibration - resolution + uncertainty + residual`` is the score.

    Exact, at every bin count, for both scores. A decomposition that
    only approximately adds up would let a term absorb an error and
    still look like a term.
    """
    probs, labels = calibrated
    parts = decompose(probs, labels, n_bins=n_bins)
    assert parts.total == pytest.approx(score(probs, labels), abs=1e-12)


@pytest.mark.parametrize("decompose,_score", DECOMPOSITIONS)
def test_a_model_that_knows_only_the_base_rate_has_no_resolution(
    decompose, _score, uninformative
):
    probs, labels = uninformative
    assert decompose(probs, labels, n_bins=10).resolution == pytest.approx(
        0.0, abs=1e-12
    )


@pytest.mark.parametrize("decompose,_score", DECOMPOSITIONS)
def test_a_model_that_separates_records_has_resolution(
    decompose, _score, calibrated
):
    probs, labels = calibrated
    assert decompose(probs, labels, n_bins=10).resolution > 0.01


def test_uncertainty_is_the_label_variance_and_nothing_else(calibrated):
    probs, labels = calibrated
    rate = float(labels.mean())
    brier = brier_decomposition(probs, labels, n_bins=10)
    assert brier.uncertainty == pytest.approx(rate * (1 - rate), abs=1e-4)


def test_uncertainty_does_not_depend_on_the_model(calibrated):
    """Same split, same term — which is what lets two runs be compared."""
    probs, labels = calibrated
    mine = brier_decomposition(probs, labels, n_bins=10).uncertainty
    theirs = brier_decomposition(
        np.full_like(probs, 0.5), labels, n_bins=10
    ).uncertainty
    assert mine == pytest.approx(theirs, abs=1e-12)


@pytest.mark.parametrize("decompose,_score", DECOMPOSITIONS)
def test_a_calibrated_model_pays_almost_nothing_in_calibration(
    decompose, _score, calibrated
):
    probs, labels = calibrated
    parts = decompose(probs, labels, n_bins=10)
    assert parts.calibration < 0.02 * parts.resolution


@pytest.mark.parametrize("decompose,_score", DECOMPOSITIONS)
def test_inflating_probabilities_costs_calibration_and_not_resolution(
    decompose, _score, calibrated
):
    """The reading the module exists for, as an assertion.

    A strictly increasing transform leaves the equal-mass bins holding
    exactly the same records, so every observed rate is unchanged and
    the resolution term cannot move. Only the stated probabilities
    moved, and only the calibration term reports it. A run that
    degrades this way has kept its signal and is quotable as such.
    """
    probs, labels = calibrated
    before = decompose(probs, labels, n_bins=10)
    after = decompose(np.clip(probs * 1.5, 0.0, 1.0), labels, n_bins=10)

    assert after.resolution == pytest.approx(before.resolution, abs=1e-12)
    assert after.calibration > 20 * before.calibration


@pytest.mark.parametrize("decompose,_score", DECOMPOSITIONS)
def test_destroying_signal_costs_resolution(decompose, _score, calibrated, rng):
    """The other failure, which has to land in the other term.

    Shuffling the predictions against the labels leaves the marginal
    distribution of the predictions untouched — so the model is still
    right on average — and removes every bit of signal in them.
    """
    probs, labels = calibrated
    before = decompose(probs, labels, n_bins=10)
    after = decompose(rng.permutation(probs), labels, n_bins=10)

    assert after.resolution < 0.01 * before.resolution


@pytest.mark.parametrize("decompose,_score", DECOMPOSITIONS)
def test_one_bin_hides_all_the_resolution_in_the_residual(
    decompose, _score, calibrated
):
    """A negative residual is the partition's fault, and says so.

    With a single bin there is nowhere for signal to be credited: the
    resolution term is zero by construction and the predictions are
    still informative, so the whole of it lands in the residual with a
    negative sign. That is the diagnostic the docstring promises.
    """
    probs, labels = calibrated
    parts = decompose(probs, labels, n_bins=1)
    assert parts.resolution == pytest.approx(0.0, abs=1e-12)
    assert parts.residual < 0.0


@pytest.mark.parametrize("decompose,_score", DECOMPOSITIONS)
def test_more_bins_move_signal_out_of_the_residual(
    decompose, _score, calibrated
):
    probs, labels = calibrated
    coarse = decompose(probs, labels, n_bins=2)
    fine = decompose(probs, labels, n_bins=200)
    assert abs(fine.residual) < abs(coarse.residual)
    assert fine.resolution > coarse.resolution
