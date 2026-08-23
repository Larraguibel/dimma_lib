"""Whether the stated probabilities are the rates that occurred.

A ranking score asks whether the model put the right records at the top.
These ask a different question: of the records it called 0.30, did about
30% convert? Nothing in a ranking score answers that, because scaling
every prediction leaves the order alone, and a model whose order is
perfect but whose numbers are twice too large is a working ranker and a
broken bidder.

The distinction has a name — discrimination against calibration — and it
is the one that matters for a private run. Clipping and noise act on the
gradient, and a gradient that has been clipped is biased toward the
majority class, so the failure mode a private model reaches first is
usually a preserved order with shifted probabilities. That is invisible
to every ranking metric and is exactly what these report.

`calibration_ratio` is the cheap aggregate, `reliability_curve` the full
picture, and `expected_calibration_error` the curve summarized back to
one number for a sweep. The ratio can sit at 1.0 over a curve that is
wrong everywhere, since over- and under-prediction cancel in the total;
that is the reason all three are here rather than only the first.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from dimma.metrics._binning import Bins, Strategy, bin_predictions
from dimma.metrics._inputs import as_probabilities_and_labels


class ReliabilityCurve(NamedTuple):
    """A reliability diagram as numbers: what was said, what happened.

    Plot `mean_predicted` against `mean_observed`; the diagonal is
    perfect calibration, points below it are over-prediction, points
    above it under-prediction. Weight anything read off it by `count` —
    the extreme bins are where a private run's damage shows first and
    also where the observed rate is estimated from the fewest records.

    Attributes
    ----------
    mean_predicted : np.ndarray of shape (k,), float in [0, 1]
        Mean predicted probability in each bin.
    mean_observed : np.ndarray of shape (k,), float in [0, 1]
        Fraction of each bin that was actually positive.
    count : np.ndarray of shape (k,), int
        Records per bin. Never zero: empty bins are dropped.
    lower : np.ndarray of shape (k,), float
        Left edge of each bin.
    upper : np.ndarray of shape (k,), float
        Right edge of each bin.
    """

    mean_predicted: np.ndarray
    mean_observed: np.ndarray
    count: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @property
    def gap(self) -> np.ndarray:
        """``(k,)`` signed ``predicted - observed``, positive when over."""
        return self.mean_predicted - self.mean_observed


def reliability_curve(
    probs: object,
    labels: object,
    n_bins: int = 15,
    strategy: Strategy = "equal_mass",
) -> ReliabilityCurve:
    """Bin the predictions and report what each bin actually did.

    Parameters
    ----------
    probs : array-like of shape (n,)
        Predicted probabilities of the positive class, in ``[0, 1]``.
    labels : array-like of shape (n,)
        Binary labels in ``{0, 1}``.
    n_bins : int, default 15
        Requested bin count; the result may hold fewer. It is a
        smoothing parameter and `dimma.metrics._binning` records what it
        trades.
    strategy : {"equal_mass", "equal_width"}, default "equal_mass"
        ``"equal_mass"`` cuts at quantiles, ``"equal_width"`` cuts
        ``[0, 1]`` evenly. Equal-width is the classic diagram and the
        one to take when matching a published figure.

    Returns
    -------
    ReliabilityCurve
        One point per surviving bin: what was predicted there, what was
        observed, how many records, and the bin's edges.

    Raises
    ------
    ValueError
        If ``n_bins`` is below 1, if ``strategy`` is neither name, or
        for any of the input problems `dimma.metrics._inputs` validates.
    """
    bins = bin_predictions(probs, labels, n_bins=n_bins, strategy=strategy)
    return ReliabilityCurve(
        mean_predicted=bins.mean_predicted,
        mean_observed=bins.mean_observed,
        count=bins.count,
        lower=bins.lower,
        upper=bins.upper,
    )


def expected_calibration_error(
    probs: object,
    labels: object,
    n_bins: int = 15,
    strategy: Strategy = "equal_mass",
) -> float:
    """Return the mean absolute calibration gap, weighted by bin occupancy.

    ``sum_k (n_k / n) * |predicted_k - observed_k|`` — the reliability
    curve's distance from the diagonal, collapsed to one number so a
    sweep can be ordered by it. Reads in the units of the thing being
    predicted: 0.02 means the stated probabilities are off by about two
    percentage points where the data actually is.

    This is an estimate with a known bias, and the bias runs one way.
    Each observed rate carries its own sampling error, the absolute
    value cannot cancel it, and so a perfectly calibrated model scores
    above zero and scores worse the more bins it is given. Compare ECEs
    only at equal ``n_bins``, equal ``strategy``, and comparable sample
    size, and treat a small difference between two of them as noise
    rather than as a result — a paired comparison of the same evaluation
    set under two models is the reading this supports.

    Parameters
    ----------
    probs : array-like of shape (n,)
        Predicted probabilities of the positive class, in ``[0, 1]``.
    labels : array-like of shape (n,)
        Binary labels in ``{0, 1}``.
    n_bins : int, default 15
        Requested bin count; the result may use fewer.
    strategy : {"equal_mass", "equal_width"}, default "equal_mass"
        Which partition to take.

    Returns
    -------
    float
        In ``[0, 1]``. Lower is better.

    Raises
    ------
    ValueError
        If ``n_bins`` is below 1, if ``strategy`` is neither name, or
        for any of the input problems `dimma.metrics._inputs` validates.
    """
    bins: Bins = bin_predictions(
        probs, labels, n_bins=n_bins, strategy=strategy
    )
    gap = np.abs(bins.mean_predicted - bins.mean_observed)
    return float(np.sum(bins.weight * gap))


def calibration_ratio(probs: object, labels: object) -> float:
    """Return observed positives over predicted ones. 1.0 is calibrated.

    Reads directly: 0.9 means the model claims about 11% more clicks than
    happened, and a bid built on it overpays by roughly that much.

    One ratio over the whole set, so it detects a uniform shift and
    nothing finer — the module docstring says what that misses and which
    of these three catches it. Cheap enough to watch every epoch; not
    enough on its own.

    Parameters
    ----------
    probs : array-like of shape (n,)
        Predicted probabilities of the positive class, in ``[0, 1]``.
    labels : array-like of shape (n,)
        Binary labels in ``{0, 1}``.

    Returns
    -------
    float
        ``>= 0``. Below 1.0 the model over-predicts, above it
        under-predicts, and 1.0 is calibrated in aggregate.

    Raises
    ------
    ValueError
        If the predicted probabilities sum to zero, leaving nothing to
        divide by, or for any of the input problems
        `dimma.metrics._inputs` validates.

    References
    ----------
    .. [1] He et al., "Practical Lessons from Predicting Clicks on Ads at
       Facebook", ADKDD 2014.
    """
    p, y = as_probabilities_and_labels(probs, labels)
    expected = float(np.sum(p))
    if expected == 0.0:
        raise ValueError(
            "Predicted probabilities sum to zero, so there is no expected "
            "count to compare the observed one against."
        )
    return float(np.sum(y)) / expected
