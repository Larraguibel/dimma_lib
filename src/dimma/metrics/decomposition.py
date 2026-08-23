"""Split a proper score into calibration and discrimination::

    score = calibration - resolution + uncertainty + residual

**Calibration** is what the stated probabilities cost by not being the
observed rates. **Resolution** is how far the bins pull apart from the
base rate — the discrimination term a ranking score is a
monotone-invariant proxy for, and the only one that helps rather than
hurts. **Uncertainty** is a property of the evaluation set, identical
for every model scored on it, which is why decompositions from different
splits do not compare. **Residual** is what the binning discarded.

The split is the diagnostic no single number gives under DP-SGD: noise
degrades what the model can tell apart and shows up as resolution
falling, while clipping shifts the probabilities without necessarily
disturbing their order and shows up as calibration rising. A log-loss
gap reports the sum of the two, and a ranking score sees one and is
blind to the other.

Under equal-mass binning the bins are cut at quantiles, so a strictly
increasing transform of the predictions leaves every bin holding the
same records: resolution is then precisely the part a ranking score can
see. Equal-width bins are fixed intervals, so some of that distortion
lands in resolution instead; mixing the two within one comparison is the
mistake. Both decompositions are exact against their **binned** score
and ``residual`` carries the gap to the real one — near zero means the
bins were fine enough, and **negative** means predictions and labels
still move together inside them, which asks for more bins rather than
fewer.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from dimma.metrics._binning import Strategy, bin_predictions
from dimma.metrics._inputs import base_rate_entropy
from dimma.metrics.scoring import brier_score, log_loss

_CLIP = 1e-15


class BrierDecomposition(NamedTuple):
    """Murphy's three terms, plus the gap the binning opened.

    ``brier_score == calibration - resolution + uncertainty + residual``.

    Murphy calls the first term *reliability*; it is named ``calibration``
    here so that this and `LogLossDecomposition` share one vocabulary and
    a caller can read either without translating. The two differ in
    units, not in what they name.

    Attributes
    ----------
    calibration : float >= 0
        Squared calibration gap, occupancy-weighted. Lower is better.
    resolution : float >= 0
        Spread of bin rates about the base rate. **Higher** is better.
    uncertainty : float
        ``base_rate * (1 - base_rate)``. Fixed by the evaluation set.
    residual : float
        Real score minus binned score. See the module docstring.
    """

    calibration: float
    resolution: float
    uncertainty: float
    residual: float

    @property
    def total(self) -> float:
        """The Brier score the four terms reconstruct."""
        return (
            self.calibration - self.resolution + self.uncertainty + self.residual
        )


class LogLossDecomposition(NamedTuple):
    """The same split in nats, against log loss rather than Brier.

    ``log_loss == calibration - resolution + uncertainty + residual``.

    Attributes
    ----------
    calibration : float >= 0, in nats
        Mean KL from observed rate to stated probability. Lower is
        better.
    resolution : float, in nats
        Entropy the bins removed from the base rate. **Higher** is
        better.
    uncertainty : float, in nats
        Binary entropy of the base rate. Fixed by the evaluation set.
    residual : float, in nats
        Real score minus binned score. See the module docstring.
    """

    calibration: float
    resolution: float
    uncertainty: float
    residual: float

    @property
    def total(self) -> float:
        """The log loss the four terms reconstruct."""
        return (
            self.calibration - self.resolution + self.uncertainty + self.residual
        )


def brier_decomposition(
    probs: object,
    labels: object,
    n_bins: int = 15,
    strategy: Strategy = "equal_mass",
) -> BrierDecomposition:
    """Split the Brier score into Murphy's three terms and a residual.

    The score split is `dimma.metrics.scoring.brier_score`, and the
    residual is the gap the binning opened between it and the three
    terms taken over the bins.

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
    BrierDecomposition
        The three terms in squared-probability units plus the residual,
        which sum back to the Brier score.

    Raises
    ------
    ValueError
        If ``n_bins`` is below 1, if ``strategy`` is neither name, or
        for any of the input problems `dimma.metrics._inputs` validates.
    """
    bins = bin_predictions(probs, labels, n_bins=n_bins, strategy=strategy)
    weight = bins.weight
    base_rate = float(np.sum(weight * bins.mean_observed))

    calibration = float(
        np.sum(weight * (bins.mean_predicted - bins.mean_observed) ** 2)
    )
    resolution = float(np.sum(weight * (bins.mean_observed - base_rate) ** 2))
    uncertainty = base_rate * (1.0 - base_rate)

    binned = calibration - resolution + uncertainty
    residual = brier_score(probs, labels) - binned
    return BrierDecomposition(
        calibration=calibration,
        resolution=resolution,
        uncertainty=uncertainty,
        residual=residual,
    )


def _binary_entropy(rate: np.ndarray) -> np.ndarray:
    """Return each Bernoulli rate's entropy, in nats, with ``0 log 0 = 0``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(rate > 0.0, rate * np.log(rate), 0.0) + np.where(
            rate < 1.0, (1.0 - rate) * np.log1p(-rate), 0.0
        )
    return -terms


def _binary_kl(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Return the KL from observed rate to stated probability, in nats.

    ``predicted`` is clipped exactly as
    `dimma.metrics.scoring.log_loss` clips, so both sides of the
    identity are clipped alike.
    """
    p = np.clip(predicted, _CLIP, 1.0 - _CLIP)
    cross = -(
        np.where(observed > 0.0, observed * np.log(p), 0.0)
        + np.where(observed < 1.0, (1.0 - observed) * np.log1p(-p), 0.0)
    )
    return cross - _binary_entropy(observed)


def log_loss_decomposition(
    probs: object,
    labels: object,
    n_bins: int = 15,
    strategy: Strategy = "equal_mass",
) -> LogLossDecomposition:
    """Split the log loss into three terms and a residual, in nats.

    The score split is `dimma.metrics.scoring.log_loss`, and the residual
    is the gap the binning opened between it and the three terms taken
    over the bins. This is the information-theoretic reading of the split
    Murphy's decomposition gives Brier, and the one whose uncertainty
    term is `dimma.metrics.scoring.normalized_entropy`'s denominator — so
    ``(calibration - resolution) / uncertainty`` is what normalized
    entropy has left after the base rate is divided out, up to the
    residual.

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
    LogLossDecomposition
        The three terms in nats plus the residual, which sum back to the
        log loss.

    Raises
    ------
    ValueError
        If the labels are constant, leaving no uncertainty to split; if
        ``n_bins`` is below 1 or ``strategy`` is neither name; or for any
        of the input problems `dimma.metrics._inputs` validates.
    """
    bins = bin_predictions(probs, labels, n_bins=n_bins, strategy=strategy)
    weight = bins.weight
    base_rate = float(np.sum(weight * bins.mean_observed))

    uncertainty = base_rate_entropy(base_rate)
    within = float(np.sum(weight * _binary_entropy(bins.mean_observed)))
    resolution = uncertainty - within
    calibration = float(
        np.sum(weight * _binary_kl(bins.mean_observed, bins.mean_predicted))
    )

    binned = calibration - resolution + uncertainty
    residual = log_loss(probs, labels) - binned
    return LogLossDecomposition(
        calibration=calibration,
        resolution=resolution,
        uncertainty=uncertainty,
        residual=residual,
    )
