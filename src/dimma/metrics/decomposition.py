"""Splitting a proper score into calibration and discrimination.

This is the answer to the question a ranking score forces and cannot
settle. PR-AUC and ROC-AUC measure discrimination only; the calibration
numbers next door measure calibration only; and choosing between them
looks like a choice about which half of the model to care about. It is
not, because a strictly proper score already contains both, and the
containment is an identity rather than an analogy::

    score = calibration - resolution + uncertainty + residual

Read left to right. **Calibration** is what the stated probabilities cost
by not being the observed rates, and is zero for a model that is right
on average in every bin. **Resolution** is how far the bins pull apart
from the base rate — the discrimination term, the one a ranking score is
a monotone-invariant proxy for, and the only term that helps rather than
hurts. **Uncertainty** is the entropy or variance of the base rate: a
property of the evaluation set, identical for every model scored on it,
and the reason two decompositions from different splits do not compare.
**Residual** is what the binning discarded.

Why this earns its place in a private run. Both terms move under DP-SGD
and they move for unrelated reasons. Noise added to the gradient
degrades what the model can tell apart, which shows up as resolution
falling. Clipping biases the update toward the majority class, which
shifts the probabilities without necessarily disturbing their order, and
shows up as calibration rising. A single number — private log loss
against non-private log loss — reports the sum of the two and so cannot
say which happened, while a ranking score reports a proxy for one of
them and is blind to the other by construction. The decomposition says
which, and that is a claim about the mechanism rather than about the
score.

What the partition does to the split. Under equal-mass binning the bins
are cut at quantiles, so any strictly increasing transform of the
predictions leaves every bin holding exactly the same records and every
observed rate untouched: **resolution is then rank-based, and is
precisely the part a ranking score can see**, with calibration carrying
everything a ranking score is invariant to. That is the cleanest reading
and the reason equal-mass is the default — the two terms land exactly on
"what PR-AUC measures" and "what PR-AUC cannot". Equal-width bins are
fixed intervals, so a monotone transform moves records across them and
some of the same distortion appears in resolution instead. Neither is
wrong; they answer slightly different questions, and mixing them within
one comparison is what would be.

Both decompositions here are exact against their **binned** score, and
`residual` carries the gap to the real one, so the identity closes to
floating-point precision and `tests/metrics/test_decomposition.py` holds
it to that. A residual near zero means the bins are fine enough that
nothing was lost. A **negative** residual means predictions and labels
still move together inside the bins — real resolution the partition
threw away — and asks for more bins rather than fewer.
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
    """

    calibration: float
    """Squared calibration gap, occupancy-weighted. Lower is better."""
    resolution: float
    """Spread of bin rates about the base rate. **Higher** is better."""
    uncertainty: float
    """``base_rate * (1 - base_rate)``. Fixed by the evaluation set."""
    residual: float
    """Real score minus binned score. See the module docstring."""

    @property
    def total(self) -> float:
        """The Brier score the four terms reconstruct."""
        return (
            self.calibration - self.resolution + self.uncertainty + self.residual
        )


class LogLossDecomposition(NamedTuple):
    """The same split in nats, against log loss rather than Brier.

    ``log_loss == calibration - resolution + uncertainty + residual``.
    """

    calibration: float
    """Mean KL from observed rate to stated probability. Lower is better."""
    resolution: float
    """Entropy the bins removed from the base rate. **Higher** is better."""
    uncertainty: float
    """Binary entropy of the base rate, in nats. Fixed by the set."""
    residual: float
    """Real score minus binned score. See the module docstring."""

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
    """Split `dimma.metrics.scoring.brier_score` into its three terms.

    Parameters
    ----------
    probs
        Predicted probabilities of the positive class, in ``[0, 1]``.
    labels
        Binary labels in ``{0, 1}``.
    n_bins : int, default 15
        Requested bin count; the result may use fewer.
    strategy : {"equal_mass", "equal_width"}, default "equal_mass"
        Which partition to take.

    Returns
    -------
    BrierDecomposition
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
    """Entropy of each Bernoulli rate, in nats, with ``0 log 0 = 0``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(rate > 0.0, rate * np.log(rate), 0.0) + np.where(
            rate < 1.0, (1.0 - rate) * np.log1p(-rate), 0.0
        )
    return -terms


def _binary_kl(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """KL from the observed rate to the stated probability, in nats.

    ``predicted`` is clipped away from 0 and 1 exactly as
    `dimma.metrics.scoring.log_loss` clips, so a bin that stated
    certainty and was wrong contributes a large number rather than an
    infinite one, and the two sides of the identity are clipped alike.
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
    """Split `dimma.metrics.scoring.log_loss` into its three terms.

    The information-theoretic reading of the same split Murphy's
    decomposition gives Brier, and the one whose uncertainty term is the
    denominator of `dimma.metrics.scoring.normalized_entropy` — so
    ``(calibration - resolution) / uncertainty`` is what normalized
    entropy has left after the base rate is divided out, up to the
    residual.

    Parameters
    ----------
    probs
        Predicted probabilities of the positive class, in ``[0, 1]``.
    labels
        Binary labels in ``{0, 1}``.
    n_bins : int, default 15
        Requested bin count; the result may use fewer.
    strategy : {"equal_mass", "equal_width"}, default "equal_mass"
        Which partition to take.

    Returns
    -------
    LogLossDecomposition

    Raises
    ------
    ValueError
        If the labels are constant, leaving no uncertainty to split.
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
