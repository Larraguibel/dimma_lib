"""Grouping predictions, which every calibration number is built on.

A reliability curve, an expected calibration error, and both
decompositions in `dimma.metrics.decomposition` are one operation read
four ways: partition the predictions, then compare the mean prediction
in each group against the rate actually observed there. The partition is
written once here, and which partition to take is a parameter rather
than a constant, because it is the choice that moves the answer.

Equal-mass is the default. Predicted click probabilities pile up around
the base rate and thin out fast on both sides, so equal-width bins spend
most of their bins on stretches holding almost nothing and drop most of
the data into two or three. Equal-mass bins hold the same count each,
which also estimates every bin's observed rate to about the same
precision — and the observed rate is the quantity being compared.

Bins are a smoothing parameter, not a detail. Too few and a model that
is badly calibrated within a bin looks calibrated, because the gap
averages out inside the bin and never reaches the number. Too many and
each observed rate is a mean over so few records that its own sampling
noise dominates, which pushes every gap-based number upward whether the
model deserves it or not. Report the count alongside anything computed
from these.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

from dimma.metrics._inputs import as_probabilities_and_labels

Strategy = Literal["equal_mass", "equal_width"]


class Bins(NamedTuple):
    """A partition of the predictions, with per-bin statistics.

    Empty bins are dropped, so ``count`` never holds a zero and the
    arrays may be shorter than the ``n_bins`` asked for. Equal-mass
    binning drops them when ties collapse quantile edges; equal-width
    binning drops them wherever the predictions never went.
    """

    index: np.ndarray
    """``(n,)`` bin each record fell in, indexing the arrays below."""
    count: np.ndarray
    """``(k,)`` records per bin."""
    mean_predicted: np.ndarray
    """``(k,)`` mean predicted probability in each bin."""
    mean_observed: np.ndarray
    """``(k,)`` fraction of each bin that was actually positive."""
    lower: np.ndarray
    """``(k,)`` left edge of each bin."""
    upper: np.ndarray
    """``(k,)`` right edge of each bin."""

    @property
    def weight(self) -> np.ndarray:
        """``(k,)`` share of the records in each bin, summing to 1."""
        return self.count / self.count.sum()


def _edges(probs: np.ndarray, n_bins: int, strategy: Strategy) -> np.ndarray:
    if strategy == "equal_width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "equal_mass":
        # Duplicate edges mean a quantile range collapsed onto a single
        # repeated prediction. Dropping them keeps equal predictions in
        # one bin, which is what makes a bin's observed rate a statement
        # about a prediction rather than about a sort order.
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(probs, quantiles))
        if edges.size < 2:
            edges = np.array([edges[0], np.nextafter(edges[0], 1.0)])
        return edges
    raise ValueError(
        f"Unknown strategy {strategy!r}. Expected 'equal_mass' or "
        f"'equal_width'."
    )


def bin_predictions(
    probs: object,
    labels: object,
    n_bins: int = 15,
    strategy: Strategy = "equal_mass",
) -> Bins:
    """Partition predictions into bins and summarize each.

    Parameters
    ----------
    probs
        Predicted probabilities of the positive class, in ``[0, 1]``.
    labels
        Binary labels in ``{0, 1}``.
    n_bins : int, default 15
        Requested bin count. The result may hold fewer, never more; see
        `Bins`.
    strategy : {"equal_mass", "equal_width"}, default "equal_mass"
        ``"equal_mass"`` cuts at the quantiles of ``probs``, so each bin
        holds about the same number of records. ``"equal_width"`` cuts
        ``[0, 1]`` into equal intervals, which is the classic reliability
        diagram and is what to use when the comparison is against a
        published number that took it.

    Returns
    -------
    Bins

    Raises
    ------
    ValueError
        If ``n_bins`` is below 1, if ``strategy`` is neither name, or
        for any of the input problems `dimma.metrics` validates.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}.")

    p, y = as_probabilities_and_labels(probs, labels)
    edges = _edges(p, n_bins, strategy)

    # ``side="right"`` puts a value equal to an interior edge in the bin
    # above it; the clip folds the two outer half-open ends back in, so
    # 0.0 and 1.0 are binned rather than dropped.
    raw = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, edges.size - 2)

    n_raw = edges.size - 1
    count = np.bincount(raw, minlength=n_raw)
    kept = np.flatnonzero(count)
    total_predicted = np.bincount(raw, weights=p, minlength=n_raw)[kept]
    total_observed = np.bincount(raw, weights=y, minlength=n_raw)[kept]
    count = count[kept]

    # Renumber onto the bins that survived, so ``index`` addresses the
    # returned arrays rather than the ones that were asked for.
    remap = np.empty(n_raw, dtype=np.intp)
    remap[kept] = np.arange(kept.size)

    return Bins(
        index=remap[raw],
        count=count,
        mean_predicted=total_predicted / count,
        mean_observed=total_observed / count,
        lower=edges[kept],
        upper=edges[kept + 1],
    )
