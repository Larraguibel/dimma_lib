"""The score-sorted walk both threshold-sweeping metrics are built on.

`dimma.metrics.ranking` and `dimma.metrics.operating_point` ask
different questions of the same walk — sort by predicted probability
descending, then take precision and recall at every prefix — and it is
written once here so the two cannot drift apart on it.

Tie order is why. Runs of equal scores are ordered by NumPy's default
sort, which is deterministic for a given input, is not stable, and is
not input order; both the curve and the cut chosen from it depend on
where the positives land inside such a run, so a model emitting few
distinct probabilities is the case to watch. Two spellings of
``np.argsort`` would be two tie orders. This module is the one home for
that argument, and the two callers cite it rather than restate it.

Degenerate input is the caller's to refuse: labels holding no positive
leave recall undefined, and each public caller raises with the message
its own question calls for.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from dimma.metrics._inputs import as_probabilities_and_labels


class RankedScores(NamedTuple):
    """One descending pass over the records, and the curve along it.

    Every array is in descending score order and as long as the input,
    so index ``i`` throughout describes the cut that admits the top
    ``i + 1`` records.

    Attributes
    ----------
    scores : np.ndarray of shape (n,), float in [0, 1]
        The predicted probabilities, sorted descending.
    hits : np.ndarray of shape (n,), float in {0.0, 1.0}
        The labels in that same order, 1.0 where positive.
    found : np.ndarray of shape (n,), float
        Positives at or above each cut, cumulative.
    precision : np.ndarray of shape (n,), float in [0, 1]
        ``found`` over the number of records at or above the cut.
    recall : np.ndarray of shape (n,), float in [0, 1]
        ``found`` over all positives; zeros when there are none.
    positives : float
        How many positives the labels hold. Zero is degenerate; see
        above.
    """

    scores: np.ndarray
    hits: np.ndarray
    found: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    positives: float


def rank_by_score(probs: object, labels: object) -> RankedScores:
    """Sort by predicted probability descending and walk the prefixes.

    Parameters
    ----------
    probs : array-like of shape (n,)
        Predicted probabilities of the positive class, in ``[0, 1]``.
    labels : array-like of shape (n,)
        Binary labels in ``{0, 1}``.

    Returns
    -------
    RankedScores
        The scores and labels in descending score order, with precision,
        recall and the cumulative positive count at every prefix.

    Raises
    ------
    ValueError
        For any of the input problems `dimma.metrics._inputs` validates.
        Labels with no positive are not one of them here; the caller
        refuses those, with the message its own question calls for.
    """
    p, y = as_probabilities_and_labels(probs, labels)

    # The default (non-stable) sort kind is load-bearing: it is the tie
    # order both callers are pinned to. Do not pass ``kind=``.
    order = np.argsort(-p)
    hits = y[order]
    positives = float(hits.sum())

    found = np.cumsum(hits)
    # The floor only ever applies when there are no positives, where
    # `found` is all zeros and the caller is about to raise regardless;
    # it keeps a 0/0 warning out of a path that never returns.
    return RankedScores(
        scores=p[order],
        hits=hits,
        found=found,
        precision=found / np.arange(1, hits.size + 1),
        recall=found / (positives if positives else 1.0),
        positives=positives,
    )
