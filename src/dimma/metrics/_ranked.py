"""The score-sorted walk both threshold-sweeping metrics are built on.

`dimma.metrics.ranking` and `dimma.metrics.operating_point` ask
different questions of the same walk: sort the records by predicted
probability descending, then read precision and recall at every prefix.
`ranking` integrates that curve; `operating_point` maximises F1 along it
and names the score where the maximum sat. The walk is written once
here, so the two cannot drift apart on the one thing they share.

The sort is the reason this is shared rather than duplicated. Ties are
broken by NumPy's default sort, which is deterministic but not stable,
and both the curve and the cut chosen from it depend on where the
positives land inside a run of equal scores. Two spellings of
``np.argsort`` would be two tie orders, and the divergence would show up
only on the models that emit few distinct probabilities — the case
neither caller can afford to get differently from the other.

Degenerate input is the caller's to refuse. Labels holding no positive
leave recall undefined, but what to say about that differs between the
two callers, so this module divides by a floor and returns; each public
function checks `positives` and raises its own error before returning
anything.
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
    """

    scores: np.ndarray
    """``(n,)`` the predicted probabilities, sorted descending."""
    hits: np.ndarray
    """``(n,)`` the labels in that same order, 1.0 where positive."""
    found: np.ndarray
    """``(n,)`` positives at or above each cut, cumulative."""
    precision: np.ndarray
    """``(n,)`` `found` over the number of records at or above the cut."""
    recall: np.ndarray
    """``(n,)`` `found` over all positives; zeros when there are none."""
    positives: float
    """How many positives the labels hold. Zero is degenerate; see above."""


def rank_by_score(probs: object, labels: object) -> RankedScores:
    """Sort by predicted probability descending and walk the prefixes.

    Parameters
    ----------
    probs
        Predicted probabilities of the positive class, in ``[0, 1]``.
    labels
        Binary labels in ``{0, 1}``.

    Returns
    -------
    RankedScores

    Raises
    ------
    ValueError
        For any of the input problems `dimma.metrics` validates. Labels
        with no positive are not one of them here; the caller refuses
        those, with the message its own question calls for.
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
