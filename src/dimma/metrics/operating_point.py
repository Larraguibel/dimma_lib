"""The F1-maximising cut, and the four counts at a cut.

Two functions over probabilities and labels already off the device.
`best_f1_threshold` searches the cuts the data itself offers — the
observed scores, taken descending — and returns the one that maximises
F1 together with the F1 it reached. `confusion_at` applies a cut and
reports the 2x2 table behind it.

The cut is inclusive. A record is predicted positive when its
probability is at or above the threshold, so the returned threshold is
itself predicted positive, and a caller quoting a number from here has
to quote the ``>=`` with it.

Ties
----
The search walks prefixes of the score-sorted order, so a prefix can end
part-way through a run of equal scores. When it does, the F1 reported is
the one that prefix reached and no threshold reproduces it: `confusion_at`
at the returned threshold admits the whole tied run and scores lower.
Distinct scores — what a continuous model on real features produces —
have no such prefix, and the two agree exactly. The order within a tied
run comes from ``np.argsort(-probs)`` at NumPy's default sort, which is
deterministic for a given input but is not input order.

NumPy in float64, like the rest of `dimma.metrics`. Confusion counts come
back as Python ints, and the 2x2 is returned as four named fields rather
than a bare tuple, because four counts in a row are transposed by
accident. Nothing here draws anything.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from dimma.metrics._inputs import as_probabilities_and_labels
from dimma.metrics._ranked import rank_by_score

_F1_FLOOR = 1e-12


class OperatingPoint(NamedTuple):
    """A cut and the F1 it reached.

    `threshold` is one of the observed probabilities, and is inclusive:
    pass it to `confusion_at` to get the table behind `f1`.
    """

    threshold: float
    """The probability at or above which a record is predicted positive."""
    f1: float
    """Harmonic mean of precision and recall at that cut, in ``[0, 1]``."""


class Confusion(NamedTuple):
    """The 2x2 table at one cut, as four counts that name themselves.

    The four sum to the number of records scored. Ordered ``(tn, fp, fn,
    tp)`` when unpacked positionally, which is scikit-learn's ravelled
    order.
    """

    true_negatives: int
    """Predicted negative, actually negative."""
    false_positives: int
    """Predicted positive, actually negative."""
    false_negatives: int
    """Predicted negative, actually positive."""
    true_positives: int
    """Predicted positive, actually positive."""


def best_f1_threshold(probs: object, labels: object) -> OperatingPoint:
    """The cut maximising F1, and the F1 there.

    Every observed score is a candidate, so this is the exact maximiser
    over cuts rather than a search over a grid. Where several cuts reach
    the same F1 the largest threshold among them is returned — the one
    that predicts positive least often.

    Parameters
    ----------
    probs
        Predicted probabilities of the positive class, in ``[0, 1]``.
    labels
        Binary labels in ``{0, 1}``.

    Returns
    -------
    OperatingPoint

    Raises
    ------
    ValueError
        If the labels hold no positive. Recall is then undefined at every
        cut and there is no F1 to maximise.
    """
    ranked = rank_by_score(probs, labels)
    if ranked.positives == 0.0:
        raise ValueError(
            "labels hold no positive, so recall is undefined at every cut "
            "and no threshold maximises anything."
        )

    precision, recall = ranked.precision, ranked.recall
    f1 = 2 * precision * recall / np.maximum(precision + recall, _F1_FLOOR)

    best = int(np.argmax(f1))
    return OperatingPoint(
        threshold=float(ranked.scores[best]), f1=float(f1[best])
    )


def confusion_at(
    probs: object, labels: object, threshold: float
) -> Confusion:
    """The 2x2 table for `predict positive when p >= threshold`.

    Parameters
    ----------
    probs
        Predicted probabilities of the positive class, in ``[0, 1]``.
    labels
        Binary labels in ``{0, 1}``.
    threshold : float
        The cut, applied inclusively. Below every score it predicts every
        record positive; above every score, none.

    Returns
    -------
    Confusion
    """
    p, y = as_probabilities_and_labels(probs, labels)

    predicted = p >= threshold
    actual = y > 0.5
    return Confusion(
        true_negatives=int((~predicted & ~actual).sum()),
        false_positives=int((predicted & ~actual).sum()),
        false_negatives=int((~predicted & actual).sum()),
        true_positives=int((predicted & actual).sum()),
    )
