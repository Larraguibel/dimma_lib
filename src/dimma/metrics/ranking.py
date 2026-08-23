"""Precision and recall at every cut, and the area under that curve.

Sort the records by predicted probability, descending, and walk down the
list. After each record, precision is the fraction of everything seen so
far that was positive and recall is the fraction of all positives that
has been seen; the two sequences are the curve. Average precision is that
curve integrated against recall — precision summed at each positive and
divided by the number of positives — and it is the single number usually
quoted as PR-AUC.

Every cut is taken, so nothing here fixes an operating point — see
ADR-0016. What it reads is the *order* the model put the records in.
Any strictly increasing transform of the scores leaves that order alone,
so a model whose probabilities are three times too large scores exactly
what the calibrated one scores — which is the half of a proper score that
`dimma.metrics.decomposition` calls discrimination, and none of the half
that `dimma.metrics.calibration` measures. Read alongside
`dimma.metrics.scoring`, never instead of it.

The floor is the base rate, not zero: an ordering that carries no
information scores about the fraction of records that are positive, so
average precision is only interpretable against that number, and it moves
between datasets for reasons that have nothing to do with the model.

Both the curve and its area depend on how runs of equal scores are
ordered; `dimma.metrics._ranked` owns that tie order and the argument
for it.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from dimma.metrics._ranked import rank_by_score


class PrecisionRecallCurve(NamedTuple):
    """The curve as two arrays, and the number that summarizes it.

    `precision` against `recall` is the plot, one point per record in
    descending score order, so both arrays are as long as the input.
    Neither is monotone on its own account — recall rises and then holds
    flat wherever a negative is passed, precision drops there and climbs
    back at the next positive — and the sawtooth is the curve rather than
    noise in it.

    Attributes
    ----------
    precision : np.ndarray of shape (n,), float in [0, 1]
        Positives at or above each cut, over the number of records at or
        above that cut.
    recall : np.ndarray of shape (n,), float in [0, 1]
        Positives at or above each cut, over all positives.
    average_precision : float in [0, 1]
        Precision averaged over the positives: the area under the curve.
    """

    precision: np.ndarray
    recall: np.ndarray
    average_precision: float


def pr_curve(probs: object, labels: object) -> PrecisionRecallCurve:
    """Return precision and recall at every cut, and the area under them.

    Parameters
    ----------
    probs : array-like of shape (n,)
        Predicted probabilities of the positive class, in ``[0, 1]``.
    labels : array-like of shape (n,)
        Binary labels in ``{0, 1}``.

    Returns
    -------
    PrecisionRecallCurve
        One point per record in descending score order, plus the average
        precision over them.

    Raises
    ------
    ValueError
        If the labels hold no positives — recall and average precision
        both divide by that count, and there is no ordering of a set
        with nothing to find in it — or for any of the input problems
        `dimma.metrics._inputs` validates.
    """
    ranked = rank_by_score(probs, labels)
    if ranked.positives == 0.0:
        raise ValueError(
            "labels hold no positives, so there is nothing for an ordering "
            "to put at the top and no positive count to divide by."
        )

    return PrecisionRecallCurve(
        precision=ranked.precision,
        recall=ranked.recall,
        average_precision=float(
            (ranked.precision * ranked.hits).sum() / ranked.positives
        ),
    )
