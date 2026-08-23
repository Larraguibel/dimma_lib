"""Threshold-free scores: log loss, Brier, normalized entropy.

These take a probability and a label and return one number, with no
operating point chosen anywhere. That is the property that makes them
usable for model selection here: a threshold is a free parameter with no
referent in this task, and a metric whose verdict moves with one is
answering a question the task does not pose.

Log loss and Brier are both **strictly proper**: each is minimized,
uniquely, by reporting the true conditional probability. Ranking scores
are not — every one of them is invariant to any increasing transform of
the predictions, so a model that orders records perfectly and states
probabilities three times too large scores exactly as well as the
calibrated one. `dimma.metrics.decomposition` splits a proper score into
the part a ranking score sees and the part it cannot.

Probabilities, not logits
-------------------------
Everything here takes probabilities, because calibration is a statement
about probabilities and there is nothing to say about a logit's
agreement with an observed rate. That costs precision at the tails: a
probability that reached 1.0 in float32 has already lost the logit that
would have scored it, and `log_loss` can only clip and report a bound.
`dimma.models.losses.batch_bce_loss` computes the same log loss from
logits without the round trip and is the one to quote when the model is
confident; the two agree to well past reporting precision otherwise.
"""

from __future__ import annotations

import numpy as np

from dimma.metrics._inputs import as_probabilities_and_labels, base_rate_entropy

_CLIP = 1e-15


def log_loss(probs: object, labels: object) -> float:
    """Return the mean binary cross-entropy, in nats. Lower is better.

    The same quantity the optimizer descends, which is why it is the
    default for choosing between runs: selection and training then agree
    on what better means, with no proxy in between.

    A prediction of exactly 0 or 1 that turned out wrong contributes
    about 34.5 rather than infinity, since probabilities are clipped into
    ``[1e-15, 1 - 1e-15]`` first. That bound is a floor on how bad a
    single record is allowed to look, not a fact about the model; see the
    module docstring for the route that does not need it.

    Parameters
    ----------
    probs : array-like of shape (n,)
        Predicted probabilities of the positive class, in ``[0, 1]``.
    labels : array-like of shape (n,)
        Binary labels in ``{0, 1}``.

    Returns
    -------
    float
        Mean cross-entropy over the records, in nats, at least 0.0 and
        at most about 34.5. Lower is better.

    Raises
    ------
    ValueError
        For any of the input problems `dimma.metrics._inputs` validates.
    """
    p, y = as_probabilities_and_labels(probs, labels)
    p = np.clip(p, _CLIP, 1.0 - _CLIP)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def brier_score(probs: object, labels: object) -> float:
    """Return the mean squared error against the 0/1 label. Lower is better.

    The other strictly proper score in common use, and the one whose
    decomposition is exact rather than approximate. Bounded in
    ``[0, 1]``, unlike log loss, so a single confident mistake moves it
    by at most ``1/n`` — which makes it the more stable of the two to
    watch across a hyperparameter sweep, and the less sensitive to the
    tail behaviour a private run is most likely to damage. Report both.

    Parameters
    ----------
    probs : array-like of shape (n,)
        Predicted probabilities of the positive class, in ``[0, 1]``.
    labels : array-like of shape (n,)
        Binary labels in ``{0, 1}``.

    Returns
    -------
    float
        Mean squared error, in ``[0, 1]``. Lower is better.

    Raises
    ------
    ValueError
        For any of the input problems `dimma.metrics._inputs` validates.
    """
    p, y = as_probabilities_and_labels(probs, labels)
    return float(np.mean((p - y) ** 2))


def normalized_entropy(probs: object, labels: object) -> float:
    """Return the log loss divided by the base rate's entropy. Lower is better.

    Makes log losses comparable across datasets with different base
    rates. Raw log loss falls as the base rate approaches 0 or 1 for
    reasons that have nothing to do with the model — predicting a rare
    event is simply cheaper to be right about on average — so a log loss
    of 0.15 is a different achievement at a 3% base rate than at 25%.
    Dividing by the entropy of the base rate removes that.

    Reads directly: 1.0 is what predicting the base rate for every record
    scores, so anything at or above 1.0 has learned nothing, and the
    distance below 1.0 is the fraction of the available uncertainty the
    model removed.

    Parameters
    ----------
    probs : array-like of shape (n,)
        Predicted probabilities of the positive class, in ``[0, 1]``.
    labels : array-like of shape (n,)
        Binary labels in ``{0, 1}``.

    Returns
    -------
    float
        Log loss in units of the base rate's entropy, ``>= 0``. 1.0 is
        the constant predictor. Lower is better.

    Raises
    ------
    ValueError
        If the labels are constant, leaving no uncertainty to normalize
        against, or for any of the input problems
        `dimma.metrics._inputs` validates.

    References
    ----------
    .. [1] He et al., "Practical Lessons from Predicting Clicks on Ads at
       Facebook", ADKDD 2014.
    """
    p, y = as_probabilities_and_labels(probs, labels)
    return log_loss(p, y) / base_rate_entropy(float(np.mean(y)))
