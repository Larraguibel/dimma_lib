"""One coercion and one set of checks, shared by every metric here.

Metrics are handed whatever the caller has — a JAX array off the device,
a pandas column, a Python list — and every function in this package
needs the same three things from it: float64, one dimension, and the
range each argument is supposed to occupy.

float64 rather than the float32 the model trains in. These are sums over
a few hundred thousand records, and a float32 accumulator loses the last
digits of exactly the quantity the whole exercise is about: a
calibration gap is a small difference between two numbers near the base
rate, and it is read at the fourth decimal.
"""

from __future__ import annotations

import numpy as np


def as_probabilities_and_labels(
    probs: object, labels: object
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(probs, labels)`` as validated 1-D float64 arrays.

    Parameters
    ----------
    probs
        Predicted probabilities of the positive class, in ``[0, 1]``.
        Note that these are probabilities, not the logits the model
        emits; `dimma.metrics.scoring` says why the distinction is
        load-bearing and where the stable route is.
    labels
        Binary labels in ``{0, 1}``.

    Raises
    ------
    ValueError
        If the shapes disagree, either argument is not one-dimensional
        or is empty, either holds a non-finite value, ``probs`` leaves
        ``[0, 1]``, or ``labels`` holds a value other than 0 or 1.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)

    if p.ndim != 1 or y.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional arrays, got probs.ndim={p.ndim} and "
            f"labels.ndim={y.ndim}."
        )
    if p.shape != y.shape:
        raise ValueError(
            f"probs and labels must be the same length, got {p.shape[0]} "
            f"and {y.shape[0]}."
        )
    if p.size == 0:
        raise ValueError("Cannot score an empty array.")
    if not np.all(np.isfinite(p)):
        raise ValueError(
            "probs holds a non-finite value. A diverged run reaches here as "
            "nan; decide what that run means at the call site rather than "
            "letting it propagate into a reported number."
        )
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError(
            f"probs must lie in [0, 1], got range "
            f"[{p.min():.6g}, {p.max():.6g}]. Logits are the likely cause: "
            f"these functions take probabilities."
        )
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("labels must be 0 or 1.")

    return p, y


def base_rate_entropy(base_rate: float) -> float:
    """Binary entropy of the base rate, in nats.

    The uncertainty no model can remove, and the denominator of
    `dimma.metrics.scoring.normalized_entropy`.

    Raises
    ------
    ValueError
        If the base rate is 0 or 1. Entropy is then zero, every model
        scores a log loss of zero on a constant label, and the
        normalization it anchors divides by nothing.
    """
    if base_rate <= 0.0 or base_rate >= 1.0:
        raise ValueError(
            f"Base rate is {base_rate:.6g}: the labels are constant, so "
            f"there is no uncertainty to normalize against."
        )
    q = 1.0 - base_rate
    return float(-base_rate * np.log(base_rate) - q * np.log(q))
