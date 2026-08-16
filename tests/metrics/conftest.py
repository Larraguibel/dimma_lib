"""Synthetic predictions whose calibration is known by construction.

Every calibration claim needs a case where the right answer is not in
doubt. Drawing ``p`` first and then ``y ~ Bernoulli(p)`` gives exactly
that: ``p`` *is* the true conditional probability, so a model reporting
it is perfectly calibrated, and any gap a metric finds is that metric's
own sampling noise rather than a fault in the predictions.

Sizes are large enough that the noise sits well below what the
assertions test for, and every draw is seeded.
"""

from __future__ import annotations

import numpy as np
import pytest

N = 200_000


@pytest.fixture
def rng():
    return np.random.default_rng(20260815)


@pytest.fixture
def calibrated(rng):
    """``(probs, labels)`` where ``probs`` is the truth, base rate ~0.25.

    Beta(2, 6) has mean 0.25, which puts this at Criteo's base rate with
    a realistic spread rather than a uniform one.
    """
    probs = rng.beta(2.0, 6.0, size=N)
    labels = (rng.random(N) < probs).astype(np.float64)
    return probs, labels


@pytest.fixture
def uninformative(rng):
    """``(probs, labels)`` for a model that only knows the base rate."""
    labels = (rng.random(N) < 0.25).astype(np.float64)
    return np.full(N, float(labels.mean())), labels
