"""Poisson subsampling with truncation: a *modified* mechanism.

.. warning::

    Not the standard Poisson-subsampled mechanism. The standard
    ``PoissonSampledDpEvent`` bound is a lower bound on this
    mechanism's true privacy cost, not a valid guarantee: truncation
    gives the adversary additional advantage. Do not use it for
    published privacy claims without a tighter analysis.

Exists to support research into modified-mechanism accountants. If you
do not need truncated semantics use :mod:`dimma.core.sampling.poisson`.

It is a separate module so the choice is visible in the import line
rather than in a docstring inside a shared file. Everything but the
truncation is inherited from the standard mechanism.
"""

from __future__ import annotations

import numpy as np

from dimma.core.sampling.poisson import _pad_and_mask


def subsample(rng: np.random.Generator, n: int, p: float,
              b_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw a batch, truncating to ``b_max`` instead of raising.

    As :func:`dimma.core.sampling.poisson.subsample`, except an oversize
    draw is reduced via ``rng.choice``. Read the module warning first:
    that truncation invalidates the standard accounting.
    """
    bern = rng.random(n) < p
    idx = np.flatnonzero(bern)
    if idx.size > b_max:
        idx = rng.choice(idx, size=b_max, replace=False)
    return _pad_and_mask(idx, b_max)
