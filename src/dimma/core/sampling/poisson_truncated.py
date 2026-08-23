"""Poisson subsampling with truncation: a *modified* mechanism.

.. warning::

    Not the standard Poisson-subsampled mechanism. The standard
    ``PoissonSampledDpEvent`` bound does not apply here and is not a
    valid guarantee, and whether the true cost is higher or lower is
    unestablished. Do not use this for published privacy claims
    without an analysis of this mechanism. ADR-0007 records why, and
    why the lower-bound claim earlier code made was withdrawn.

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

    Parameters
    ----------
    rng
        Drives the inclusion coins and the truncating choice. Pass one
        generator for all sampling steps.
    n : int > 0
        Training set size; indices are drawn from ``0..n - 1``.
    p : float in [0, 1]
        Per-example inclusion probability of the untruncated draw. The
        realized marginal is lower whenever the truncation binds.
    b_max : int > 0
        The size an oversize draw is cut down to. Unlike the standard
        sampler's cap this bounds the mechanism, not only the memory.

    Returns
    -------
    indices : numpy.ndarray, shape ``(b_max,)``, ``int64``
        The drawn rows: padded out with index 0 when the draw came in
        under the cap, cut down via ``rng.choice`` when it did not.
    mask : numpy.ndarray, shape ``(b_max,)``, ``float32``
        1.0 in a real slot, 0.0 in a padded one. All ones on a
        truncated draw, since every slot is then real.

    See Also
    --------
    dimma.core.sampling.poisson.subsample : the standard mechanism,
        which raises on an oversize draw instead of truncating.

    Notes
    -----
    Raises nothing: an oversize draw is always representable, since it
    is cut down to fit. Read the module warning first — truncation
    invalidates the standard accounting.
    """
    bern = rng.random(n) < p
    idx = np.flatnonzero(bern)
    if idx.size > b_max:
        idx = rng.choice(idx, size=b_max, replace=False)
    return _pad_and_mask(idx, b_max)
