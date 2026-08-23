"""Poisson subsampling: the standard mechanism.

Each example is independently included with probability ``p``. This is
the sampling assumption standard subsampled-Gaussian accounting is
stated against (``dp_accounting.PoissonSampledDpEvent``). States what
it samples and computes no privacy budget; ADR-0003.

Cardinality is data-dependent, so the draw uses a
``numpy.random.Generator`` outside any JIT region and pads to a fixed
shape so everything downstream is compilable. The mask is consumed by
:mod:`dimma.core.aggregation`.
"""

from __future__ import annotations

import math

import numpy as np


def padded_batch_size(b_expected: int, n: int,
                      margin_sigmas: float = 6.0) -> int:
    """Return the padding cap ``b_max`` for a Poisson-subsampled batch.

    Parameters
    ----------
    b_expected : int > 0
        Expected batch size, so the sampling rate is
        ``p = b_expected / n``.
    n : int > 0
        Training set size.
    margin_sigmas : float >= 0, default 6.0
        Standard deviations of headroom above ``b_expected``. At 6 a
        draw exceeds the cap with probability below ~1e-9 for typical
        DP-SGD configurations.

    Returns
    -------
    int
        ``min(ceil(b_expected + margin_sigmas * sigma + 4), n)``, where
        ``sigma = sqrt(b_expected * max(1 - p, 0))`` — the sigma margin
        plus four slots of absolute slack, rounded up, then clamped to
        the dataset size. Where the clamp to ``n`` binds, the cap is
        exact rather than probabilistic and :func:`subsample` cannot
        raise.

    Notes
    -----
    A padding cap, not a privacy parameter. Too low makes
    :func:`subsample` raise; too high only wastes memory. Passing ``n``
    straight to :func:`subsample` is always sound and makes the failure
    impossible, at the cost of a batch the size of the dataset — the
    trade ADR-0007 records, and the reason the clamp to ``n`` is not
    truncation in the sense of
    :mod:`dimma.core.sampling.poisson_truncated`.
    """
    p = b_expected / n
    std = math.sqrt(b_expected * max(1.0 - p, 0.0))
    return min(int(math.ceil(b_expected + margin_sigmas * std + 4)), n)


def _pad_and_mask(idx: np.ndarray, b_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad ``idx`` to ``b_max`` slots with index 0 and build the mask."""
    k = idx.size
    pad = b_max - k
    indices = np.concatenate([idx, np.zeros(pad, dtype=np.int64)])
    mask = np.concatenate(
        [np.ones(k, dtype=np.float32), np.zeros(pad, dtype=np.float32)]
    )
    return indices, mask


def subsample(rng: np.random.Generator, n: int, p: float,
              b_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw a batch, raising if it exceeds ``b_max``.

    Parameters
    ----------
    rng
        Drives the per-example inclusion coins. Pass one generator for
        all sampling steps, which is what keeps the draws independent.
    n : int > 0
        Training set size; indices are drawn from ``0..n - 1``.
    p : float in [0, 1]
        Per-example inclusion probability.
    b_max : int > 0
        Padding cap, typically from :func:`padded_batch_size`.

    Returns
    -------
    indices : numpy.ndarray, shape ``(b_max,)``, ``int64``
        The drawn rows, padded out to the cap with index 0.
    mask : numpy.ndarray, shape ``(b_max,)``, ``float32``
        1.0 in a real slot, 0.0 in a padded one. Padded slots hold a
        real row, so the mask is what makes them harmless.

    Raises
    ------
    RuntimeError
        If the draw exceeds ``b_max``, meaning the cap was set too low.
        Raising rather than truncating; ADR-0007.
    """
    bern = rng.random(n) < p
    idx = np.flatnonzero(bern)
    if idx.size > b_max:
        raise RuntimeError(
            f"Poisson subsample drew {idx.size} samples but b_max={b_max} "
            f"(n={n}, p={p}). Increase margin_sigmas in padded_batch_size, "
            f"or use dimma.core.sampling.poisson_truncated if you accept "
            f"its heuristic accountant."
        )
    return _pad_and_mask(idx, b_max)
