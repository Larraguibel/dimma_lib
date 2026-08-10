"""Shuffled epoch sampling: the ordinary, non-private draw.

Not a mechanism. Every example is visited once per epoch, so there is
no amplification and no accounting is stated against this at all. It is
here for the non-private baselines (ADR-0005).

Cardinality is fixed, so nothing is padded and there is no mask. The
tail of an epoch that does not divide evenly is dropped rather than
yielded short: a short batch is a second shape and would retrace the
compiled step.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


def batches(rng: np.random.Generator, n: int,
            batch_size: int) -> Iterator[np.ndarray]:
    """Yield index batches forever, reshuffling once per epoch.

    Parameters
    ----------
    rng
        Drives the permutation. Pass one generator for the whole run.
    n
        Training set size.
    batch_size
        Examples per batch. Every yielded array has exactly this
        length; the epoch's remainder is dropped.

    Yields
    ------
    numpy.ndarray
        Indices, shape ``(batch_size,)``. An array on its own, not the
        ``(indices, mask)`` pair the Poisson samplers return.

    Unbounded, because the loop above is counted in steps and not in
    epochs.

    Raises
    ------
    ValueError
        If ``batch_size`` is outside ``(0, n]``. Above ``n`` no epoch
        holds a full batch and the stream would be empty. Raised when
        the stream is built, not when it is first advanced.
    """
    if not 0 < batch_size <= n:
        raise ValueError(
            f"batch_size={batch_size} must be in (0, n] with n={n}; "
            f"above n no epoch contains a full batch and the stream "
            f"is empty."
        )

    def _stream() -> Iterator[np.ndarray]:
        while True:
            order = rng.permutation(n)
            for start in range(0, n - batch_size + 1, batch_size):
                yield order[start:start + batch_size]

    return _stream()
