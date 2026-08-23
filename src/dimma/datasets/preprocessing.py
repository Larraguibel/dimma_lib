"""Maps a loader composes, shared across datasets. Names no dataset.

Every function here says whether it reads across records. ADR-0012
rests on that difference and records what it costs either way.
"""

from __future__ import annotations

import warnings

import numpy as np


def cap_feature_norms(
    x: np.ndarray, bound: float
) -> tuple[np.ndarray, float]:
    """Rescale each row to ``l_2`` norm at most ``bound``. Per record.

    Row ``i`` of the output depends on row ``i`` of ``x`` and on
    ``bound``, and on nothing else. See ADR-0012.

    Parameters
    ----------
    x : np.ndarray, shape ``(n, d)``
        One record per row.
    bound : float
        The norm to enforce.

    Returns
    -------
    capped : np.ndarray, shape ``(n, d)``
        ``x`` with every row inside the ball of radius ``bound``.
    bound : float
        The bound just enforced, so that the number reaches an
        accountant from the operation that made it true.

    Raises
    ------
    ValueError
        If ``bound`` is not finite and positive.

    Warns
    -----
    UserWarning
        If any row's norm is not finite, so that ``bound`` is not
        enforced for it and no constant derived from ``bound`` bounds
        this data. Existence only, never a count: ADR-0012 records why
        that one data-dependent signal leaves here and how narrowly it
        is drawn.
    """
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError(f"bound={bound} must be finite and positive.")

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        # How many rows is a statistic of the data, and a warning
        # carries it out of here as surely as a return value would.
        warnings.warn(
            "At least one row has a norm that is not finite, so `bound` "
            "is not enforced for it and a constant derived from `bound` "
            "does not bound this data. How many such rows there are is "
            "deliberately not reported.",
            stacklevel=2,
        )
    # A non-finite norm is the only way this divide goes invalid, and
    # the warning above has already said so, in more words than numpy's
    # would and ahead of a caller who turns warnings into errors.
    with np.errstate(invalid="ignore"):
        return x / np.maximum(1.0, norms / bound), bound
