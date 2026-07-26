"""Poisson subsampling: the standard mechanism.

Each example is independently included with probability ``p``. This is
the sampling assumption standard subsampled-Gaussian accounting is
stated against (``dp_accounting.PoissonSampledDpEvent``). States what it
samples; computes no privacy budget.

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
    """Padding cap ``b_max`` for a Poisson-subsampled batch.

    ``b_expected + margin_sigmas * sqrt(b_expected * (1 - p))`` with
    ``p = b_expected / n``. At 6 sigmas a draw exceeds the cap with
    probability below ~1e-9 for typical DP-SGD configurations.

    Clamped to ``n``, which is the cap the mechanism already carries:
    the draw is ``Binomial(n, p)``, so no cap above ``n`` is meaningful
    and clamping to it removes no mass. Where the clamp binds the cap is
    exact rather than probabilistic and :func:`subsample` cannot raise.
    That is not truncation in the sense of
    :mod:`dimma.core.sampling.poisson_truncated`, which caps *inside*
    the support and does change the mechanism.

    This is a padding cap, not a privacy parameter. Too low causes an
    exception here or truncation there; too high only wastes memory.
    Passing ``n`` directly to :func:`subsample` is always sound and
    costs an ``O(n)`` batch, which is what this function trades away for
    an ``O(b_expected)`` one at a ~1e-9 failure rate.
    """
    p = b_expected / n
    std = math.sqrt(b_expected * max(1.0 - p, 0.0))
    return min(int(math.ceil(b_expected + margin_sigmas * std + 4)), n)


def _pad_and_mask(idx: np.ndarray, b_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad indices to ``b_max`` slots and build the matching mask.

    Shared by every Poisson variant. Padded slots are index 0, which is
    a real row producing a real gradient, so the mask is what makes them
    harmless.
    """
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

    Raises rather than truncating, because truncating would change the
    mechanism. Use one ``rng`` for all sampling steps to keep the draws
    independent.

    Returns ``(indices, mask)``, both shape ``(b_max,)``: padded indices
    into the training set, and 1.0 for real entries against 0.0 for
    padding.

    Raises
    ------
    RuntimeError
        If the draw exceeds ``b_max``, meaning the cap was set too low.
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
