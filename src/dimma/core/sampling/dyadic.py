"""Dyadic subsampling: a fixed-size draw at a randomly chosen scale.

A scale ``N`` is drawn from a truncated geometric law over
``0..max_scale(n)``, and the batch is then ``2 ** (N + 1)`` distinct
examples drawn uniformly without replacement, carrying its own split
into two halves of ``2 ** N`` each, alongside one record drawn
independently of it. `dyadic` names the ladder of sizes the scale
indexes, and no algorithm above it.

Cardinality is fixed by the scale before any data is touched, so it is
not data-dependent: nothing is padded and there is no mask. A draw is
indices alone, as in `shuffled` and unlike either Poisson sampler.

No standard subsampled-Gaussian accounting is stated against this. The
batch is a fixed-size draw without replacement rather than a Poisson
one, and a caller releasing the whole and both halves releases three
quantities from a single draw, which amplify jointly and not three
times over. The amplification statement that covers it is Lemma 5.3's,
and it belongs with the accountant that uses it,
`dimma.accounting.bias_reduced_sgd`. This module states what it
samples; it computes no privacy budget.

ADR-0007's raise-or-truncate question does not arise here.
``2 ** (max_scale(n) + 1) == 2 ** floor(log2 n) <= n`` for every
``n >= 2``, so an oversize draw is impossible: there is no padding cap
to exceed and nothing to raise about. What carries over is the ADR's
principle, that a draw is never reshaped to fit a memory bound. The
bound here is `max_scale` itself, passed lower: that runs a different
law over the scale, which is analysable as such, rather than cutting
down batches the law has already sized.

Reference: B. Ghazi, C. Guzman, P. Kamath, R. Kumar, P. Manurangsi,
"Differentially Private Optimization with Sparse Gradients", NeurIPS
2024. Definition 5.1 is the scale law; Section 5 is where the draw is
consumed.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class DyadicDraw(NamedTuple):
    """One step's draw: a batch, its two halves, and one record.

    `whole` is in uniformly random order, so `odd` and `even` are a
    uniformly random equal partition of it. Nothing downstream may
    depend on more than that: the two halves are only ever required to
    be disjoint, to cover `whole`, and to be identically distributed.

    Three fields and no mask. Every index is real, so there is nothing
    for a mask to switch off.
    """

    scale: int
    """``N``, the drawn scale. The batch holds ``2 ** (scale + 1)``
    indices and each half holds ``2 ** scale``."""

    whole: np.ndarray
    """``B``, shape ``(2 ** (scale + 1),)``, ``int64``, distinct."""

    single: np.ndarray
    """``I``, shape ``(1,)``, ``int64``, drawn independently of
    `whole` and so free to collide with it."""

    @property
    def odd(self) -> np.ndarray:
        """The first half of `whole`, shape ``(2 ** scale,)``."""
        return self.whole[: 1 << self.scale]

    @property
    def even(self) -> np.ndarray:
        """The second half of `whole`, shape ``(2 ** scale,)``."""
        return self.whole[1 << self.scale :]


def max_scale(n: int) -> int:
    """Definition 5.1's ``M = floor(log2 n) - 1``.

    Computed as ``n.bit_length() - 2``, which is exact for every ``n``.
    ``math.log2`` is not: at large powers of two its result rounds the
    wrong way and the floor comes back one too low or one too high.

    Raises
    ------
    ValueError
        If ``n < 2``. A dataset of one admits no scale, since the
        smallest batch the ladder describes holds two examples.
    """
    if n < 2:
        raise ValueError(
            f"n={n} must be at least 2; the smallest batch on the "
            f"ladder holds 2 ** (0 + 1) = 2 examples, so a smaller "
            f"dataset admits no scale at all."
        )
    return int(n).bit_length() - 2


def scale_probabilities(max_scale: int) -> np.ndarray:
    """Definition 5.1's pmf over ``0..max_scale``.

    ``p_k = C_M / 2 ** k`` with ``C_M = 1 / (2 * (1 - 2 ** -(M + 1)))``,
    the normalizer that makes the truncated geometric law sum to one.
    Returns length ``max_scale + 1``, in ``float64``.

    Read by the debiasing weight ``1 / p_N`` as well as by
    :func:`draw_scale`, so the two cannot disagree about the law.

    Raises
    ------
    ValueError
        If ``max_scale`` is negative.
    """
    if max_scale < 0:
        raise ValueError(
            f"max_scale={max_scale} must be non-negative; the law is "
            f"over the scales 0..max_scale, which is empty below zero."
        )
    normalizer = 1.0 / (2.0 * (1.0 - 2.0 ** -(max_scale + 1)))
    return normalizer / 2.0 ** np.arange(max_scale + 1, dtype=np.float64)


def draw_scale(rng: np.random.Generator, max_scale: int) -> int:
    """Draw ``N ~ TGeom(max_scale)`` by inverse CDF over the pmf table.

    Parameters
    ----------
    rng
        Drives the coin. Pass one generator for the whole run, as for
        every other sampler here.
    max_scale
        ``M``. Usually :func:`max_scale` of the training set size;
        lower is a different law and not a truncation of this one.

    Returns
    -------
    int
        A scale in ``0..max_scale``. A plain `int`, not a
        zero-dimensional array: it sizes the batch, so it is a host
        quantity throughout.
    """
    cumulative = np.cumsum(scale_probabilities(max_scale))
    drawn = int(np.searchsorted(cumulative, rng.random(), side="right"))
    return min(drawn, max_scale)


def subsample(rng: np.random.Generator, n: int, scale: int) -> DyadicDraw:
    """Draw the batch, its halves and the single index at ``scale``.

    ``2 ** (scale + 1)`` distinct indices uniformly at random without
    replacement, shuffled so their order carries no structure, plus one
    index drawn uniformly and independently. Indices already in the
    batch are *not* rejected from the single draw: it is uniform over
    the whole training set by construction, and rejecting collisions
    would change both its distribution and the sensitivity argument
    that rests on it.

    Parameters
    ----------
    rng
        Drives both draws. One generator for all sampling steps keeps
        them independent.
    n
        Training set size.
    scale
        ``N``, from :func:`draw_scale`. The batch size is fixed by it
        before any data is touched.

    Returns
    -------
    DyadicDraw
        Indices only, at exact sizes. Not the ``(indices, mask)`` pair
        the Poisson samplers return.

    Raises
    ------
    ValueError
        If ``n < 2``, or if ``scale`` is outside ``0..max_scale(n)``.
        Above the maximum the batch would not fit the training set,
        which is what the maximum is.
    """
    ceiling = max_scale(n)
    if not 0 <= scale <= ceiling:
        raise ValueError(
            f"scale={scale} must be in 0..{ceiling} for n={n}; at "
            f"scale {scale} the batch holds {1 << (scale + 1)} of "
            f"{n} examples, drawn without replacement."
        )
    batch_size = 1 << (scale + 1)
    chosen = rng.choice(n, size=batch_size, replace=False)
    whole = rng.permutation(chosen).astype(np.int64, copy=False)
    single = rng.integers(0, n, size=1, dtype=np.int64)
    return DyadicDraw(scale=scale, whole=whole, single=single)
