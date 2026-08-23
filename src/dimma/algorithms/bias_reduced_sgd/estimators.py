"""The inner mean estimator, as a seam, and Algorithm 1 behind it.

Algorithm 3 calls a private mean estimator four times per step — on the
batch, on each of its two halves, and on the single record — and
nothing else in the step depends on which estimator answers. This
module is that seam. A `MeanEstimator` carries the callable together
with the privacy claim one call makes, so an accountant reads what ran
off the claim instead of inferring it from the code that produced it.

One factory per Section 3 algorithm. `projection_estimator` is
Algorithm 1, the projection mechanism: perturb the batch mean, then
project onto the ``l_1`` ball. Algorithm 2, Gaussian ``l_1``-recovery,
is the other one the paper defines; it drops in later as a second
factory of this shape, and swapping it changes accuracy rather than
the step's arithmetic. Why Algorithm 1 stands in its slots at all is
the package docstring's first departure.

A factory rather than a bare function because of what varies. The
clipping norm, the radius and the noise multiplier are fixed for a run;
the batch size changes four times within one step, by orders of
magnitude. Closing over the first three and taking the fourth at the
call site is what lets a single estimator serve every slot.

Makes no privacy claim about a run: `GaussianMeanClaim` says what one
release is, and turning a sequence of releases into an (epsilon, delta)
belongs to `dimma.accounting`, per ADR-0003. The claim type is defined
there and re-exported here; its own docstring says why.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax

from dimma.accounting.bias_reduced_sgd import GaussianMeanClaim
from dimma.core import noise, projection

__all__ = [
    "GaussianMeanClaim",
    "MeanEstimator",
    "projection_estimator",
]


class MeanEstimator(NamedTuple):
    """One of Section 3's mean estimators, in the shape the four slots
    of Algorithm 3 take.

    Holds a function, so it is a *static* `jax.jit` argument and never a
    traced one: bind it outside the loop, as
    :mod:`~dimma.algorithms.bias_reduced_sgd.step` shows.
    """

    name: str
    """Which algorithm this is, for a caller reporting what ran.
    ``"projection"`` is Algorithm 1."""

    claim: GaussianMeanClaim
    """What one call releases, in the form the accountant reads."""

    estimate: Callable[[Any, jax.Array, float], Any]
    """``(mean_pytree, key, batch_size) -> private mean``. The mean
    arrives already divided by ``batch_size``; the size is passed too
    because the sensitivity, and so the noise scale, tracks it."""


def projection_estimator(
    *, clip_norm: float, radius: float, noise_multiplier: float
) -> MeanEstimator:
    """Algorithm 1: Gaussian noise, then projection onto ``K``.

    ``K = B_1(0, radius)``, one global ``l_1`` ball across the whole
    parameter pytree. The noise scale is
    ``noise_multiplier * 2 * clip_norm / batch_size``: the multiplier
    is dimensionless and fixed for the run, while the scale tracks each
    slot's own cardinality, ``batch_size == 1`` included.

    Parameters
    ----------
    clip_norm
        The paper's ``L``, and the norm stage 4 clips to. Read from
        `MeanEstimator.claim` by the step, so the bound and the noise
        calibrated against it cannot be given different numbers.
    radius
        ``K``'s radius, and the caller's number per ADR-0015. The
        paper's is ``clip_norm * sqrt(s)`` for ``s``-sparse per-sample
        gradients. It does *not* grow with the batch: each per-sample
        gradient is ``s``-sparse with ``l_2`` norm at most
        ``clip_norm``, hence ``l_1`` norm at most
        ``clip_norm * sqrt(s)``, and a mean of vectors in the convex
        ``K`` lies in ``K`` again. That is why Lemma 3.1's bound is
        available at every slot rather than only at the largest.
    noise_multiplier
        The scale over the sensitivity ``2 * clip_norm / batch_size``.
        Calibrating it from a privacy budget is the accountant's, not
        this module's.

    Returns
    -------
    MeanEstimator
        Closing over all three numbers, taking a mean, a key and a
        batch size.

    Notes
    -----
    Clipping does not spoil the sparsity the radius rests on: stage 4
    rescales a per-sample gradient by a scalar, which leaves its support
    exactly where it was, so ``L`` is enforced without touching ``s``.

    ``batch_size == 1`` — the ``G_0`` slot — is not special-cased and
    needs no case. Algorithm 1 carries no regime condition; the paper's
    own instantiation with Algorithm 2 carries one that fails at a batch
    of one, which the research note records.
    """
    if clip_norm <= 0:
        raise ValueError(
            f"clip_norm={clip_norm} must be positive; it is the l_2 "
            f"bound the noise is calibrated against, and a "
            f"non-positive one calibrates against nothing."
        )
    if radius < 0:
        raise ValueError(
            f"radius={radius} must be non-negative; no point lies "
            f"inside a ball of negative radius."
        )
    if noise_multiplier < 0:
        raise ValueError(
            f"noise_multiplier={noise_multiplier} must be "
            f"non-negative; it is a noise scale over a sensitivity."
        )

    def estimate(mean: Any, key: jax.Array, batch_size: float) -> Any:
        scale = noise_multiplier * 2.0 * clip_norm / batch_size
        return projection.project_l1_ball_pytree(
            noise.add_gaussian(mean, key, scale), radius
        )

    claim = GaussianMeanClaim(
        clip_norm=clip_norm, noise_multiplier=noise_multiplier
    )
    return MeanEstimator(name="projection", claim=claim, estimate=estimate)
