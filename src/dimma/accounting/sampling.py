"""Poisson-subsampled Gaussian accounting.

One function per sampler in `dimma.core.sampling`, so a call site names
the mechanism it accounts for. Both share an implementation: the
truncated sampler has no analysis of its own, so its function returns
the standard number under a different claim.

Assumed mechanism: each step independently includes every example with
probability ``q``, clips each per-example gradient to ``l_2`` norm
``C``, sums, and adds ``N(0, (z C)^2)``. Steps compose independently.
That is `dimma.algorithms.dp_sgd` exactly. Shuffled or fixed-size
batches, a data-dependent lot size, or clipping that misses a released
contribution all invalidate the number.

Adjacency is add-or-remove-one, which is what subsampling amplification
is stated for. Never accounted: hyperparameter search across runs, and
the gap between float and real-valued Gaussians.
"""

from __future__ import annotations

from typing import Literal

from dp_accounting import dp_event, pld, rdp

Method = Literal["rdp", "pld"]


def _epsilon(sampling_probability: float, noise_multiplier: float,
             num_compositions: int, target_delta: float,
             method: Method) -> float:
    """Compose the subsampled Gaussian event and read off epsilon."""
    if not 0.0 < sampling_probability <= 1.0:
        raise ValueError(
            f"sampling_probability={sampling_probability} must be in (0, 1]."
        )
    if noise_multiplier <= 0.0:
        raise ValueError(
            f"noise_multiplier={noise_multiplier} must be positive."
        )
    if num_compositions < 0:
        raise ValueError(f"num_compositions={num_compositions} must be >= 0.")
    if not 0.0 < target_delta < 1.0:
        raise ValueError(f"target_delta={target_delta} must be in (0, 1).")
    if method not in ("rdp", "pld"):
        raise ValueError(f"method={method!r} must be 'rdp' or 'pld'.")

    if num_compositions == 0:
        # No access to the data, no cost. Handled here because
        # dp-accounting rejects a zero count and the training loops
        # accept ``steps=0``; the two should agree on that edge.
        return 0.0

    accountant = rdp.RdpAccountant() if method == "rdp" else pld.PLDAccountant()
    accountant.compose(
        dp_event.PoissonSampledDpEvent(
            sampling_probability=sampling_probability,
            event=dp_event.GaussianDpEvent(noise_multiplier=noise_multiplier),
        ),
        count=num_compositions,
    )
    return float(accountant.get_epsilon(target_delta=target_delta))


def poisson_gaussian_epsilon(sampling_probability: float,
                             noise_multiplier: float, num_compositions: int,
                             target_delta: float, *,
                             method: Method = "rdp") -> float:
    """Epsilon for the standard Poisson-subsampled Gaussian mechanism.

    Accounts `dimma.core.sampling.poisson`. A valid
    ``(epsilon, target_delta)`` guarantee for a run matching the
    mechanism this module assumes.

    Parameters
    ----------
    sampling_probability
        ``q = L / N``, the per-example inclusion probability per step.
    noise_multiplier
        ``z = std / sensitivity``, the noise standard deviation over the
        clipping norm. This is `dp_sgd`'s ``noise_multiplier``
        unchanged; passing ``z * C`` overstates the guarantee.
    num_compositions
        Optimizer steps ``T``. Steps, not epochs.
    target_delta
        The ``delta`` at which epsilon is read off.
    method
        ``"rdp"`` (default) is the modern form of the moments accountant
        Abadi et al. introduced, so it is the bound faithful to the
        paper. ``"pld"`` is materially tighter for this mechanism,
        typically 30-40% lower. Hold it fixed across a comparison and
        report which was used: measuring DP-SGD under the looser bound
        while a non-classical method uses a bespoke tight one flatters
        the latter.

    Returns
    -------
    epsilon : float
        The privacy cost at ``target_delta``.
    """
    return _epsilon(sampling_probability, noise_multiplier, num_compositions,
                    target_delta, method)


def poisson_gaussian_truncated_epsilon(sampling_probability: float,
                                       noise_multiplier: float,
                                       num_compositions: int,
                                       target_delta: float, *,
                                       method: Method = "rdp") -> float:
    """The standard number for the truncated sampler. **Not a bound.**

    Accounts `dimma.core.sampling.poisson_truncated`, which caps draws
    inside the support of ``Binomial(n, q)``. That makes inclusion
    dependent across examples, and independence is what subsampling
    amplification assumes, so the standard analysis does not apply.

    The direction of the error is unestablished. Truncation couples the
    draws, which plausibly costs privacy; it also lowers each example's
    marginal inclusion probability, which plausibly saves some. Neither
    effect is quantified here, so this is a reference value and not a
    bound in either direction. Earlier versions of this code called it a
    lower bound; that claim had no proof behind it.

    Sound for development sanity checks and for ranking configurations,
    whose ordering it preserves under typical parameter changes. Never
    for a privacy claim. Numerically identical to
    `poisson_gaussian_epsilon`, whose parameters it takes.

    Returns
    -------
    epsilon_reference : float
        The standard mechanism's epsilon, for a mechanism that is not
        the standard one.
    """
    return _epsilon(sampling_probability, noise_multiplier, num_compositions,
                    target_delta, method)
