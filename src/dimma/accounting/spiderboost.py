"""Private SpiderBoost accounting: two branches, one accountant.

Accounts `dimma.algorithms.spiderboost`. This module travels with that
algorithm and is not general-purpose, per the membership rule in
`dimma.accounting`. What is *not* bespoke here is the mechanism: both
branches are ordinary Poisson-subsampled Gaussian releases, composed by
`sampling.composed_poisson_gaussian_epsilon`; ADR-0010 records why.
What this module owns is the *mapping*: which two schedules Algorithm 2
runs, and how the paper's three noise scales relate to the single
multiplier they share.

Assumed mechanism, beyond `sampling`'s
-------------------------------------
Every ``anchor_interval`` steps the anchor branch draws at rate
``b1/n`` and releases a mean gradient; every other step the variation
branch draws at rate ``b2/n`` and releases a mean gradient
*difference*. Over ``steps`` releases that is
``ceil(steps / anchor_interval)`` of the first and the rest of the
second. The branch sequence is ``t % anchor_interval``, fixed before
the run from ``steps`` and the interval alone, so the two schedules
compose *non-adaptively* with respect to each other - which is what
`sampling.composed_poisson_gaussian_epsilon` states its order
independence for. What is chosen adaptively is the variation branch's
noise *magnitude*, a function of prior releases only; ADR-0010 records
why that is sound.

Preconditions this cannot check
-------------------------------
Algorithm 2 has no clipping line and dimma adds none (ADR-0009), so
nothing here makes the sensitivity bounds true; they are assumed. The
numbers below are a guarantee only if, for every example ``x``:

- ``f(.; x)`` is ``lipschitz_constant``-Lipschitz, which is what bounds
  a per-sample gradient by ``L0`` and so the anchor mean's sensitivity
  by ``L0/b1``;
- ``f(.; x)`` is ``smoothness_constant``-smooth, which is what bounds a
  per-sample gradient difference by ``L1*||w_t - w_{t-1}||`` and so the
  variation mean's sensitivity by that over ``b2``;
- the step size is Theorem B.3's ``1/(2*L1)``, which the training loop
  takes as an `Optimizer` and cannot inspect.

Supply an ``L0`` the data exceeds and the noise is calibrated against a
sensitivity that is not the real one: the reported epsilon is then
false, silently and with no crash. Fitting either constant on the data
is itself an unaccounted access (ADR-0008, ADR-0009).

The saturating bound the fixed multiplier rests on is read off the
algorithm box rather than the privacy proof; ADR-0010 records that gap.

What is claimed is the privacy, not the convergence rate (ADR-0010).
"""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

from dimma.accounting.sampling import (
    Method,
    PoissonGaussianSchedule,
    calibrate_noise_multiplier,
    composed_poisson_gaussian_epsilon,
)


class NoiseScales(NamedTuple):
    """Algorithm 2's three scales, named as `train` takes them.

    Pass straight through to `dimma.algorithms.spiderboost.train`:
    every field is that keyword. They are standard deviations on the
    *released mean*, not on a sum, because that is where lines 9 and 13
    add the noise.
    """

    anchor_noise_scale: float
    variation_noise_rate: float
    variation_noise_cap: float


def release_counts(steps: int, anchor_interval: int) -> tuple[int, int]:
    """How many releases each branch makes over a run.

    Step ``t`` is an anchor step when ``t % anchor_interval == 0`` for
    ``t`` in ``range(steps)``, so step 0 always is. Exact rather than
    approximate: this is what the accountant composes over, and a
    mis-count is a mis-stated guarantee rather than a rounding.

    Parameters
    ----------
    steps : int >= 2
        The run's length in optimizer updates, which for this algorithm
        is also its length in releases.
    anchor_interval : int >= 1
        Algorithm 2's phase size ``q``, in steps. An interval of 1 makes
        every step an anchor.

    Returns
    -------
    anchor_releases, variation_releases : int
        Both non-negative and summing to ``steps``.

    Raises
    ------
    ValueError
        If ``steps`` is below 2 or ``anchor_interval`` is below 1.
    """
    _check_schedule(steps, anchor_interval)
    anchors = math.ceil(steps / anchor_interval)
    return anchors, steps - anchors


def noise_scales(*, lipschitz_constant: float, smoothness_constant: float,
                 target_epsilon: float, target_delta: float, steps: int,
                 anchor_interval: int, anchor_expected_batch_size: int,
                 variation_expected_batch_size: int, dataset_size: int,
                 method: Method = "rdp") -> NoiseScales:
    """The three noise scales a privacy budget buys.

    The calibrating direction. Both branches carry the *same* noise
    multiplier ``mu`` - the paper's constants make the variation
    branch's scale and its sensitivity saturate together, so their
    ratio is constant - which is why one scalar search settles all
    three scales:

    ===========================  =============================
    ``anchor_noise_scale``       ``mu * L0 / b1``
    ``variation_noise_rate``     ``mu * L1 / b2``
    ``variation_noise_cap``      ``2 * mu * L0 / b2``
    ===========================  =============================

    The returned scales satisfy ``cap / rate == 2 * L0 / L1`` by
    construction, which is the alignment `epsilon` insists on.

    Parameters
    ----------
    lipschitz_constant, smoothness_constant
        The paper's ``L0`` and ``L1``. Premises of the guarantee, not
        facts about the code - see the module docstring for what
        supplying the wrong ones costs.
    target_epsilon, target_delta
        The budget to calibrate against. Met, not approached: epsilon
        at the returned scales is at or just under ``target_epsilon``.
    steps, anchor_interval
        The run's schedule, as `dimma.algorithms.spiderboost.train`
        takes it. Must be the values the run will use.
    anchor_expected_batch_size, variation_expected_batch_size
        Algorithm 2's ``b_1`` and ``b_2``, each at most
        ``dataset_size``. Must be the values the run will use.
    dataset_size
        ``n``, which with the batch sizes fixes both sampling rates.
    method
        ``"rdp"`` (default) or ``"pld"``; ADR-0011 records why. It
        moves the answer materially, so calibrate and report under the
        same one.

    Returns
    -------
    scales : NoiseScales
        Keyword-for-keyword what `train` takes.

    Raises
    ------
    ValueError
        On a constant that is not positive, a schedule that does not
        describe a run, a batch larger than the dataset, or a budget no
        multiplier reaches.
    """
    releases = _releases_from_multiplier(
        steps=steps, anchor_interval=anchor_interval,
        anchor_expected_batch_size=anchor_expected_batch_size,
        variation_expected_batch_size=variation_expected_batch_size,
        dataset_size=dataset_size,
    )
    _check_constants(lipschitz_constant, smoothness_constant)

    multiplier = calibrate_noise_multiplier(
        releases, target_epsilon, target_delta, method=method)

    scales = NoiseScales(
        anchor_noise_scale=(
            multiplier * lipschitz_constant / anchor_expected_batch_size),
        variation_noise_rate=(
            multiplier * smoothness_constant / variation_expected_batch_size),
        variation_noise_cap=(
            2.0 * multiplier * lipschitz_constant
            / variation_expected_batch_size),
    )
    # An invariant, not input validation: these are the scales `epsilon`
    # would refuse if the formulas above ever drifted apart.
    assert math.isclose(
        scales.variation_noise_cap / scales.variation_noise_rate,
        2.0 * lipschitz_constant / smoothness_constant, rel_tol=1e-9)
    return scales


def epsilon(*, lipschitz_constant: float, smoothness_constant: float,
            anchor_noise_scale: float, variation_noise_rate: float,
            variation_noise_cap: float, target_delta: float, steps: int,
            anchor_interval: int, anchor_expected_batch_size: int,
            variation_expected_batch_size: int, dataset_size: int,
            method: Method = "rdp",
            accept_misaligned_scales: bool = False) -> float:
    """The epsilon a completed run spent.

    The reporting direction, and the one that takes scales rather than
    producing them - so it is here, not in `noise_scales`, that a
    caller can present a variation rate and cap whose ratio is not the
    ``2 * L0 / L1`` the fixed multiplier rests on (ADR-0010). Those are
    refused, with the conservative value
    ``b2 * min(rate / L1, cap / (2 * L0))`` named in the message;
    ``accept_misaligned_scales`` reports from it instead, with a
    warning.

    Parameters
    ----------
    lipschitz_constant, smoothness_constant
        The paper's ``L0`` and ``L1``. These must be the constants the
        scales were calibrated against; passing different ones reports
        the epsilon of a run that did not happen.
    anchor_noise_scale, variation_noise_rate, variation_noise_cap
        The three scales the run was given, exactly as passed to
        `train`. Unconverted, so no translation step can silently
        disagree with what ran.
    target_delta
        The ``delta`` at which epsilon is read off.
    steps, anchor_interval
        The run's schedule, as `noise_scales` takes it.
    anchor_expected_batch_size, variation_expected_batch_size
        The two branches' ``b_1`` and ``b_2``, as `noise_scales` takes
        them.
    dataset_size
        ``n``, which with the batch sizes fixes both sampling rates.
    method
        ``"rdp"`` (default) or ``"pld"``; ADR-0011 records why.
    accept_misaligned_scales
        Report the conservative epsilon for scales off the ratio
        instead of refusing them. The number is an upper bound, so it
        is safe to report, but it describes a mechanism whose noise the
        paper's analysis does not pin down.

    Returns
    -------
    epsilon : float
        The privacy cost at ``target_delta``, for a run matching the
        mechanism and preconditions this module assumes.

    Raises
    ------
    ValueError
        On a non-positive constant or scale, a schedule that does not
        describe a run, or misaligned scales without
        ``accept_misaligned_scales``.

    Warns
    -----
    UserWarning
        When misaligned scales are accepted, and as
        `sampling.composed_poisson_gaussian_epsilon` warns.
    """
    releases = _releases_from_multiplier(
        steps=steps, anchor_interval=anchor_interval,
        anchor_expected_batch_size=anchor_expected_batch_size,
        variation_expected_batch_size=variation_expected_batch_size,
        dataset_size=dataset_size,
    )
    _check_constants(lipschitz_constant, smoothness_constant)
    for name, scale in (("anchor_noise_scale", anchor_noise_scale),
                        ("variation_noise_rate", variation_noise_rate),
                        ("variation_noise_cap", variation_noise_cap)):
        if scale <= 0.0:
            raise ValueError(f"{name}={scale} must be positive.")

    anchor_multiplier = (
        anchor_noise_scale * anchor_expected_batch_size / lipschitz_constant)
    variation_multiplier = _variation_multiplier(
        lipschitz_constant, smoothness_constant, variation_noise_rate,
        variation_noise_cap, variation_expected_batch_size,
        accept_misaligned_scales,
    )

    anchor, variation = releases(1.0)
    return composed_poisson_gaussian_epsilon(
        [anchor._replace(noise_multiplier=anchor_multiplier),
         variation._replace(noise_multiplier=variation_multiplier)],
        target_delta, method=method,
    )


def _variation_multiplier(lipschitz_constant: float,
                          smoothness_constant: float, rate: float, cap: float,
                          batch_size: int, accept_misaligned: bool) -> float:
    """The variation branch's multiplier, or a refusal."""
    from_rate = batch_size * rate / smoothness_constant
    from_cap = batch_size * cap / (2.0 * lipschitz_constant)
    if math.isclose(from_rate, from_cap, rel_tol=1e-9):
        return from_rate

    required = 2.0 * lipschitz_constant / smoothness_constant
    conservative = min(from_rate, from_cap)
    if not accept_misaligned:
        raise ValueError(
            f"variation_noise_cap / variation_noise_rate = {cap / rate:.6g}, "
            f"but this mechanism needs 2 * L0 / L1 = {required:.6g}. Off "
            "that ratio the variation branch's effective noise multiplier "
            "varies with how far the parameters move, so no single epsilon "
            f"describes it. The sound value is {conservative:.6g}; pass "
            "accept_misaligned_scales=True to report epsilon from it, or "
            "take the scales from noise_scales(), which aligns them."
        )
    warnings.warn(
        f"variation_noise_cap / variation_noise_rate = {cap / rate:.6g} "
        f"rather than 2 * L0 / L1 = {required:.6g}, so the variation "
        f"branch is priced at its conservative multiplier {conservative:.6g}. "
        "The epsilon is an upper bound, but it is not the epsilon of the "
        "mechanism the paper analyses.",
        stacklevel=3,
    )
    return conservative


def _releases_from_multiplier(*, steps: int, anchor_interval: int,
                              anchor_expected_batch_size: int,
                              variation_expected_batch_size: int,
                              dataset_size: int):
    """Build the ``multiplier -> two schedules`` map both directions
    use, which is also the form `sampling.calibrate_noise_multiplier`
    takes."""
    _check_schedule(steps, anchor_interval)
    if dataset_size <= 0:
        raise ValueError(f"dataset_size={dataset_size} must be positive.")
    for name, size in (
            ("anchor_expected_batch_size", anchor_expected_batch_size),
            ("variation_expected_batch_size", variation_expected_batch_size)):
        if size <= 0:
            raise ValueError(f"{name}={size} must be positive.")
        if size > dataset_size:
            raise ValueError(
                f"{name}={size} exceeds dataset_size={dataset_size}, so its "
                "sampling rate would be above 1."
            )

    anchors, variations = release_counts(steps, anchor_interval)
    anchor_rate = anchor_expected_batch_size / dataset_size
    variation_rate = variation_expected_batch_size / dataset_size

    def releases(multiplier: float) -> list[PoissonGaussianSchedule]:
        return [
            PoissonGaussianSchedule(anchor_rate, multiplier, anchors),
            PoissonGaussianSchedule(variation_rate, multiplier, variations),
        ]

    return releases


def _check_schedule(steps: int, anchor_interval: int) -> None:
    if steps < 2:
        raise ValueError(
            f"steps={steps} must be at least 2; below that the algorithm's "
            "output rule has empty support."
        )
    if anchor_interval < 1:
        raise ValueError(f"anchor_interval={anchor_interval} must be >= 1.")


def _check_constants(lipschitz_constant: float,
                     smoothness_constant: float) -> None:
    if lipschitz_constant <= 0.0:
        raise ValueError(
            f"lipschitz_constant={lipschitz_constant} must be positive."
        )
    if smoothness_constant <= 0.0:
        raise ValueError(
            f"smoothness_constant={smoothness_constant} must be positive."
        )
