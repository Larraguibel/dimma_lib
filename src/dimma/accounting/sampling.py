"""Poisson-subsampled Gaussian accounting.

One function per sampler in `dimma.core.sampling`, so a call site names
the mechanism it accounts for. Both share an implementation: the
truncated sampler has no analysis of its own, so its function returns
the standard number under a different claim.

Assumed mechanism: each step independently includes every example with
probability ``q``, forms a sum whose ``l_2`` sensitivity under
add-or-remove-one is ``S``, and adds ``N(0, (z S)^2)``. Steps compose
independently. What bounds ``S`` is the caller's business and differs
by algorithm: `dimma.algorithms.dp_sgd` clips each per-example gradient
to ``C``, so ``S = C``; `dimma.algorithms.spiderboost` clips nothing
and bounds it by assumption instead (ADR-0009). Shuffled or fixed-size
batches, a data-dependent lot size, or a released contribution the
sensitivity bound does not cover all invalidate the number.

Releases need not share a rate or a multiplier. A method that releases
on two schedules composes as a *sequence* of these events, which is
`composed_poisson_gaussian_epsilon`; the single-schedule functions are
that with one entry.

Adjacency is add-or-remove-one, which is what subsampling amplification
is stated for. Never accounted: hyperparameter search across runs, and
the gap between float and real-valued Gaussians.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple

from dp_accounting import (
    calibrate_dp_mechanism,
    dp_event,
    mechanism_calibration,
    pld,
    rdp,
)

Method = Literal["rdp", "pld"]

# The least noise `calibrate_noise_multiplier` will consider. Not a
# privacy parameter: below it epsilon is large enough that the Renyi
# accountant drops orders wholesale, and a budget met there is a budget
# not worth calibrating to.
_SEARCH_LOWER_BOUND = 1e-2

# Where the bracket search starts doubling from. A noise multiplier of
# order 1 is the common case, so most searches bracket in a step or two.
_SEARCH_INITIAL_GUESS = 1.0

# The most noise `calibrate_noise_multiplier` will consider. Past here
# a budget is treated as unreachable rather than chased.
_SEARCH_UPPER_BOUND = 1e4


class PoissonGaussianSchedule(NamedTuple):
    """A run of releases sharing one rate and one multiplier.

    The unit `composed_poisson_gaussian_epsilon` composes. Field names
    match `poisson_gaussian_epsilon`'s parameters, because one entry of
    this is exactly that call.
    """

    sampling_probability: float
    noise_multiplier: float
    num_compositions: int


class _DroppedOrders(logging.Handler):
    """Collects `RdpAccountant`'s failure-to-converge notices.

    They arrive through `absl` logging rather than `warnings`, so
    `warnings.catch_warnings` does not see them and a caller who has
    turned warnings into errors is not protected. This turns them back
    into the channel a caller can control.
    """

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "failed to converge" in message:
            self.messages.append(message)


@contextlib.contextmanager
def _watching_rdp_orders():
    """Capture dropped Renyi orders, and mute the log line meanwhile.

    Muting has to be done by displacing the handlers rather than by
    raising the logger's level, which would drop the records before
    anything could collect them. Restores both on the way out, so a
    caller's own `absl` logging configuration survives.
    """
    logger = logging.getLogger("absl")
    handler = _DroppedOrders()
    handlers, propagate = logger.handlers[:], logger.propagate
    logger.handlers = [handler]
    logger.propagate = False
    try:
        yield handler
    finally:
        logger.handlers = handlers
        logger.propagate = propagate


def _validate(releases: Sequence[PoissonGaussianSchedule],
              target_delta: float, method: Method, *,
              qualify: bool = True) -> None:
    """Reject nonsense before it becomes a number someone reports.

    ``qualify`` is False for the single-schedule wrappers, whose caller
    typed bare parameter names and should not be told about an index
    into a sequence they never passed.
    """
    if not 0.0 < target_delta < 1.0:
        raise ValueError(f"target_delta={target_delta} must be in (0, 1).")
    if method not in ("rdp", "pld"):
        raise ValueError(f"method={method!r} must be 'rdp' or 'pld'.")
    for index, release in enumerate(releases):
        where = f"releases[{index}]." if qualify else ""
        if not 0.0 < release.sampling_probability <= 1.0:
            raise ValueError(
                f"{where}sampling_probability="
                f"{release.sampling_probability} must be in (0, 1]."
            )
        if release.noise_multiplier <= 0.0:
            raise ValueError(
                f"{where}noise_multiplier={release.noise_multiplier} "
                "must be positive."
            )
        if release.num_compositions < 0:
            raise ValueError(
                f"{where}num_compositions={release.num_compositions} "
                "must be >= 0."
            )


def _bracket(epsilon_at: Callable[[float], float], target_epsilon: float,
             target_delta: float) -> tuple[float, float]:
    """Two multipliers the answer lies between, found by doubling.

    Expands outward from a guess rather than handing the root-finder
    the whole plausible range: `brentq`'s tolerance is absolute, so a
    bracket spanning decades is paid for in iterations, and PLD's
    discretisation grows without bound as the multiplier shrinks, so a
    search that evaluates the small end only when it must is the one
    that terminates.

    Epsilon is monotone decreasing in the multiplier, which is what
    makes doubling sound.
    """
    guess = _SEARCH_INITIAL_GUESS
    if epsilon_at(guess) > target_epsilon:
        lower, upper = guess, guess * 2.0
        while epsilon_at(upper) > target_epsilon:
            lower, upper = upper, upper * 2.0
            if upper > _SEARCH_UPPER_BOUND:
                raise ValueError(
                    f"No noise multiplier up to {_SEARCH_UPPER_BOUND:g} "
                    f"reaches epsilon={target_epsilon} at "
                    f"delta={target_delta} for these releases. Under 'rdp' "
                    "epsilon is a minimum over a finite grid of Renyi "
                    "orders, so it floors above zero however much noise is "
                    "added; 'pld' has no such floor."
                )
        return lower, upper

    upper, lower = guess, guess / 2.0
    while epsilon_at(lower) <= target_epsilon:
        upper, lower = lower, lower / 2.0
        if lower < _SEARCH_LOWER_BOUND:
            raise ValueError(
                f"epsilon={target_epsilon} at delta={target_delta} is met "
                f"already by a noise multiplier under {_SEARCH_LOWER_BOUND}, "
                "below which this does not search. The budget is so loose "
                "for this many releases that calibrating to it is "
                "meaningless."
            )
    return lower, upper


def _accountant(method: Method):
    return rdp.RdpAccountant() if method == "rdp" else pld.PLDAccountant()


def composed_poisson_gaussian_epsilon(
        releases: Sequence[PoissonGaussianSchedule], target_delta: float, *,
        method: Method = "rdp") -> float:
    """Epsilon for several Poisson-subsampled Gaussian schedules at once.

    The general form of `poisson_gaussian_epsilon`: a method that
    releases at more than one rate, multiplier or schedule passes one
    entry per schedule and gets the cost of all of them composed in a
    single accountant. `dimma.accounting.spiderboost` is the caller
    that needs it; ADR-0010 records why, and what composing buys over
    summing the schedules' epsilons separately.

    Order does not matter: the schedules are non-adaptive with respect
    to each other, each one's privacy curve being fixed before the run.

    Parameters
    ----------
    releases
        One `PoissonGaussianSchedule` each. Entries with
        ``num_compositions == 0`` cost nothing and are dropped, so an
        empty sequence - or one describing no releases at all - is 0.0.
    target_delta
        The ``delta`` at which epsilon is read off, for the whole
        composition rather than per schedule.
    method
        ``"rdp"`` (default) or ``"pld"``; ADR-0011 records why. Hold it
        fixed across a comparison and report which was used.

    Returns
    -------
    epsilon : float
        The privacy cost of every schedule together, at
        ``target_delta``.

    Warns
    -----
    UserWarning
        When the Renyi accountant drops an order it could not compute.
        The epsilon stays a valid upper bound, but a looser one than
        the method can give.
    """
    return _composed(releases, target_delta, method, qualify=True)


def _composed(releases: Sequence[PoissonGaussianSchedule],
              target_delta: float, method: Method, *,
              qualify: bool, warn: bool = True) -> float:
    """`composed_poisson_gaussian_epsilon`, minus the naming of blame.

    ``warn`` is False for the multipliers a search merely probes: those
    are not results, and a dropped order at one says nothing about the
    answer the search returns.
    """
    _validate(releases, target_delta, method, qualify=qualify)

    charged = [r for r in releases if r.num_compositions > 0]
    if not charged:
        # No access to the data, no cost. Handled here because
        # dp-accounting rejects a zero count and the training loops
        # accept ``steps=0``; the two should agree on that edge.
        return 0.0

    accountant = _accountant(method)
    with _watching_rdp_orders() as dropped:
        for release in charged:
            accountant.compose(
                dp_event.PoissonSampledDpEvent(
                    sampling_probability=release.sampling_probability,
                    event=dp_event.GaussianDpEvent(
                        noise_multiplier=release.noise_multiplier),
                ),
                count=release.num_compositions,
            )
        epsilon = float(accountant.get_epsilon(target_delta=target_delta))

    if dropped.messages and warn:
        warnings.warn(
            f"The Renyi accountant dropped {len(dropped.messages)} order(s) "
            "it could not compute at these parameters, so this epsilon is "
            "an upper bound looser than RDP can give. It is still a valid "
            "bound - epsilon is a minimum over orders. Small noise "
            f"multipliers are the usual cause. First: {dropped.messages[0]}",
            stacklevel=2,
        )
    return epsilon


def _epsilon(sampling_probability: float, noise_multiplier: float,
             num_compositions: int, target_delta: float,
             method: Method) -> float:
    """Compose the subsampled Gaussian event and read off epsilon."""
    return _composed(
        [PoissonGaussianSchedule(sampling_probability, noise_multiplier,
                                 num_compositions)],
        target_delta, method, qualify=False,
    )


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
        ``"rdp"`` (default) or ``"pld"``; ADR-0011 records why. PLD is
        tighter for this mechanism by an amount that depends strongly on
        the sampling rate and the budget, so the two are not
        interchangeable in a reported result. Hold it fixed across a
        comparison and report which was used.

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


def calibrate_noise_multiplier(
        releases_from_multiplier: Callable[
            [float], Sequence[PoissonGaussianSchedule]],
        target_epsilon: float, target_delta: float, *,
        method: Method = "rdp") -> float:
    """The smallest shared noise multiplier meeting a privacy budget.

    The other direction of this module: `composed_poisson_gaussian_epsilon`
    prices a run, this picks the parameter that makes a run affordable.

    Takes a *builder* rather than a rate and a count, because the
    schedules a multiplier has to cover are not always one. Given a
    candidate multiplier it must return the schedules a run would have
    at that multiplier; the search then bisects over the composed cost
    of all of them at once. `dimma.accounting.spiderboost` returns two,
    both carrying the same multiplier, which is why one scalar suffices
    there.

    Parameters
    ----------
    releases_from_multiplier
        ``multiplier -> releases``. Called many times during the search,
        so it should be cheap and free of side effects. The schedules
        it returns may vary in rate and count but the returned
        multiplier is the one it was handed.
    target_epsilon
        The epsilon to spend. The result meets it, so epsilon at the
        returned multiplier is at or just under this.
    target_delta
        The ``delta`` the budget is stated at.
    method
        ``"rdp"`` (default) or ``"pld"``; ADR-0011 records why, and
        what the choice costs. Calibrate and report under the same one.

    Returns
    -------
    noise_multiplier : float
        The multiplier to build the run's noise scales from.

    Raises
    ------
    ValueError
        If the budget is not positive, if it is so loose that no
        multiplier is meaningfully the smallest sufficient one, or if
        no multiplier meets it at all. That last is reachable under
        ``"rdp"`` at a very small target, for the reason the error
        gives: epsilon there floors above zero.

    Warns
    -----
    UserWarning
        As `composed_poisson_gaussian_epsilon`, and checked **at the
        returned multiplier only**. The search passes through small
        multipliers where the Renyi accountant routinely drops orders;
        warning from those would report a problem the answer does not
        have.
    """
    if not target_epsilon > 0.0:
        raise ValueError(f"target_epsilon={target_epsilon} must be positive.")
    if not 0.0 < target_delta < 1.0:
        raise ValueError(f"target_delta={target_delta} must be in (0, 1).")
    if method not in ("rdp", "pld"):
        raise ValueError(f"method={method!r} must be 'rdp' or 'pld'.")

    def _epsilon_at(multiplier: float) -> float:
        return _composed(releases_from_multiplier(multiplier), target_delta,
                         method, qualify=True, warn=False)

    def _charged(multiplier: float) -> list[PoissonGaussianSchedule]:
        releases = releases_from_multiplier(multiplier)
        _validate(releases, target_delta, method)
        charged = [r for r in releases if r.num_compositions > 0]
        if not charged:
            raise ValueError(
                "releases_from_multiplier described no releases, so every "
                "multiplier meets the budget and none is smallest."
            )
        return charged

    def event_from_multiplier(multiplier: float) -> dp_event.DpEvent:
        return dp_event.ComposedDpEvent([
            dp_event.SelfComposedDpEvent(
                dp_event.PoissonSampledDpEvent(
                    sampling_probability=r.sampling_probability,
                    event=dp_event.GaussianDpEvent(
                        noise_multiplier=r.noise_multiplier),
                ),
                r.num_compositions,
            )
            for r in _charged(multiplier)
        ])

    # Quiet during the search: it probes small multipliers where the
    # Renyi accountant drops orders, and those probes are not results.
    with _watching_rdp_orders():
        _charged(_SEARCH_INITIAL_GUESS)  # reject an empty schedule up front
        lower, upper = _bracket(_epsilon_at, target_epsilon, target_delta)
        multiplier = float(calibrate_dp_mechanism(
            lambda: _accountant(method),
            event_from_multiplier,
            target_epsilon,
            target_delta,
            mechanism_calibration.ExplicitBracketInterval(lower, upper),
        ))

    # Re-price at the answer, with the warning channel live, so a
    # dropped order is reported when it applies to the number returned.
    composed_poisson_gaussian_epsilon(
        releases_from_multiplier(multiplier), target_delta, method=method)
    return multiplier
