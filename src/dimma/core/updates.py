"""Stage 7 - optimization.

Apply an estimate to the parameters. The estimate is whatever stage 6
released: a pytree matching ``params``, formed however the algorithm
above forms it. This stage does not inspect it, so a variance-reduced
or otherwise non-gradient direction passes through unchanged::

    opt = updates.sgd(0.1)
    state = updates.init(opt, params)
    params, state = updates.apply(opt, params, estimate, state)

An `Optimizer` is an ``(init, update)`` pair, structurally optax's
`GradientTransformation`, so an optimizer implemented there threads
through `init` and `apply` unchanged.

Steps, not epochs. An optimizer's state advances once per `apply`
call, so a schedule indexes on the same count a run's length is
measured in.

Makes no privacy claim; ADR-0003.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from dimma.core import pytree

__all__ = [
    "Optimizer",
    "OptState",
    "Schedule",
    "SgdState",
    "sgd",
    "init",
    "apply",
]

OptState = Any
"""Whatever an optimizer carries between steps, shape-invariant across
calls so a loop can thread it through `jax.jit` without retracing."""

Schedule = Callable[[jax.Array], float | jax.Array]
"""``count -> learning rate``, indexed on update calls rather than on
epochs. optax's schedules satisfy it."""


class Optimizer(NamedTuple):
    """An ``(init, update)`` pair, structurally optax's transformation.

    ``update(estimate, state, params=None) -> (increment, new_state)``,
    where ``increment`` is *added* to the parameters. ``params`` is part
    of the protocol rather than a courtesy: an implementation may ignore
    it, as `sgd` does, or require it and raise when it is ``None``, as
    `dimma.transforms.projection.l1_projected` does. `apply` always
    passes it, so a rule needing the current point is safe there and
    only a hand-rolled ``update(estimate, state)`` call can miss it. The
    match with optax is structural, not nominal - an `isinstance` check
    against optax's type fails; ADR-0002.

    Holds functions, so it is a *static* argument to `jax.jit` rather
    than a traced one: bind it with `functools.partial`, or name it in
    ``static_argnums``.

    Attributes
    ----------
    init : callable
        ``params -> OptState``, building the initial state.
    update : callable
        ``(estimate, state, params=None) -> (increment, new_state)``.
    """

    init: Callable[[Any], OptState]
    update: Callable[..., tuple[Any, OptState]]


class SgdState(NamedTuple):
    """`sgd`'s state: a step count and nothing else."""

    count: jax.Array
    """Updates applied so far, shape ``()``, ``int32``. The unit a run's
    length is measured in, and the index a `Schedule` reads."""


def sgd(learning_rate: float | Schedule) -> Optimizer:
    """Descend along the estimate: ``theta <- theta - eta * estimate``.

    No momentum and no state beyond the count. An unused option at
    stage 7 is a way for two runs to differ without the difference
    being reported; ADR-0002 is the test a rule passes to live here.

    Parameters
    ----------
    learning_rate : float > 0, or a `Schedule`
        A constant ``eta``, or a `Schedule` for a step-dependent one.
        A schedule is called with the update count.

    Returns
    -------
    Optimizer
        Whose ``init`` returns a `SgdState` and whose ``update``
        ignores ``params`` and advances the count by one per call.

    Raises
    ------
    ValueError
        If ``learning_rate`` is a constant and not positive. A schedule
        is not checked, since its values are unknown until it is called.
    """
    if not callable(learning_rate) and learning_rate <= 0.0:
        raise ValueError(
            f"learning_rate={learning_rate} must be positive; a "
            f"non-positive step ascends the objective."
        )

    def init_fn(params: Any) -> SgdState:
        del params  # the count does not depend on the pytree
        return SgdState(count=jnp.zeros((), jnp.int32))

    def update_fn(estimate: Any, state: SgdState,
                  params: Any = None) -> tuple[Any, SgdState]:
        del params
        eta = (learning_rate(state.count) if callable(learning_rate)
               else learning_rate)
        return pytree.scale(estimate, -eta), SgdState(count=state.count + 1)

    return Optimizer(init_fn, update_fn)


def init(optimizer: Optimizer, params: Any) -> OptState:
    """Build the initial optimizer state, to thread through the loop."""
    return optimizer.init(params)


def apply(optimizer: Optimizer, params: Any, estimate: Any,
          opt_state: OptState) -> tuple[Any, OptState]:
    """Apply one optimizer step, returning ``(new_params, new_opt_state)``.

    The optimizer supplies a signed increment, so this adds rather than
    subtracts. Each call advances the count the optimizer keeps.
    """
    increment, opt_state = optimizer.update(estimate, opt_state, params)
    return pytree.add(params, increment), opt_state
