"""Stage 7 - optimization.

Update parameters from the privatized gradient. dimma delegates
optimizers to `optax`, as it delegates accounting to `dp-accounting`.
Pinning a private method and its baseline to the same `optax.adam` is
what keeps that comparison controlled.

The optimizer itself comes from optax, named at the call site:

    import optax
    from dimma.core import updates

    opt = optax.adam(optax.cosine_decay_schedule(lr, decay_steps=T))
    state = updates.init(opt, params)
    params, state = updates.apply(opt, params, grad, state)

dimma does not wrap or re-export optax's optimizers. Which optimizer a
run used is part of what makes a comparison reproducible, so it is
better read from the caller's own import than from a dimma alias.

Steps, not epochs. A DP algorithm's privacy cost composes over
optimizer steps, so the step count is what must be held fixed when two
methods are compared. optax agrees by construction: its schedules index
on `update` calls. A schedule's horizon and the privacy horizon are the
same run, so they take the same number:

    schedule = cosine_decay_schedule(init_value=lr, decay_steps=T)

Optimizer state derived from a privatized gradient is post-processing
and carries no additional privacy cost.
"""

from __future__ import annotations

from typing import Any

from optax import GradientTransformation, OptState, Schedule, apply_updates

__all__ = [
    "GradientTransformation",
    "OptState",
    "Schedule",
    "apply_updates",
    "init",
    "apply",
]


def init(optimizer: GradientTransformation, params: Any) -> OptState:
    """Build the initial optimizer state, to thread through the loop."""
    return optimizer.init(params)


def apply(optimizer: GradientTransformation, params: Any, grad: Any,
          opt_state: OptState) -> tuple[Any, OptState]:
    """Apply one optimizer step, returning ``(new_params, new_opt_state)``.

    Collapses optax's ``update`` plus ``apply_updates`` into the single
    operation this stage denotes. Each call advances the counter optax
    schedules index on, which is what keeps a schedule aligned with the
    privacy composition.
    """
    updates, opt_state = optimizer.update(grad, opt_state, params)
    return apply_updates(params, updates), opt_state
