"""Stage 7 - optimization.

Update parameters from the privatized gradient. dimma delegates
optimizers to `optax`, as it delegates accounting to `dp-accounting`.
Pinning a private method and its baseline to the same `optax.adam` is
what keeps that comparison controlled.

Every optax optimizer and schedule is reachable from here, so callers
never import optax directly:

    from dimma.core.updates import adam, cosine_decay_schedule

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

import optax
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


def __getattr__(name: str) -> Any:
    """Expose optax's optimizers and schedules under this module.

    Delegation rather than a re-export list, so the surface cannot drift
    as optax adds optimizers.
    """
    try:
        return getattr(optax, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r} "
            f"(and neither does optax)"
        ) from None


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(dir(optax)))
