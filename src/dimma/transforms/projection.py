"""The ``l_1`` projection, applied at the optimizer seam.

`dimma.core.projection` is the geometry; this module is the layer that
applies it. `l1_projected` wraps an `Optimizer` so every update lands
inside one global ``l_1`` ball, and because every training loop in
`dimma.algorithms` takes its optimizer through the same seam, one
wrapper serves all of them — the caller writes::

    optimizer = l1_projected(updates.sgd(0.1), radius=5.0)

and passes it to any `train`, none of which know it is there.

Makes no privacy claim. Whether the projection is free in a given run
is a statement about the mechanism it post-processes, stated where
that run's accounting is stated. A projected update rule is also a
departure from any paper whose rule does not project — Abadi et al.'s
Algorithm 1 does not — and, as with any other choice at stage 7, that
departure is the caller's to make and to report.
"""

from __future__ import annotations

import jax

from dimma.core import projection, pytree, updates

__all__ = [
    "l1_projected",
]


def l1_projected(
    optimizer: updates.Optimizer, radius: float | jax.Array
) -> updates.Optimizer:
    """Constrain ``optimizer`` so every update lands in the ``l_1`` ball.

    Runs the wrapped rule, adds its increment to the current
    parameters, projects the result onto ``{w : ‖w‖_1 <= radius}`` —
    one ball, global across every leaf — and re-expresses the projected
    point as the increment `updates.apply` will add back.

    Parameters
    ----------
    optimizer
        The rule to constrain. Anything satisfying `updates.Optimizer`'s
        structural contract, so an optax transformation wraps the same
        way `updates.sgd` does.
    radius
        The ball's radius. May be traced; a concrete negative value is
        rejected here rather than projecting onto an empty set.

    Returns
    -------
    updates.Optimizer
        Carrying the wrapped rule's ``init`` and state unchanged: the
        wrapper adds no state of its own, so a schedule keeps indexing
        the same count.

    Notes
    -----
    The constraint holds to floating-point round-off rather than
    bit-exactly: the seam traffics in increments, so the projected
    point is re-derived as ``projected - params`` and added back, and
    each of those rounds once.

    The wrapped ``update`` requires the current parameters.
    `updates.apply` always passes them; a bare ``update(estimate,
    state)`` call raises rather than silently skipping the projection.
    """
    if isinstance(radius, (int, float)) and not isinstance(radius, bool):
        if radius < 0:
            raise ValueError(
                f"radius={radius} must be non-negative; no point lies "
                f"inside a ball of negative radius."
            )

    def update_fn(
        estimate, state, params=None
    ) -> tuple[object, updates.OptState]:
        if params is None:
            raise ValueError(
                "l1_projected needs the current parameters to project "
                "the stepped point; update was called with params=None."
            )
        increment, new_state = optimizer.update(estimate, state, params)
        stepped = pytree.add(params, increment)
        projected = projection.project_l1_ball_pytree(stepped, radius)
        return pytree.sub(projected, params), new_state

    return updates.Optimizer(optimizer.init, update_fn)
