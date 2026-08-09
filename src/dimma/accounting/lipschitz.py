"""The constants Private SpiderBoost assumes, for the model dimma ships.

Produces `spiderboost.noise_scales`'s ``lipschitz_constant`` and
``smoothness_constant`` for `dimma.models.logreg` under
`dimma.models.losses.per_sample_bce_loss`, from an enforced bound on the
feature norm rather than from the data. ADR-0012 records why, and what
the closed forms are; ADR-0009 records what supplying constants the data
exceeds costs.

Assumed, and not checked here
-----------------------------
That every feature vector the run sees is inside the ball of radius
``feature_norm_bound`` — which is what
`dimma.datasets.preprocessing.cap_feature_norms` makes true, and which
nothing else does. The bound is an argument, so a caller may pass one
larger than the cap they applied, or apply no cap at all; either way the
constants below bound nothing and the epsilon calibrated from them is
false with no crash.

That the model is the one named above. Any other loss, link function or
regularizer has different constants, and this module has no way to tell.
"""

from __future__ import annotations

import math
from typing import NamedTuple


class LipschitzConstants(NamedTuple):
    """The two constants and the step size Theorem B.3 pairs with them.

    ``lipschitz_constant`` and ``smoothness_constant`` are keyword-for-
    keyword what `dimma.accounting.spiderboost.noise_scales` takes;
    ``step_size`` is what the training loop's optimizer must descend
    with for the accountant's number to describe the run.
    """

    lipschitz_constant: float
    smoothness_constant: float
    step_size: float


def logreg_bce_constants(
    feature_norm_bound: float, *, has_bias: bool
) -> LipschitzConstants:
    """The triple implied by a feature-norm bound. Takes no data.

    Parameters
    ----------
    feature_norm_bound : float
        ``R``: the ``l_2`` bound every feature vector satisfies, as
        enforced by
        `dimma.datasets.preprocessing.cap_feature_norms`, not as
        measured on the data.
    has_bias : bool
        Whether the model carries a bias, which augments the feature
        vector with a constant 1 and so raises the squared norm by one.
        Keyword-only and without a default: at ``R = 1`` it is worth a
        factor of two in the smoothness constant and in the step size,
        and a wrong one reports an epsilon that is simply false.

    Returns
    -------
    constants : LipschitzConstants

    Raises
    ------
    ValueError
        If ``feature_norm_bound`` is not finite and positive.
    """
    if not math.isfinite(feature_norm_bound) or feature_norm_bound <= 0.0:
        raise ValueError(
            f"feature_norm_bound={feature_norm_bound} must be finite and "
            f"positive."
        )

    augmented_squared = feature_norm_bound**2 + (1.0 if has_bias else 0.0)
    smoothness = augmented_squared / 4.0
    return LipschitzConstants(
        lipschitz_constant=math.sqrt(augmented_squared),
        smoothness_constant=smoothness,
        step_size=1.0 / (2.0 * smoothness),
    )
