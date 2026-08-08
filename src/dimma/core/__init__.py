"""The architecture-agnostic math the pipeline is built from.

Membership rule
---------------
`core` is architecture-agnostic pytree math that names no algorithm,
model, or dataset and makes no privacy claim. Something enters `core`
only if it implements one of the seven pipeline stages, or is
stage-independent math with at least two consumers in different
modules.

Two questions decide it. *Which stage is this?* If none: *who else uses
it?* If neither has an answer, it belongs with its consumer.

Describing what a sampler samples, or which accounting assumption a
primitive satisfies, is factual and allowed. Computing an epsilon,
calibrating a scale to a budget, or calling a transformation free are
claims, and belong to the accounting and transform layers.

The stages
----------
=====  ======================  =============================
Stage  What it does            Where it lives
=====  ======================  =============================
1      Batch generation        `dimma.core.sampling`
2      Forward pass            the caller's model
3      Per-sample gradients    `dimma.core.gradients`
4      Clipping                `dimma.core.clipping`
5      Aggregation             `dimma.core.aggregation`
6      Perturbation            `dimma.core.noise`
7      Optimization            `dimma.core.updates` (optax)
=====  ======================  =============================

Stages 2 and 7 are the caller's, the model and the update rule. dimma
owns the five in between, which are the ones that turn a gradient into
a private one. That split is deliberate.

Implementing no stage: `dimma.core.pytree` (vector-space operations on
pytrees) and `dimma.core.projection` (``l_1``-ball geometry). Both are
admitted by the second half of the rule, and both are closed sets.

Imports
-------
No functions are re-exported. Import from the stage module, so the
import line says which stage the call belongs to:

    from dimma.core.clipping import per_sample_clip
"""

from dimma.core import (
    aggregation,
    clipping,
    gradients,
    noise,
    projection,
    pytree,
    sampling,
    updates,
)

__all__ = [
    "aggregation",
    "clipping",
    "gradients",
    "noise",
    "projection",
    "pytree",
    "sampling",
    "updates",
]
