"""The architecture-agnostic math the pipeline is built from.

Organized by pipeline stage, one module per stage:

=====  ======================  =============================
Stage  What it does            Where it lives
=====  ======================  =============================
1      Batch generation        `dimma.core.sampling`
2      Forward pass            the caller's model
3      Per-sample gradients    `dimma.core.gradients`
4      Clipping                `dimma.core.clipping`
5      Aggregation             `dimma.core.aggregation`
6      Perturbation            `dimma.core.noise`
7      Optimization            `dimma.core.updates`
=====  ======================  =============================

Stage 2 is the caller's, the model, which runs inside the per-sample
loss they supply; dimma owns the other six. `dimma.core.pytree`
(vector-space operations on pytrees) and `dimma.core.projection`
(``l_1``-ball geometry) implement no stage and are admitted by the
second half of the membership rule.

The rule: something enters `core` only if it implements one of the
seven stages, or is stage-independent math with at least two consumers
in different modules. ADR-0001 is why. Which update rules `updates`
implements and which are named from `optax` at the call site is
ADR-0002; the line between describing what a primitive does and
claiming what it costs is ADR-0003. Nothing is re-exported — import
from the stage module, so the import line says which stage the call
belongs to (ADR-0004)::

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
