# Core stages

The architecture-agnostic primitives of the
[seven-stage pipeline](../library/pipeline.md). Import from the stage
module, so the import line names the stage.

## Stage 1 — sampling

::: dimma.core.sampling.poisson

::: dimma.core.sampling.poisson_truncated

The other two samplers — `shuffled` (ordinary epochs, for the baselines)
and `dyadic` (the fixed-size draw at a randomly drawn scale that
[bias-reduced sparse SGD](../algorithms/bias-reduced-sparse-sgd.md)
uses) — are read in the source:
[`src/dimma/core/sampling/`](https://github.com/Larraguibel/dimma_lib/tree/main/src/dimma/core/sampling).

## Stage 3 — gradients

::: dimma.core.gradients

## Stage 4 — clipping

::: dimma.core.clipping

## Stage 5 — aggregation

::: dimma.core.aggregation

## Stage 6 — noise

::: dimma.core.noise

## Stage 7 — updates

The optimizer seam — the `Optimizer` pair, `updates.sgd`, `init`, and
`apply` — is described on the [pytrees page](../pytrees.md) and read in
the source:
[`src/dimma/core/updates.py`](https://github.com/Larraguibel/dimma_lib/blob/main/src/dimma/core/updates.py).

## Stage-independent math

::: dimma.core.pytree

::: dimma.core.projection
