# dimma

A JAX library for differentially private optimization, built so that every
algorithm is assembled from the same primitives.

Essentially every method in the DP-SGD family does the same seven things —
sample, forward, per-sample gradients, clip, aggregate, perturb, update —
and differs only in what it chooses at each. dimma turns that observation
into architecture: the [seven-stage pipeline](library/pipeline.md) is
implemented once, and an algorithm is a choice at each stage rather than a
from-scratch training loop. Two methods become comparable stage by stage
instead of merely adjacent, and a method and its privacy-free counterpart —
which dimma ships, [built from the same stages](algorithms/baselines.md) —
differ in the privacy and nothing else.

The library covers seven fronts. Four of them are the optimizers
themselves:

- **Non-classical DP optimizers** — the reason the library exists:
  private methods beyond classical DP-SGD.
  [Private SpiderBoost](algorithms/spiderboost.md) (Arora et al., ICML 2023)
  and [bias-reduced sparse SGD](algorithms/bias-reduced-sparse-sgd.md)
  (Ghazi et al., NeurIPS 2024) are the first two.
- **Classical [DP-SGD](algorithms/dp-sgd.md)** — the reference every
  non-classical method is measured against, with the same primitives,
  tests, and documentation as anything else.
- **[Non-private baselines](algorithms/baselines.md)** — the privacy-free
  counterpart of everything implemented privately, so that "what did
  privacy cost here" is a controlled question.
- **[Transforms](transforms/index.md)** — changes to a quantity that are
  not algorithms and compose across them, applied at a seam every
  algorithm shares.

The other three sit beside the training loop rather than inside it. No
algorithm imports them, and none of them is a neutral utility — each
carries choices that decide what a reported number means:

- **[Datasets](evaluation-stack/datasets.md)** — loaders whose options
  are independent axes rather than mode names, so that what was done to
  the data is recorded rather than implied. Standardization defaults
  *off*, because on Criteo it is paid for in the privacy budget.
- **[Models](evaluation-stack/models.md)** — one reference model,
  logistic regression, chosen because its Lipschitz and smoothness
  constants are known in closed form, which is what lets a private
  method's preconditions be computed rather than asserted.
- **[Metrics](evaluation-stack/metrics.md)** — strictly proper scores for
  selection, one named cut for reporting, and no ROC-AUC or accuracy
  anywhere.

!!! note

    **Structure of this site.** [Getting started](getting-started.md)
    installs the library and runs one private training end to end.
    [DP optimization in practice](dp-optimization-in-practice.md) covers
    the distinctions applied DP training forces, for readers arriving from
    the theory side. [The library](library/pipeline.md) explains the
    pipeline and the [package map](library/map.md);
    [working with pytrees](pytrees.md) covers the JAX idiom everything is
    written in. Each [algorithm](algorithms/index.md) has one page, each
    [transform](transforms/index.md) likewise. The evaluation stack —
    [datasets](evaluation-stack/datasets.md),
    [models](evaluation-stack/models.md),
    [metrics](evaluation-stack/metrics.md) — has a page each for the
    choices behind it, and [evaluation on Criteo](evaluation.md) collects
    what the executed comparisons found. The
    [API reference](reference/index.md) documents the people-facing
    surface.

## Status

Pre-release: nothing is tagged and the public API is unstable until `1.0`.
This library is a deliberate re-cut of an earlier project of the same name,
whose documentation had grown around a single algorithm; implementation is
ported in selectively, and the [package map](library/map.md) shows what has
landed.
