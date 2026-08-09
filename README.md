# dimma

A JAX based library for differentially private optimization, built so that every
algorithm is assembled from the same primitives.

## The pipeline

Essentially every DP-SGD method has the same seven stages. dimma's `core` layer
is organized around them:

1. **Batch generation** — sample the data under a specific private methodology.
2. **Forward pass** — push the batch through the model.
3. **Per-sample gradients** — one gradient per individual example.
4. **Clipping** — bound each per-sample gradient's norm, bounding sensitivity.
5. **Aggregation** — sum or average the clipped gradients.
6. **Perturbation** — add noise calibrated to the privacy budget.
7. **Optimization** — update parameters from the privatized gradient.

dimma implements stages 1 and 3 through 7. Stages 1 and 3 through 6 are the ones
that turn a gradient into a private one; stage 7 is the update rule the
algorithms descend with, which their papers state and which is a few lines of
pytree arithmetic. Stage 2 is the caller's model, which runs inside the
per-sample loss they supply. Adam, which we use for a baseline, is named at the
call site from `optax` and passes through the same seam, so a private method and
its baseline can still be pinned to the same optimizer. `core` still names all
seven, because the stage an algorithm does not choose is as much a part of its
description as the ones it does.

A non-private method is the same pipeline with stages 4 and 6 dropped, stage 3 changed into per-batch gradient and stage 1 relaxed to ordinary sampling. That is deliberate: it is what makes a private algorithm and its non-private counterpart comparable rather than merely adjacent, and it is why the baselines live inside the library instead of in a scripts folder next to it.

## Scope

dimma covers four fronts, all sharing the pipeline above.

**Non-classical DP optimizers.** The primary front, and the reason the library exists: differentially private methods that are not classical DP-SGD — variance-reduced, second-order, sparsity-aware, and others as they are implemented. Private SpiderBoost (Arora et al., ICML 2023) is the first.

**Classical DP-SGD.** It is the reference every non-classical method is measured against, so it gets the same primitives, the same tests, and the same documentation as anything else.

**Non-private baselines.** SGD, Adam, non-private SpiderBoost. Each is the
privacy-free counterpart of something dimma implements privately, built from the
same stages, so that "what did privacy cost here" is a controlled question.

**Transforms.** Changes to a quantity that are not algorithms and compose across
them — for example the ℓ₁-ball projection applied to an already-privatized
gradient as post-processing. Transforms are a separate axis from algorithms: one
transform can apply to several algorithms, and an algorithm can use several. Not
"mechanisms", which the literature reserves for a complete map carrying its own
privacy analysis; a transform carries none.

Three layers sit underneath all four.

**Core.** Architecture-agnostic pytree math that names no algorithm, model, or
dataset and makes no privacy claim. Something enters `core` only if it implements
one of the stages, or is stage-independent math with at least two consumers in
different modules.

**Accounting.** Calibrating noise to a target `(ε, δ)`, and computing the `ε` a
run spent. Standard mechanisms use Google's `dp-accounting`; the non-classical
methods this library exists for usually fall outside its assumptions or are
bounded too loosely by it, so those ship their own accountant alongside the
algorithm.

**Training.** Reference models, losses, dataset loaders, and the loops that run
an algorithm end to end — so nothing has to be assembled before a method can be
run once. Loops take a caller-supplied per-sample loss over arbitrary pytrees,
and the algorithms never import the models.

## Layout

`core`, the dataset loaders, the reference model, and the first two algorithms;
the rest is being ported in.

```
src/dimma/
├── accounting/              where the privacy claims live
│   ├── lipschitz.py         L₀, L₁ and the step size, from an enforced bound
│   └── sampling.py          subsampled-Gaussian ε, via dp-accounting
├── algorithms/              one package per algorithm
│   ├── dp_sgd/               classical DP-SGD (Abadi et al., 2016)
│   │   ├── step.py             one iteration; the privatized gradient
│   │   └── train.py            the loop, and stage 1
│   └── spiderboost/          Private SpiderBoost (Arora et al., 2023)
│       ├── step.py             two mechanisms; a release each, and its apply
│       └── train.py            the loop, stage 1, and the output rule
├── core/                    the pipeline stages
│   ├── sampling/            stage 1 — one module per mechanism
│   │   ├── poisson.py         the standard one; raises on an oversize draw
│   │   └── poisson_truncated.py   modified mechanism, unaccounted
│   ├── gradients.py         stage 3 — per-sample and batch gradients
│   ├── clipping.py          stage 4
│   ├── aggregation.py       stage 5 — sum/average, Poisson masking
│   ├── noise.py             stage 6 — Gaussian and Laplace
│   ├── updates.py           stage 7 — sgd, and the seam optax also fits
│   ├── pytree.py            pytree vector-space ops
│   └── projection.py        ℓ₁-ball geometry
├── datasets/                loaders; no algorithm imports these
│   ├── criteo.py             Criteo 1M — columns × preprocessing, four ways
│   └── preprocessing.py      maps loaders compose; the per-record norm cap
└── models/                  reference models; no algorithm imports these
    ├── logreg.py             logistic regression — one linear layer
    └── losses.py             sigmoid BCE, per-sample and batch

tests/                       mirrors src/dimma/
docs/agents/                 agent-facing context, not published
```

## Documentation

Narrative documentation is built with MkDocs and covers the conceptual
foundations of DP-SGD, the JAX tooling the library is built on, the library's
own structure, and — per algorithm — where the paper's theory and the
implementation diverge. The MkDocs source is the published site; `docs/` is
agent-facing context and is not published.

## Status

Pre-`0.1.0`, and empty by design: this repository is a deliberate re-cut of an
earlier `dimma` whose documentation had grown around a single algorithm.
Implementation is being ported here selectively. The public API is unstable
until `1.0`.
