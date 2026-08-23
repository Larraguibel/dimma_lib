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

dimma implements stages 1 and 3 through 7; stage 2 is the caller's model, run
inside the per-sample loss they supply. `core` names all seven, because the
stage an algorithm does not choose is as much a part of its description as the
ones it does.

A non-private method is the same pipeline with clipping and noise dropped,
per-sample gradients relaxed to a batch gradient, and sampling relaxed to
ordinary epochs. Building baselines from the same stages is what makes a
private algorithm and its non-private counterpart comparable rather than
merely adjacent.

## Scope

dimma covers four fronts, all sharing the pipeline above.

**Non-classical DP optimizers.** The primary front, and the reason the library
exists: differentially private methods that are not classical DP-SGD —
variance-reduced, second-order, sparsity-aware, and others as they are
implemented. Private SpiderBoost (Arora et al., ICML 2023) is the first;
bias-reduced sparse SGD (Ghazi et al., NeurIPS 2024) is the second.

**Classical DP-SGD.** The reference every non-classical method is measured
against, so it gets the same primitives, tests, and documentation as anything
else.

**Non-private baselines.** SGD, Adam, non-private SpiderBoost. Each is the
privacy-free counterpart of something dimma implements privately, built from
the same stages, so that "what did privacy cost here" is a controlled question.

**Transforms.** Post-processing that composes across algorithms — for example
the ℓ₁-ball projection applied to an already-privatized gradient. One transform
can apply to several algorithms, and an algorithm can use several.

Three layers sit underneath all four.

**Core.** Architecture-agnostic pytree math that names no algorithm, model, or
dataset and makes no privacy claim.

**Accounting.** Calibrating noise to a target `(ε, δ)`, and computing the `ε` a
run spent. Standard mechanisms use Google's `dp-accounting`; methods that fall
outside its assumptions ship their own accountant alongside the algorithm.

**Training.** Reference models, losses, dataset loaders, and the loops that run
an algorithm end to end. Loops take a caller-supplied per-sample loss over
arbitrary pytrees, and the algorithms never import the models.

## Getting started

Not on PyPI yet; install from a clone. Python 3.12 or newer.

```
git clone https://github.com/Larraguibel/dimma_lib
cd dimma_lib
pip install -e .
```

The base install runs JAX on CPU. Extras: `[gpu]` for the CUDA build,
`[datasets]` for the Criteo loader, `[notebooks]` for the notebook stack,
`[dev]` for tests.

Train logistic regression with DP-SGD on toy data, then ask the accountant
what it cost:

```python
import jax
import jax.numpy as jnp
import numpy as np

from dimma.accounting.sampling import poisson_gaussian_epsilon
from dimma.algorithms.dp_sgd import train as dp_sgd
from dimma.core import updates
from dimma.models.logreg import init_params
from dimma.models.losses import per_sample_bce_loss

rng = np.random.default_rng(0)
x = jnp.asarray(rng.normal(size=(1000, 20)))
y = jnp.asarray(rng.integers(0, 2, size=1000), dtype=jnp.float32)

params = init_params(jax.random.key(0), num_features=20)
params = dp_sgd.train(
    per_sample_bce_loss, params, updates.sgd(0.1),
    x, y, key=jax.random.key(1), rng=np.random.default_rng(1),
    steps=200, expected_batch_size=100,
    clip_norm=1.0, noise_multiplier=2.0,
)

epsilon = poisson_gaussian_epsilon(
    sampling_probability=100 / 1000, noise_multiplier=2.0,
    num_compositions=200, target_delta=1e-5,
)  # 3.68
```

The loop returns parameters and nothing else. Evaluation and the epsilon
claim both belong to the caller: the first spends privacy budget the
algorithm does not account for, and the second is only valid if the run
matched the mechanism the accountant assumes.

## Layout

Everything below has landed; the rest is being ported in.

```
src/dimma/
├── accounting/              where the privacy claims live
│   ├── bias_reduced_sgd.py  the (ε, δ) privacy filter, in closed form
│   ├── lipschitz.py         L₀, L₁ and the step size, from an enforced bound
│   ├── sampling.py          subsampled-Gaussian ε, via dp-accounting
│   └── spiderboost.py       two subsampled Gaussians, composed
├── algorithms/              one package per algorithm
│   ├── bias_reduced_sgd/     bias-reduced sparse SGD (Ghazi et al., 2024)
│   │   ├── estimators.py       the inner mean estimator, swappable
│   │   ├── step.py             two noise additions; the debiased combination
│   │   └── train.py            the loop the privacy filter stops
│   ├── dp_sgd/               classical DP-SGD (Abadi et al., 2016)
│   │   ├── step.py             one iteration; the privatized gradient
│   │   └── train.py            the loop, and the sampling
│   ├── sgd/                  non-private SGD — DP-SGD's baseline
│   │   ├── step.py             one iteration; no privacy stages
│   │   └── train.py            the loop, and the sampling
│   └── spiderboost/          Private SpiderBoost (Arora et al., 2023)
│       ├── step.py             the two privatized releases, and the update
│       └── train.py            the loop, the sampling, and the output rule
├── core/                    the pipeline stages
│   ├── sampling/            stage 1 — one module per sampler
│   │   ├── dyadic.py          a fixed-size draw at a randomly drawn scale
│   │   ├── poisson.py         the standard one; raises on an oversize draw
│   │   ├── poisson_truncated.py   capped variant; no accountant covers it
│   │   └── shuffled.py        ordinary epochs, nothing private
│   ├── gradients.py         stage 3 — per-sample and batch gradients
│   ├── clipping.py          stage 4
│   ├── aggregation.py       stage 5 — sum/average, Poisson masking
│   ├── noise.py             stage 6 — Gaussian and Laplace
│   ├── updates.py           stage 7 — sgd, and the interface optax also fits
│   ├── pytree.py            pytree vector-space ops
│   └── projection.py        ℓ₁-ball geometry
├── datasets/                loaders; no algorithm imports these
│   ├── base.py               the split type every loader returns
│   ├── criteo.py             the Criteo 1M loader
│   └── preprocessing.py      composable preprocessing; the per-record norm cap
├── metrics/                 evaluation; selection is threshold-free,
│   │                        reporting is not
│   ├── scoring.py            log loss, Brier, normalized entropy
│   ├── calibration.py        reliability curve, ECE, observed/expected
│   ├── decomposition.py      calibration − resolution + uncertainty
│   ├── ranking.py            precision/recall at every cut; area under it
│   └── operating_point.py    the best-F1 cut, and the counts at a cut
├── models/                  reference models; no algorithm imports these
│   ├── logreg.py             logistic regression — one linear layer
│   └── losses.py             sigmoid BCE, per-sample and batch
└── transforms/              post-processing that composes across algorithms
    └── projection.py         the ℓ₁-ball projection of a privatized gradient

tests/                       mirrors src/dimma/, plus integration/
docs/                        ADRs, agent context, research notes; not published
```

## Documentation

Narrative documentation will be published with MkDocs, covering the conceptual
foundations of DP-SGD, the JAX tooling the library is built on, the library's
own structure, and — per algorithm — where the paper's theory and the
implementation diverge. It does not exist yet. Until then, design decisions
are recorded in `docs/adr/` and the domain vocabulary in `CONTEXT.md`; the
rest of `docs/` is agent-facing context and is not published.

Executed runs live in [`notebooks/`](notebooks/README.md): per-algorithm
hyperparameter tuning, and head-to-head comparisons under a shared protocol.

## Status

Pre-release: nothing is tagged and the public API is unstable until `1.0`.
This repository is a deliberate re-cut of an earlier `dimma`; implementation
is still being ported in selectively, and the layout above shows what has
landed.
