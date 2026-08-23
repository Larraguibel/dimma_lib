# Getting started

## Install

Not on PyPI yet; install from a clone. Python 3.12 or newer.

```bash
git clone https://github.com/Larraguibel/dimma_lib
cd dimma_lib
pip install -e .
```

The base install runs JAX on CPU. Extras: `[gpu]` for the CUDA build,
`[datasets]` for the Criteo loader, `[notebooks]` for the notebook stack,
`[dev]` for tests, `[docs]` to build this site.

## One private run, end to end

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

Three things about this snippet carry the library's shape:

**The caller supplies the model, as a per-sample loss.**
`per_sample_bce_loss` maps one example and one set of parameters to one
scalar. That function is the only place a model appears; the training loop
never sees a model object. dimma's reference models are plain functions
over plain pytrees — see [working with pytrees](pytrees.md).

**The loop returns parameters and nothing else.** Evaluation and the
epsilon claim both belong to the caller: evaluating on the training data is
another access to it, spending budget the algorithm does not account for,
so that call sits at the call site where it is visible rather than in a
callback inside the private loop.

**The accountant is a separate call, and it states its assumptions.**
`poisson_gaussian_epsilon` prices a specific mechanism — Poisson-subsampled
Gaussian releases at the rate and multiplier you name. The number is a
guarantee only if the run matched that mechanism; a loop that departs from
it silently invalidates the number. Why claims and primitives live in
different packages is explained in the
[package map](library/map.md).

## Where to next

- Arriving from the theory side:
  [DP optimization in practice](dp-optimization-in-practice.md).
- The design in one page: [the seven-stage pipeline](library/pipeline.md).
- Straight to an algorithm: [DP-SGD](algorithms/dp-sgd.md),
  [Private SpiderBoost](algorithms/spiderboost.md),
  [bias-reduced sparse SGD](algorithms/bias-reduced-sparse-sgd.md).
