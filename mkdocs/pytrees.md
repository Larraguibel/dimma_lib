# Working with pytrees

!!! note

    This page covers the JAX idiom dimma is written in. It assumes you
    already know what per-sample gradients, clipping, and the Gaussian
    mechanism are; if the applied side of that is new, read
    [DP optimization in practice](dp-optimization-in-practice.md) first.

## Why JAX for this library

DP-SGD variants need direct control over primitives that high-level
libraries hide or fix. The libraries that do focus on DP-SGD — Opacus,
Keras-DP, `optax.contrib.dpsgd` — all assume the same rigid pipeline:
batch sampling → per-sample gradients → clip to a single norm `C` →
average → Gaussian noise → optimizer update. Anything outside that shape
is awkward or impossible to express.

dimma's algorithms break that shape.
[Private SpiderBoost](algorithms/spiderboost.md) alternates two kinds of
step in one loop — anchor steps releasing a privatized gradient, variation
steps releasing the privatized *difference* of gradients across
consecutive iterates, with noise scaled to that difference. Two update
rules, two noise scales, one loop.
[Bias-reduced sparse SGD](algorithms/bias-reduced-sparse-sgd.md) goes
further: each step draws a batch whose size is `2^(N+1)` for a *randomly
drawn* scale `N`, releases four private means from it, and stops the loop
where a privacy filter says so. No premade batch-handling function exposes
the inside of batch construction, and no Opacus hook lets you in deep
enough to inject any of this cleanly.

JAX exposes the primitives directly: per-sample gradients, custom batch
handling, explicit randomness, and a training loop dimma writes itself.

## The four primitives you actually need

If you understand `grad`, `vmap`, `jit`, and `random`, you have enough JAX
to implement any DP-SGD variant. The rest is library convenience.

### `jax.grad` — scalar-output autodiff

Differentiates a function returning a scalar with respect to its first
argument. Unlike PyTorch's `.backward()`, which mutates `.grad` attributes
as a side effect, `jax.grad` returns the gradient as a value. Code becomes
a chain of pure functions, not a graph of mutable tensors.

```python
def loss(params, x, y):
    return jnp.mean((forward(params, x) - y) ** 2)

grad_fn = jax.grad(loss)               # (params, x, y) -> grad
g = grad_fn(params, x_batch, y_batch)  # mean gradient over the batch
```

Note this gives the **mean (or sum) gradient over the batch**, not
per-sample gradients. That is what `vmap` is for.

### `jax.vmap` — the per-sample gradient trick

This is *the* reason JAX is the natural fit for DP-SGD. `vmap` vectorizes
a function over an extra axis without a Python loop and without
retracing:

```python
def per_sample_loss(params, x, y):
    # x, y are SINGLE examples, no batch dimension
    return loss_on_one_example(params, x, y)

per_sample_grad = jax.vmap(jax.grad(per_sample_loss), in_axes=(None, 0, 0))
# in_axes=(None, 0, 0): params shared, x and y batched on axis 0

g_per_sample = per_sample_grad(params, x_batch, y_batch)
# a pytree with a leading batch dim B on every leaf
```

The composition `vmap(grad(...))` is the canonical DP-SGD pattern: each
example's gradient computed independently, returned as a pytree of shape
`(B, *param_shape)` per parameter, ready to clip per-sample, aggregate,
and noise. In dimma this composition is
`dimma.core.gradients.per_sample_grads`, and the leading-batch-axis layout
it returns is exactly what stages 4 and 5 expect. Under XLA it is fused
and roughly as fast as one batched forward/backward — dramatically faster
than a Python loop over samples, and free of the "expanded weights"
machinery Opacus needs in PyTorch.

### `jax.jit` — compile to XLA

Wraps a function so its first call traces and compiles it; later calls
with the *same input shapes and dtypes* hit the compiled version.

The gotcha that matters for DP: **tracing is shape-specialized**. If the
batch size changes across iterations, `jit` retraces every time, which
kills performance — and that is exactly the situation under Poisson
subsampling, where the realized batch size is `Binomial(n, p)`.

dimma's answer is the padding cap: `dimma.core.sampling.poisson` pads
every draw out to a fixed length `b_max` and returns a mask marking the
real entries, which `dimma.core.aggregation` applies when summing. The cap
is a memory bound, never a privacy parameter — and if a draw exceeds it,
the sampler *raises rather than truncates*. Discarding examples that were
genuinely drawn would make inclusion dependent across examples, and
independence is precisely what subsampling amplification assumes; the
standard accounting would keep returning a number for a mechanism that no
longer ran. A separate module, `poisson_truncated`, exists for callers who
accept that trade — kept separate so the choice is visible in the import
line.

### `jax.random` — explicit PRNG keys

JAX has no global random state. Every random operation takes an explicit
`key`, and keys must be split before reuse:

```python
key = jax.random.key(0)
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape=(d,))
```

For DP this is a feature, not a chore. The noise added to a gradient *is*
the privacy mechanism, so having the key as an explicit argument means
noise generation is exactly reproducible, the same noise cannot be
accidentally reused across steps (which would break the guarantee), and
the noise stream is cleanly isolated from the sampling stream. dimma's
loops carry two sources for exactly that reason: a JAX `key` for the
device-side noise and a NumPy `rng` for host-side batch sampling — a
Poisson draw has data-dependent cardinality and cannot be compiled, so
stage 1 lives host-side in `train` while everything from stage 3 on
compiles.

## Models are plain pytrees

JAX transforms want pure functions over pytrees. Rather than adopting a
model framework and converting its objects to pytrees at the boundary,
dimma keeps the boundary from existing: a model here *is* a pytree.

The contract has two functions: `init_params`, returning a plain pytree of
parameters (for logistic regression, a dict of two arrays), and a
`forward` mapping one feature vector to one scalar. The caller's
contribution to the pipeline is the **per-sample loss** — the loss of a
single example under a single set of parameters — and the model appears
only inside it:

```python
from dimma.models.logreg import init_params, forward
from dimma.models.losses import per_sample_bce_loss

params = init_params(jax.random.key(0), num_features=39)
# per_sample_bce_loss(params, x, y) -> scalar, forward() inside
```

Because `per_sample_bce_loss` is already a pure function of
`(params, x, y)` with no batch dimension, `vmap(grad(...))` applies to it
directly — no split/merge step, no filtering trainable state from buffers,
no framework-version drift. Any model you can write as `init_params` plus
a pure `forward` over a pytree slots into every algorithm in the library;
the training loops never see a model object, only the loss.

The price is that dimma ships no layers: a model with structure is yours
to write as functions. For the reference workloads — logistic regression
on Criteo — that price is one file, `dimma/models/logreg.py`, and code
examples in this documentation never import a model framework.

## Optax, and dimma's optimizer seam

Optax's mental model: an optimizer takes a *single* gradient pytree and
returns an update. dimma's stage 7 seam, `dimma.core.updates.Optimizer`,
deliberately matches optax's `(init, update)` pair — structurally, not
nominally — so one seam carries both.

**dimma implements a rule itself when a paper implemented here states it.**
`updates.sgd` is Algorithm 1's `θ − η·g̃` — two lines of pytree
arithmetic, short enough to read against the paper, and a run departing
from it is a different algorithm rather than a differently configured one.
**Everything else is named from optax at the call site.** Adam is the
instructive case: bias correction, where the epsilon sits, coupled versus
decoupled decay are wrong *quietly*, and a baseline that is subtly wrong
does not lose visibly — it makes every comparison against it meaningless.
Borrowing the reference implementation, through the same seam, keeps both
sides of a comparison pinned to the same optimizer.

**Do not use optax on the per-sample side.** Anything touching per-sample
gradients — clipping, noise calibration, variance-reduced estimators — is
written from `dimma.core` primitives. Optax transforms assume a single
aggregated gradient and cannot see per-sample structure. That is also why
dimma does not build on `optax.contrib.dpsgd`: it implements the standard
pipeline, and forecloses exactly the variations this library exists to
study.

Because the seam is structural, a wrapper can interpose on *any*
algorithm's stage 7 without the loop knowing — which is what the
[transforms](transforms/index.md) layer does.

## A minimal end-to-end DP-SGD step

The skeleton every algorithm in the library specializes, written in
dimma's own primitives — stages 3 through 7 of classical DP-SGD, one
compiled call:

```python
import jax
from functools import partial

from dimma.core import aggregation, clipping, gradients, noise, pytree, updates
from dimma.models.losses import per_sample_bce_loss

per_sample_grad_fn = gradients.per_sample_grads(per_sample_bce_loss)
optimizer = updates.sgd(0.1)

@partial(jax.jit, static_argnames=())
def dp_sgd_step(params, opt_state, key, x_batch, y_batch, mask,
                clip_norm, noise_multiplier, expected_batch_size):
    # 3. one gradient per example: leading batch dim on every leaf
    g_ps = per_sample_grad_fn(params, x_batch, y_batch)
    # 4. clip each to the clipping norm
    g_ps = clipping.per_sample_clip(g_ps, clip_norm)
    # 5. sum, masking the padded slots of the Poisson draw
    g_sum = aggregation.sum_over_batch(g_ps, mask)
    # 6. Gaussian noise on the SUM: the sensitivity bound belongs to it
    g_sum = noise.add_gaussian(g_sum, key, noise_multiplier * clip_norm)
    # scale by the expected batch size — a divisor must never depend
    # on the data, and under Poisson draws the realized count does
    estimate = pytree.scale(g_sum, 1.0 / expected_batch_size)
    # 7. descend along the privatized gradient
    return updates.apply(optimizer, params, estimate, opt_state)
```

Stage 1 sits outside, host-side:
`dimma.core.sampling.poisson.subsample(rng, n, p, b_max)` returns the
padded indices and the mask. The real thing — with the release/apply
split, RNG threading, and tests — is `dimma.algorithms.dp_sgd`; this
skeleton is the contract, and every variant deviates from named parts of
it.

!!! warning

    **Conventions to pin before connecting to an accountant.** dimma's
    DP-SGD noises the **sum** at scale `noise_multiplier * clip_norm`;
    many papers write the mechanism on the mean, and Private SpiderBoost
    noises the *released estimate* — each is written the way its paper
    writes it, and an accountant must match the convention of the code
    that ran. Likewise the divisor: dividing by the *expected* batch size
    is not a stylistic choice — a divisor that depends on the realized
    draw depends on the data.

## Pointers

- JAX: [docs.jax.dev](https://docs.jax.dev)
- Optax: [optax.readthedocs.io](https://optax.readthedocs.io)
- The stages these primitives implement:
  [the seven-stage pipeline](library/pipeline.md)
- A concrete algorithm using all of them: [DP-SGD](algorithms/dp-sgd.md)
