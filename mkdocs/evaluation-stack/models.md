# Models

A model in dimma is not part of the pipeline. Stage 2 has no primitive:
the model is the caller's, and it reaches an algorithm only inside a
per-sample loss — the loss of a single example under a single set of
parameters, which is what makes a per-sample gradient definable at all.
Everything an algorithm knows about a model, it knows through that one
function.

`dimma.models` ships one reference model so that a method can be run
without writing a model first, and so that two algorithms compared
against each other are compared on the same one. No algorithm imports
it. Its shape as a pytree — `init_params` plus a `forward`, no framework
objects — is covered in [working with pytrees](../pytrees.md); this page
is about which model, and why that one.

## One linear layer, and nothing else

`dimma.models.logreg` is logistic regression: the logit is
`dot(w, x) + b` for a weight vector `w`, a feature vector `x` of length
`d` (the number of features), and a scalar bias `b`. That is the whole
model.

The smallness is the point. A private run has more ways to go wrong than
a non-private one — a clipping norm that binds too hard, a noise scale
calibrated against the wrong constant, a budget spent before the model
converges — and every one of them is easier to attribute when the
architecture cannot also be the explanation. A result that says
"SpiderBoost and DP-SGD are indistinguishable at ε = 3" is a claim about
the algorithms only if the architecture is held fixed and is simple
enough that neither method can be said to have suited it better.

The model also declines to know what a feature means. `x` is a vector of
length `d`, and nothing in `logreg` asks what its entries are — which
columns were dense, how categoricals were represented, what was
normalized. That stays with the [dataset](datasets.md), where it is
recorded rather than implied.

## Why logistic regression is the one that fits

The choice is not aesthetic; it is forced by what
[Private SpiderBoost](../algorithms/spiderboost.md) assumes. That
algorithm has no clipping stage. It bounds its sensitivity by assumption
instead: the per-sample loss must be `L0`-Lipschitz (its gradients bounded
in norm by `L0`) and `L1`-smooth (its gradients Lipschitz with constant
`L1`), uniformly over every example, with the same constants used to
calibrate the noise. Supply a loss that violates them and nothing
crashes — the run simply reports an ε it did not earn.

Logistic regression with sigmoid binary cross-entropy is the model for
which **both constants are known in closed form**. Writing `σ` for the
sigmoid, `z` for the logit, `y` for the label, and `x̄ = [x; 1]` for the
feature vector augmented with the bias, the per-sample gradient is
`(σ(z) − y)·x̄` and the per-sample Hessian is `σ(z)(1 − σ(z))·x̄x̄ᵀ`. Two
suprema close them — `sup |σ − y| = 1` and `sup σ(1 − σ) = 1/4`, each
holding over *all* parameters and *every* example — so

```
L0 = max ‖x̄‖        L1 = (max ‖x̄‖)² / 4        L1 = L0² / 4
```

and both constants reduce to a single number: the largest augmented
feature norm.

That reduction is the entire gain, and it is not something a linear
predictor supplies on its own. A model with an unbounded link — an
exponential-family model with a log link, whose gradient carries
`exp(z)` — leaves the gradient unbounded in the parameters, and then no
finite `L0` exists at any feature radius at all. What logistic regression
contributes is that the residual and the link's derivative are *both*
bounded uniformly over the parameters; what the linear predictor
contributes is that the only factor left over is the feature vector.

The catch is that the one number everything reduced to is a fact about
the data, and reading it off the data is a query no budget accounts for.
So it is not measured, it is imposed: a per-record cap bounds every
feature vector by `R`, and the constants are then computed from `R` and
from whether the model carries a bias alone — there is no code path from
a feature array to a constant.
[Where the constants come from](../algorithms/spiderboost.md#where-the-constants-come-from)
carries the enforcement, and [datasets](datasets.md#the-norm-cap-goes-last-and-the-ordering-is-load-bearing)
covers where the cap sits in a loader's chain.

This is the whole reason the reference model is the model it is. Another
model would not merely have different constants — it could fail to have
uniform ones at all, leaving the algorithm's precondition asserted rather
than computed.

!!! note

    **The bias is not a detail.** The augmented norm is `‖x‖² + 1`, so at
    `R = 1` a model carrying a bias and one without differ by a factor of
    two in `L1` — and so in every noise scale calibrated from it. dimma's
    constants function takes the bias flag keyword-only and without a
    default, so no call site can leave it to be guessed.

## The loss is the seam

`forward` returns an unsquashed logit; the sigmoid lives in
`dimma.models.losses`, not in the model. Two losses sit there, and they
are not interchangeable:

- `per_sample_bce_loss(params, x, y)` — one example to one scalar. This
  is what an algorithm is handed, and the only thing it differentiates.
- `batch_bce_loss(params, x, y)` — the mean over a batch. A number to
  report at the call site, not something to differentiate.

The split matches a rule the training loops enforce: **loops report no
metrics.** Evaluating a model on the training data is another access to
it, costing budget the algorithm does not account for, so that call
belongs where it is visible rather than in a callback inside a private
loop.

Both losses evaluate the same overflow-safe form,
`max(z, 0) − z·y + log1p(exp(−|z|))`, which is the textbook expression
rearranged so the only exponential taken has a non-positive argument. The
direct form is not merely fragile at the extremes: in float32 it loses the
leading digits of `1 − σ(z)` from `|z|` of about 15 and takes `log(0)`
from about 17 — so it returns a *plausible wrong number* well before it
returns `inf`, and far short of the `|z| ≈ 88` where `exp` itself
overflows. A silently wrong loss is the worst failure mode on a page of
four-decimal comparisons.

## Two forwards, one model

`logreg` exposes a second entry point, `forward_sparse(params, idx, val)`,
which takes the coordinates a row occupies and the values it puts there
instead of a dense vector. It computes exactly what `forward` computes on
the dense row those pairs imply, without materialising it.

This is a change of representation, not of architecture: same parameters,
same model, same logit. It exists because the one-hot
[Criteo](datasets.md) encoding is wide enough that the dense matrix is
itself the obstacle, and because the per-example gradient of a linear
model is `(σ(z) − y)·x̄` — its support is the row's support. With a
one-hot encoding that support is a fixed small number of coordinates per
example, which is the sparsity condition the sparse-DP literature assumes
and which a dense encoding cannot supply. It is what lets
[ℓ₁ projection](../transforms/l1-projection.md) and
[bias-reduced sparse SGD](../algorithms/bias-reduced-sparse-sgd.md) be
evaluated on real data rather than argued about.

!!! warning

    **Sparse features do not buy a sparse per-example gradient buffer.**
    `jax.grad` with respect to `w` returns a *dense* row per example
    regardless of how the input was held, so per-example clipping still
    allocates batch-size × `d` floats. `forward_sparse` buys the
    representation and the forward pass. A sparse clip-and-scatter path
    is separate work and does not exist yet.

## What the model does not do

- **No hidden layers, no framework.** The map is `w · x + b` and nothing
  else, over a plain dict of arrays. dimma ships no layers; a model with
  structure is yours to write as functions.
- **No hashing and no encoding.** Turning a dataset into `x` belongs to
  the loader.
- **No symmetry to break at initialization.** Weights start from a
  Gaussian of scale 0.01 and the bias at zero. A linear model has no
  units to distinguish, so the randomness buys variation between runs
  rather than trainability; the small scale is what keeps the initial
  logits near zero, where the sigmoid is steepest.
- **No regularization term.** The loss is the likelihood and nothing
  added to it.

Import from the module that owns what you need — `dimma.models`
re-exports nothing:

```python
from dimma.models.logreg import init_params, forward
from dimma.models.losses import per_sample_bce_loss
```

Every executed comparison under `notebooks/` runs on this model:
`notebooks/tuning/01-dp-sgd-on-criteo.ipynb`,
`notebooks/comparisons/02-dp-sgd-vs-sgd-baseline-on-criteo.ipynb`,
`notebooks/comparisons/04-dp-sgd-vs-its-projected-counterparts-on-criteo.ipynb`,
and `notebooks/comparisons/05-spiderboost-vs-dp-sgd-on-criteo.ipynb`.
