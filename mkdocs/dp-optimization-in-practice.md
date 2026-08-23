# DP optimization in practice

!!! note

    **DP-SGD flow:**
    generate batches → batch forward → *per-sample* gradients →
    *per-sample* clip → aggregate the clipped gradients → add noise →
    update

This page is for readers who know differential privacy but arrive from the
theory or statistics side. Nothing here defines ε, δ, or the Gaussian
mechanism. What it covers is the set of distinctions *applied* DP
optimization forces — distinctions the theory literature knows but rarely
leads with, because they only start to matter when a training loop has to
run. The literature often leaves them implicit, forcing readers to piece
the logic together; this page poses the questions first. Why does *how*
you sample the batch matter for privacy? What exactly does an accountant
price, and why do different sampling strategies need different ones?

Almost every DP-SGD-family method fits a standard seven-stage pipeline:

1. **Batch generation** — sample the data under a specific private
   methodology.
2. **Forward pass** — push the batch through the model.
3. **Per-sample gradients** — one gradient per individual example.
4. **Clipping** — bound each per-sample gradient's norm, bounding the
   sensitivity of what follows.
5. **Aggregation** — sum or average the clipped gradients.
6. **Perturbation** — add noise calibrated to the privacy budget.
7. **Optimization** — update parameters from the privatized gradient.

The two things at the center of everything are **how batches are sampled**
and **how gradients are randomized**. Everything else — the accounting,
the clipping strategy, the noise calibration — follows from understanding
those two clearly. How dimma turns the pipeline into architecture is [its
own page](library/pipeline.md); this one stays on the concepts.

## Batches and gradients

Every differentially private stochastic gradient method protects the
training data through two mechanisms applied at every step: how batches
are drawn, and how gradients are randomized.

### Batch sampling

How training examples are selected at each step determines the privacy
amplification by subsampling — the reduction in per-step privacy cost that
comes from each step seeing a random subset rather than everything. If
each example is included in the batch independently with sampling rate
`p = b/n`, where `b` is the expected batch size and `n` the number of
training points, this is Poisson subsampling, the classic sampling
strategy for DP. The intuition is simple: under Poisson sampling any given
individual is simply *absent* from many steps — absent from each with
probability `1 - p` — which fundamentally limits how much information
about them can leak across the run.

In accounting terms Poisson subsampling is the gold standard: the
accountants for it are tight, so the budget is tracked with little slack.
The practical problem is that the realized batch size is `Binomial(n, p)`,
not a fixed number, and computers strongly prefer fixed-shape data. That
tension — a mechanism that wants variable-size draws inside hardware that
wants fixed shapes — runs through every implementation decision, and
[working with pytrees](pytrees.md) shows where it lands in JAX.

The most common alternative in practice is **shuffled minibatch
sampling**: partition the dataset by shuffling and iterate in epochs. This
gives perfectly regular batch sizes, which is why every deep learning
framework does it by default. Its privacy cost is harder to reason about:
accountants for shuffling are not as tight as those for Poisson
subsampling, so it means either a looser guarantee or additional
assumptions.

### Randomizing gradients

Randomizing gradients means adding noise to the gradient estimate at each
step, and the question is how much noise a target `(ε, δ)` requires.

The first obstacle is that deep learning losses are generally not
Lipschitz: gradients can grow arbitrarily large depending on the input,
the weights, and the architecture, so the sensitivity of the aggregated
gradient is unbounded. Clipping each per-sample gradient to an ℓ₂ norm of
at most `C` (the clipping norm) is what artificially imposes the bound.
Once every individual gradient is clipped, the sensitivity of the
aggregate is known and finite, and the Gaussian mechanism can be
calibrated against it.

From there the noise follows from the sensitivity and the budget — but
`C` is simultaneously an optimization decision. A larger `C` clips fewer
gradients and preserves the gradient's direction, but raises the
sensitivity and so the noise. A too-small `C` clips heavily, and heavily
clipped gradients are *biased*: they no longer satisfy the core SGD
premise that the expected gradient estimate tends to the true gradient, so
the optimization can drift or stall — the signal itself has been
corrupted.

Then there is ε. The tighter the budget, the more noise on top of whatever
sensitivity `C` imposed. Both forces act in the same direction: aggressive
clipping biases the gradient, and a tight budget buries what signal
remains under noise. Together they are the privacy–utility trade-off, and
there is no way around it — only positions along it.

## Hyperparameters entangle

One of the most disorienting aspects of DP training, coming from the
standard paradigm, is that hyperparameters are far more entangled. In
ordinary deep learning, batch size, epochs, and architecture tune somewhat
independently. In DP training almost every hyperparameter feeds the
accounting, and changing one forces reconsidering the others.

The clipping norm `C` sets the sensitivity, which determines how much
noise a given ε requires — but as above, `C` is also an optimization
choice. The budget `(ε, δ)` determines the noise on top of whatever `C`
imposed. The number of steps determines how many times that budget is
spent. Under Poisson subsampling the expected batch size controls the
sampling rate `p = b/n`, which governs amplification and therefore what
each step costs. And the noise multiplier is generally not a free
parameter at all: it is the *output* of the accountant once the others are
fixed.

Change any one and the rest move, sometimes non-obviously. Doubling the
number of steps does not simply train longer: it spends more budget, which
forces a larger multiplier, a different `C`, or a worse ε. The
relationships are governed by the accountant.

The one hyperparameter outside this web is the **learning rate**. It
appears nowhere in the accounting and has no effect on ε or δ. It still
interacts with the noise indirectly — a heavily noised gradient wants a
different learning-rate regime than a clean one — but it can be tuned
without touching the budget. It is the closest thing DP training has to a
free parameter.

!!! note

    Both of these conceptual knots — clipping as the thing that costs
    utility, and tuning as the thing that moves comparisons — show up
    empirically in [the Criteo evaluation](evaluation.md).
