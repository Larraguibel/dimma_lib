# Bias-reduced sparse SGD

Bias-reduced sparse SGD (Ghazi et al., NeurIPS 2024, Algorithms 3–4), in
`dimma.algorithms.bias_reduced_sgd`. The method whose two transforms dimma
[already ships](../transforms/l1-projection.md), ported whole. Its claim
is a convergence rate that depends on the *sparsity* of the gradients
rather than on the dimension — which makes it the first algorithm here
whose point is lost on dense data.

## One step

A step draws a scale `N` from a truncated geometric law, then a batch of
`2^(N+1)` examples, its two exact halves, and one further record drawn
independently. Four private means are released — the batch's, each
half's, and the single record's — and combined into a nearly unbiased
gradient estimate with the variance of a much larger batch: the debiased
combine reweights by `1/p_N` (the probability of the drawn scale), so
cheap small-batch steps are frequent and expensive large-batch steps are
rare, while the estimator behaves in expectation like the largest one.

The sampler is `dimma.core.sampling.dyadic`; the paper's symbols are
tabulated against the code, line by line, in the package docstring. As in
the other algorithms, stage 1 is host-side in `train` — here even the
*shape* of the batch is random, so each scale compiles once and the
per-scale programs populate lazily as scales are drawn.

Three of the four means come from a **single** draw of the batch, so
`step` returns them as one release, amplified once, jointly — not as
three releases an accountant would wrongly amplify three times. The single
record is the second mechanism, amplified at `1/n`. The mean estimator
itself sits behind a named seam (`estimators.MeanEstimator`), carrying a
typed claim of what it releases, so the accountant reads which estimator
ran from the claim rather than inferring it from code.

## The loop's length is not a setting

Every other loop in dimma takes noise scales and a step count and returns
parameters. This one takes a *budget* and returns the step count: the
loop runs while a privacy filter — the paper's Theorem A.4, in closed
form in `dimma.accounting.bias_reduced_sgd` — permits the next step, and
the run's length is where the filter stops.

No standard accountant applies, and the obstacle is structural rather than
a matter of tightness. Standard composition prices a schedule of releases
that is *fixed before the run*; here the schedule is chosen inside the
run, one scale draw at a time, and the number of steps is the stopping
time of the very process being accounted. A filter is the statement that
covers that. Two further mismatches would each be fatal on their own: the
batch is a fixed-size draw without replacement rather than a Poisson one,
and the triplet is amplified once jointly rather than three times. Every
per-step price is a function of the public coin and the budget — no data
enters a cost.

The direction of error is the safe one — the filter is advanced
composition, which over-reports — but it does mean this algorithm's ε and
DP-SGD's are not produced by the same machinery, a caveat any comparison
between them inherits. The budget is refused above ε = 1, where the
amplification lemma the prices rest on is no longer stated.

## Three departures from the paper

**The inner estimator is the paper's Algorithm 1, not Algorithm 2.** The
pseudocode writes Gaussian ℓ₁-recovery in all four estimator slots; dimma
instantiates perturb-then-project instead. The substitution is this
library's, not the paper's: the privacy analysis carries over verbatim,
and the accuracy bound loses only a `ln(d/s)` for a `ln d`. It is also
cleaner than the paper's own instantiation — no regime condition on the
dimension, no random-matrix failure event folded into δ, and no breakdown
at the batch of one that the single-record slot needs. The estimator seam
exists so Algorithm 2 can later drop in as an accuracy swap without
reopening the step.

**The paper's projection `Π_X` is absent from the loop.** Constraining the
iterates is a caller-side transform in dimma, composed
[at the optimizer seam](../transforms/index.md): wrap the optimizer in
`dimma.transforms.projection.l1_projected` to run the paper's own
containment. No loop grows a projection argument.

**The filter check includes the current step's cost.** The paper's printed
`while` prices only the steps already taken, so it always takes one step
whose cost was never checked; the theorem's own stopping time includes the
step being decided. dimma implements the theorem, stopping at or one step
before the printed rule — never later — so a completed run's realized
spend sits strictly inside its stated budget.

## A numerical ceiling, stated rather than hidden

Releases are float32; the debias combine runs in float64 on the host,
because the combined term nearly cancels and `1/p_N` then multiplies it —
and the rounding each release already carries — by about `2^(N+1)`. That
rounding is a real ceiling: the relative error of the estimate grows like
`2^(N+1)` and reaches order one near the top of a full-size scale ladder.
Nothing raises on it; it is a tested bound on how high a `max_scale` is
worth setting. Lowering `max_scale` is a *mechanism change* — a different,
fully analyzable law whose debias weights follow along — not a truncation,
which would silently break the estimator's unbiasedness.

!!! warning

    **Status: evaluation pending.** No notebook evaluates this algorithm
    yet. Its point is sparse gradients, and the dense 39-feature Criteo
    load gives it none; the one-hot Criteo mode, where per-sample
    gradients are genuinely sparse, is where the evaluation will run.
