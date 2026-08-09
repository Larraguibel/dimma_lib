# SpiderBoost's accountant composes two subsampled Gaussians

`accounting/spiderboost.py` composes the anchor and variation branches as two
Poisson-subsampled Gaussian events over `dp-accounting`, at rates `b1/n` and
`b2/n`, with noise multipliers `σ1·b1/L0` and `σ2·b2/L1`, over
`ceil(steps/anchor_interval)` and the remaining releases. It does not implement
Algorithm 2's closed-form scales.

The variation branch fits a fixed multiplier because the paper sets
`σ̂2/σ2 = 2·L0/L1`, which is exactly where its per-example bound
`min(L1‖w_t − w_{t−1}‖, 2·L0)` switches: scale and sensitivity saturate
together, so their ratio is constant. Sensitivity is bounded conditional on the
transcript — `w_t` and `w_{t−1}` are post-processing of prior releases, so
under conditioning only the one added or removed example differs. Composition
is adaptive, which is sound because each branch's privacy curve is fixed in
advance and the branch sequence is `t % anchor_interval`, decided before the
run.

The alternative was to transcribe the paper's scales. We declined because they
are asymptotic in an unspecified universal constant, because they are derived
for fixed-size batches while this library samples by Poisson (ADR-0007), and
because two of them are misprinted — the anchor scale and Theorem B.3's phase
length both disagree with their own proofs. Sampling fixed-size batches instead
would have bought literal agreement with a bound that is wrong as printed, at
the cost of splitting this algorithm off from `core`'s sampler and from DP-SGD.

## Consequences

Both algorithms report ε through the same accountant, so a comparison between
them is not also a comparison between a standard bound and a bespoke one. Which
of RDP and PLD is used must still be held fixed across the comparison.

The paper's Theorem B.2 is never invoked, so its applicability condition
`T ≥ n²ε/b²` does not apply.

Privacy is claimed; the convergence rate is not. B.3 is proved under its own
parameter settings, including `η = 1/(2·L1)`. Calibrating numerically produces
different scales, so the rate does not travel with the ε.

A caller may pass a rate and cap whose ratio is not `2·L0/L1`. The effective
multiplier then varies with how far the parameters move, and the sound value is
`b2·min(σ2/L1, σ̂2/(2·L0))`. The accountant uses that and says so rather than
returning a number that silently assumes alignment.

The guarantee remains conditional on the Lipschitz and smoothness constants,
per ADR-0009. Nothing here checks them.
