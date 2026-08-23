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
under conditioning only the one added or removed example differs. That
conditioning is what makes the *magnitude* sound: the variation branch's noise
is chosen adaptively, as a function of prior releases only. The two schedules
themselves compose *non-adaptively* — the branch sequence is
`t % anchor_interval`, fixed before the run, and each branch's privacy curve
with it, which is the order independence
`sampling.composed_poisson_gaussian_epsilon` states. Adaptive magnitude,
non-adaptive schedule; only the second is what the accountant composes.

That saturating bound is a reconstruction from the algorithm box, not a
restatement of the proof. B.3's privacy argument cites only
`L1‖w_t − w_{t−1}‖`; the cap `σ̂2` and the `2·L0` inside it appear nowhere in
it. Of the two claims the paper makes in different places, dimma implements the
tighter one.

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

The worked configuration, quoted by `accounting/spiderboost.py` and its tests:
`steps` = 2000, `anchor_interval` = 20, `b1` = 2000, `b2` = 500, `n` = 100000,
`L0` = 1, `L1` = 5, δ = 1e-6 — 100 anchor releases at `q` = 0.02 and 1900
variation releases at `q` = 0.005. Calibrating gives the shared multiplier:

| ε\* | RDP μ | PLD μ |
|---|---|---|
| 0.5 | 2.750093 | 2.562007 |
| 1.0 | 1.611770 | 1.507924 |
| 3.0 | 0.931127 | 0.868438 |

Summing the branches' separate εs instead of composing them in one accountant
costs 46% more ε at ε\* = 0.5 under RDP, rising to 63% at ε\* = 3. Summing is
sound — it is basic composition — but pays to convert to ε twice, and the
penalty survives choosing the most favourable split of δ between the branches,
which moves it by under a tenth of a percent. This is what
`sampling.composed_poisson_gaussian_epsilon` exists to avoid.

The paper's Theorem B.2 is never invoked, so its applicability condition
`T ≥ n²ε/b²` does not apply.

Privacy is claimed; the convergence rate is not. B.3 is proved under its own
parameter settings, including `η = 1/(2·L1)`. Calibrating numerically produces
different scales, so the rate does not travel with the ε.

A caller may pass a rate and cap whose ratio is not `2·L0/L1`. The effective
multiplier then varies with how far the parameters move, and the sound value is
`b2·min(σ2/L1, σ̂2/(2·L0))`. The accountant refuses by default, naming that
value in the message; `accept_misaligned_scales=True` opts into it with a
warning. Neither path returns a number that silently assumes alignment.

The guarantee remains conditional on the Lipschitz and smoothness constants,
per ADR-0009. Nothing here checks them.
