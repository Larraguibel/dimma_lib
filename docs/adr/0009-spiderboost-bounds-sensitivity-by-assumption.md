# Private SpiderBoost bounds its sensitivity by assumption, not by an operation

Algorithm 2 has no clipping line, and dimma's implementation adds none. Its
privacy rests on the function class the paper assumes — `f(·; x)` is
`L0`-Lipschitz and `L1`-smooth for every `x` — which bounds each per-sample
gradient by `L0`, and each per-sample gradient difference by
`min(L1‖w_t − w_{t−1}‖, 2·L0)`, without any operation making it so. Reading the
paper's noise scales backwards recovers exactly those two bounds, and the cap
`σ̂2` exists because the second one takes over when the parameters move far.

The alternative was to clip, as DP-SGD does, so that the bounds hold by
construction whatever loss and constants a caller supplies. Whenever the
assumptions hold that would have been free — a clip that never binds is the
identity — and it would have made the guarantee independent of an unchecked
caller claim. We declined it, accepting the risk, in order to run the algorithm
as it was conceived. The technical cost of clipping is also worse here than in
DP-SGD: Lemma 4.1 requires each release to be an unbiased estimate, and a
binding clip breaks that. In DP-SGD a biased step is one biased step; in a
variance-reduced method the bias enters the running estimate and stays there for
the rest of the phase.

## Consequences

`L0` and `L1` do not appear in `algorithms/spiderboost/` at all. They are inputs
to the accountant that produces the noise scales, and the training loop takes
the scales. Per ADR-0003 that is where they belong: they are premises of a
privacy claim, not facts about code.

The guarantee is conditional on something dimma cannot check. If the supplied
loss is not Lipschitz with the `L0` used to calibrate, or not smooth with the
`L1`, then the noise is calibrated against a sensitivity the data can exceed and
the reported (ε, δ) is false — silently, with no crash. The same holds for the
step size, which Theorem B.3 fixes at `1/(2·L1)` and which the loop cannot
verify. All three are named in the package docstring, and again in
`accounting/spiderboost.py`, in the register ADR-0007 uses for
`poisson_truncated`: a precondition stated out loud rather than left in a paper.

Criteo is where this bites. A defensible `L0` for a heavy-tailed feature set is
either fitted on the data — an access no budget accounts for, the same defect
ADR-0008 records for its medians and standard deviations — or large enough to
drown the signal. Any (ε, δ) reported over that data inherits the caveat, and
the honest outcome may be that the assumption does not hold there.

Both halves of that have since been settled and the paragraph above is kept as
written because it is what was believed at the time. ADR-0012 removed the first
horn: `L0` is not fitted but implied by an imposed `R`, so there is no code
path from a feature array to a constant. The second was measured on Criteo in
notebook 03 — it does not drown the signal at any `R` in the defensible range,
including the `R = 1` that needs no look at the data, though it does at an `R`
an order of magnitude above it. What survives is narrower than this paragraph
and belongs to whoever reports a number rather than to the library: `R`'s
provenance is not something `metadata` can record.

Stage 4 is therefore absent from this algorithm's stage table by decision rather
than by omission; see ADR-0001.
