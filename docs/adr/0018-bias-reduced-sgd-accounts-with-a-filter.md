# Bias-reduced SGD accounts with a filter, because the schedule is the run's own

`accounting/bias_reduced_sgd.py` implements Theorem A.4's `(ε,δ)`-filter in
closed form and nothing else. A step drawn at scale `N` costs
`(3·2^(N+1)+1)·(ε,δ)/(16n)` — three inner Gaussian releases at `(ε/32, δ/16)`
each, basic-composed and amplified once jointly at `2^(N+1)/n`, plus `G₀`
amplified at `1/n`, which is Lemma 5.3 as printed. The filter admits a step
while `sqrt(2·ln(4/δ)·Σεₛ²) + ½·Σεₛ² ≤ ε/2` and `Σδₛ ≤ δ/4`, i.e. Theorem A.4
at `δ′ = δ″ = δ/4`, the paper's own split. Every number is a function of the
public coin and the budget; no data enters a cost.

No existing accountant applies, and the obstacle is structural rather than a
matter of tightness. `accounting/sampling.py` composes a schedule of releases
that is **fixed before the run**; here the schedule is chosen inside the run,
one draw at a time, and the number of steps is the stopping time of the very
process being accounted. A filter is the statement that covers that, and it is
what the paper uses. Two further mismatches would each be fatal on their own:
the batch is a fixed-size draw without replacement rather than a Poisson one,
and the triplet is three releases from a *single* draw, amplified once and not
three times.

## The carve-out from ADR-0011

ADR-0011 makes `method="rdp"` the default in every `accounting` function.
Nothing here takes a `method`, and a test pins that it never will. There is no
second path for such an argument to select between: `dp_accounting` exposes
accountants that compose a fixed schedule, and nothing filter-shaped, so the
closed form above *is* the whole accountant. An argument would advertise a
choice that does not exist.

The cost is that this algorithm's ε and DP-SGD's are not produced by the same
machinery, so a comparison between them is also a comparison between advanced
composition under a filter and a Rényi accountant — which is the one thing
ADR-0011 was written to avoid. We accept it here because the alternative is not
a looser number but no number at all, and because the direction of the error is
the safe one: Theorem A.4 is advanced composition, so it over-reports. A Rényi
filter would be materially tighter and is ticketed (#35). Swapping it changes
what is claimed about every run, so it lands as an ADR, not as an argument.

## The one-step shift is removed

Algorithm 4's printed `while` sums the costs of steps `s ≤ t−1`, so it decides
step `t` without pricing it and always takes one step whose cost was never
checked; its `ε/2` threshold is what absorbs that. Theorem A.4's own stopping
time is `inf{t : ε < ε[0:t+1]}` — the check *including* step `t`. dimma
implements the theorem. Both sums are monotone, so this stops exactly one step
earlier than the printed rule, never later, and the run is covered by Lemma 5.3
with slack to spare. Concretely: the whole transcript is inside the filter, so
a completed run's realized cost is at or under `(ε/2, δ/4)` while its
guarantee is `(ε, δ)`.

## Consequences

`train` imports `accounting`, which no other loop in the library does. The
filter is the termination condition, so the alternatives were to inline its
closed form in `algorithms/` — putting a privacy claim outside `accounting/`,
which ADR-0003 forbids — or to hand the loop a callback, which hides which
analysis ran. The import is the honest form of a dependency that already
exists.

`train` takes a budget rather than noise scales, and returns `steps` rather
than taking them. Both are departures from the shape ADR-0011 left the loops
in, and both are forced: the budget *is* the stopping rule, and `T` is an
output of the algorithm. `Run.spent` and `Run.steps` are not metrics under
ADR-0006 — they are deterministic functions of the public coin and the budget,
they involve no access to the training data, and without `steps` a caller could
not account for the run at all. `train` still computes no epsilon; that
conversion is `accounting.bias_reduced_sgd.epsilon`.

The claim-type check lives here as `check_claim`, and `train` calls it before
its first step. Lemma 5.3 composes four Gaussian mean releases, so an estimator
carrying anything but a `GaussianMeanClaim` is refused rather than priced by
resemblance — Algorithm 2's Theorem 3.4 folds a random-matrix failure event
into `δ` and carries a regime condition that fails outright at the batch of one
`G₀` needs. This is what "the release boundary stays visible to the accountant"
means in code.

Two things the numbers rest on are the paper's assertions and not this
library's, stated verbatim in the module docstring on ADR-0009's pattern: that a
fixed-size draw without replacement amplifies at `2^(N+1)/n`, and that the
empirical mean's `l₂` sensitivity is `2L/k` (Theorem 3.3). Clipping enforces
`L`, so that half is code (ADR-0012); the rest is assumed. Proving the
fixed-size amplification is a research question, not a port.

The budget is refused above `ε = 1`, because Lemma 5.3's amplification is
stated there and above it every per-step price becomes an under-estimate rather
than a bound. `δ` is only warned about in prose: Lemma 5.5's stopping-time
bound assumes `δ < 1/n²`, and above that the privacy still holds while the
expected step count does not.

A property worth stating because it is easy to expect the wrong way round: both
arms of the per-step price are *proportional* to the budget, so a larger budget
does not buy more steps. It buys quieter ones — `inner_noise_multiplier` is
`32·sqrt(2·ln(20/δ))/ε`. The run's length is set by `n` and by the scales the
coin drew, and in every regime a test suite can reach it is the `δ` arm that
stops the run.
