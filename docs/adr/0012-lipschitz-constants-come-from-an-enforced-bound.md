# The Lipschitz and smoothness constants come from an enforced bound, not from the data

ADR-0009 left Private SpiderBoost's `L0` and `L1` as unchecked caller claims and
recorded the reason: a defensible constant on a heavy-tailed feature set is
either fitted on the data — an access no budget accounts for — or large enough
to drown the signal. This ADR closes that gap for the model dimma ships.

For logistic regression with sigmoid BCE both constants are known in closed
form. Writing `x̄ = [x; 1]` for the feature vector augmented with the bias, the
per-sample gradient is `(σ(z) − y)·x̄` and the per-sample Hessian is
`σ(z)(1 − σ(z))·x̄x̄ᵀ`. With `sup |σ − y| = 1` and `sup σ(1 − σ) = 1/4`,

    L0 = max_i ‖x̄_i‖ ,    L1 = (max_i ‖x̄_i‖)² / 4 ,    so  L1 = L0² / 4

both global over all parameters and uniform over every example, which is the
form Algorithm 2 assumes rather than the weaker statement about the empirical
risk. Everything therefore reduces to one number: the largest augmented feature
norm. That number is the one thing we may not measure.

So we do not measure it, we impose it. A per-example map `x ↦ x / max(1, ‖x‖/R)`
bounds every feature vector by `R` while touching one record at a time. It reads
no aggregate, so it costs no budget, and afterwards `R` is a fact about the
preprocessing rather than a claim about the data. `L0` and `L1` follow from `R`
and from whether the model carries a bias, and dimma computes them from those
two inputs alone.

We take `x ↦ x / max(1, ‖x‖/R)` rather than projecting every example onto the
sphere of radius `R`. Projection would make every norm equal and so make `L0`
exactly tight, at the cost of discarding each record's magnitude outright; the
capping form leaves the majority of records untouched and is what the DP-ERM
line does, so a reported ε is comparable with the numbers that literature
reports. Comparability is what this library is for, and buying a tighter
constant with it is the wrong trade.

The alternative that lost is to fit `R` on the data once, offline, and treat it
as a constant thereafter. Chaudhuri et al. (2011) do exactly this — normalizing
each column by the dataset maximum — and do not charge for it, which is how the
defect became conventional. We decline it: a bound read off the data is a
non-private query whatever the wall-clock distance between reading it and using
it. The second alternative, replacing uniform Lipschitzness with a moment bound
as the heavy-tail literature recommends, needs a different analysis and so a
different algorithm; ADR-0009 already declined to modify Algorithm 2.

## Consequences

`L0`, `L1` and the step size are produced together, from `R` and a bias flag, by
one function in `accounting/`. Together because Theorem B.3 fixes `η = 1/(2·L1)`,
so the three are one triple and returning them separately invites a caller to
pair one normalization with another run's step size — the reasoning ADR-0006
gives for the noise parameters. In `accounting/` because "this loss is
`L0`-Lipschitz" is the premise of a privacy claim, and per ADR-0003 that is not
`core`'s to make.

That function never takes the data. Its inputs are `R` and whether the model has
a bias, and there is no code path from a feature array to a constant. This is
the enforcement: a later contributor who thinks measuring `max ‖x‖` would be
convenient has nowhere to put it.

The bias flag is not a detail. The augmented norm is `‖x‖² + 1`, so with `R = 1`
a model with a bias has `L1 = 1/2` and one without has `L1 = 1/4`. Defaulting it
silently would halve or double every noise scale and report an ε that is simply
false, which is the failure mode ADR-0009 describes: wrong, and with no crash.

The map itself lives in `dimma/datasets/preprocessing.py` and knows nothing
about any dataset: it takes an `(n, d)` array and a norm bound, and returns the
rescaled array of the same shape together with the bound it enforced. Criteo
appears nowhere in it. It is not in `core`,
which admits only stage implementations and stage-independent math with two
consumers in different modules, and this is neither; and it is not a transform,
which acts on a quantity inside the pipeline rather than on the data before it.
It describes what it did to the data and claims nothing, the same seam ADR-0003
draws between `core` and `accounting`.

That widens the `datasets` rule from one module per dataset to one module per
dataset plus the shared maps those loaders compose, revising how ADR-0008 states
the package. The wider rule carries a condition: every function in
`preprocessing.py` says whether it reads across records. Criteo's three
preprocessing helpers all do — train-split medians, category frequencies, means
and standard deviations — so the norm cap is the first per-record map in the
library, and telling the two apart is the whole basis of this decision. A fitted
map is an unaccounted access and inherits the caveat ADR-0008 already records; a
per-record map is free. Splitting the two kinds into separate modules was
considered and declined: `poisson` and `poisson_truncated` are split so that a
caller choosing between substitutes has to say which they chose, and these two
are not substitutes.

The arithmetic coincides with stage 4 and the code is deliberately not shared.
`core.clipping.per_sample_clip` rescales a batched pytree to a norm bound, which
is the same map; but `CONTEXT.md` pins *clipping norm* to the bound on a
per-sample gradient, and an import line reading `from dimma.core.clipping import
...` in preprocessing would name the wrong stage. The two samplers are kept in
separate modules on the same reasoning (ADR-0007). The capping form is also
written `x / max(1, ‖x‖/R)` rather than `x · min(1, R/‖x‖)`, so a zero row needs
no epsilon; stage 4's `+ 1e-12` is a distortion justified for gradients, not one
to copy here.

The cap goes last in a loader's chain, after every fitted map. A fitted map
rescales columns, so it does not preserve a bound applied before it: capping to
`R = 1` and then standardizing thirteen columns of differing scale leaves the
largest norm near 8, and the accountant would be handed `L0 = √2` for data
carrying `L0 ≈ 8`. Noise calibrated at a fifth of what the mechanism needs, an ε
reported, and nothing to crash — which is the failure ADR-0009 describes, walked
back in through an ordering. Column statistics first, then the per-record cap,
is also the order the DP-ERM line applies them in.

The map returns the bound it enforced along with the rescaled data, so `R`
reaches the accountant from the operation that made it true rather than being
typed a second time at the call site. Two numbers that must agree, entered
separately, is the same unchecked-claim defect this ADR exists to remove — a cap
at `1.0` reported as `0.5` is a false ε, and nothing in the library would know.
This does not make disagreement impossible, since the accountant takes a float
and a caller may pass any float; it removes the need to restate the number,
which is where the error would actually come from. Where a loader applies the
cap, the bound also goes in the split's `metadata` per ADR-0008, so the two
routes agree.

What the map does not return is any statistic of the data it saw. The count of
records it rescaled, or the norm distribution behind them, is data-dependent,
and handing one back is an unaccounted release through the back door. The bound
is not such a statistic: it is an argument, fixed before the data was touched.

This does not make the guarantee unconditional. It moves one premise from the
caller's word to dimma's code, and the remaining looseness is real: `L0` is
attained only as the logit diverges and `L1`'s `1/4` only at a logit of zero, so
both are worst-case constants a trained model never approaches. Whether a
row-capped Criteo is still the benchmark we meant to run is a modelling
question, and is settled where the evaluation is designed rather than here.
