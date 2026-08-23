# Standardization is its own loading axis, and it defaults off

`load_criteo` takes `standardize` as a third independent parameter alongside
`features` and `preprocess`, defaulting to `False`. It previously rode inside
`preprocess=True`, which was the default, so every caller who preprocessed
standardized whether or not they meant to.

ADR-0008 split loading modes into axes on the grounds that a name hides the half
of the behaviour that matters. `preprocess` was still a name: it covered a
median fill, a clip, a `log1p` and a frequency encoding, all of which leave a
column roughly where it was, and one step that rescales every column to unit
variance. On the pinned file the first four are close to no-ops — the integers
arrive already inside `[0, 1]`, so nothing fills, nothing clips, and `log1p`
compresses into `[0, log 2]` — and the standardization is the entire difference
between the stored data and what came out. Bundling the only step with an effect
under the switch named for the four without one is the failure ADR-0008
describes, one level down.

## Why it defaults off

Standardizing sets each column's scale to 1, so it lands a typical row's ℓ₂ norm
near `sqrt(d)` regardless of where it started. On this file that is an increase,
and a large one. Over the numeric chain the largest row norm goes from about 2.0
to about 15.1; over all 39 features, from about 2.7 to about 16.7.

That largest norm is not a summary statistic here. ADR-0012 makes `R` the bound
a loader *enforces*, and `L0 = R`, `L1 = R²/4`, `η = 1/(2·L1)`. An `R` that
leaves standardized Criteo intact is roughly seven times one that leaves the
unstandardized chain intact, so `L1` and the noise on the variance-reduction
step grow by a factor near fifty, and the step size shrinks by the same. The
alternative — keeping `R` small and capping anyway — discards most of each
record, because the cap now binds on nearly every row rather than the tail.
Either way standardization is paid for in the budget. Every fitted map in this
loader is an unaccounted access to the training data, which ADR-0008 already
records; what is particular to this one is that it also moves the quantity `R`
has to cover, and so shows up a second time in the noise.

So the default is the cheap chain, and a caller who wants the conditioning asks
for it and sees the price in the ε they report. This is not a claim that
standardizing is wrong: on a heavy-tailed feature set with columns spanning
orders of magnitude it is what makes a run converge at all. It is a claim about
which way the default should fall when the parameter is invisible.

## Consequences

`standardize` composes onto whichever chain `preprocess` produced rather than
selecting a different one, and is honoured with `preprocess=False` too. That
combination standardizes the stored values, and a stored NaN then takes its
whole column with it, because the fill that would have removed it belongs to
the axis the caller declined. Propagating is correct: imputing under a parameter
named `standardize` would be the bundling this ADR removes.

The order is unchanged and still load-bearing. Fitted column maps first, then
the per-record cap, for the reason ADR-0012 gives: a cap applied before a
rescale bounds nothing afterwards. `standardize` moves inside that ordering, not
around it.

`metadata` carries `"standardize"` always, and `"feature_means"` and
`"feature_stds"` only when it is true — they track the step that fitted them
rather than `preprocess`, which fits medians and frequencies and neither a mean
nor a deviation. The prose in `metadata["preprocessing"]`, and the notice
printed once per configuration — the `emit_once` key is `features`,
`preprocess`, `standardize` and `feature_norm_bound`, so a process that loads
two chains sees two notices — name the choice in both directions: eight modes,
eight descriptions, none of them able to stand in for another.

Flipping the default changes what existing callers get from the same call, and
it changes results rather than correcting an error. Earlier runs were not
reporting a false ε: the cap ran after the standardization, so `R` bounded what
it claimed to bound. What they were doing is spending an `R` sized for
standardized norms — or, at a smaller `R`, capping nearly every row instead of
the tail, which is the outcome ADR-0012 chose the capping form specifically to
avoid. A run recorded before this change is still valid and is not comparable
with one after it; `metadata["standardize"]` and the printed notice are how the
two are told apart.
