# Non-private baselines

`dimma.algorithms.sgd` is non-private SGD: not an algorithm from a paper,
but [DP-SGD](dp-sgd.md) with the privacy taken out and *nothing else
changed*.

## Why the baseline lives inside the library

The way most DP papers compare is against whatever reference
implementation is handy — and that makes "what did privacy cost here" an
uncontrolled question, because the private and non-private runs then
differ in the optimizer, the data loading, and the loss as well as in the
privacy. dimma's baselines are built from the same stages, in the same
primitives, taking the same per-sample loss their private counterpart
would be given, so the difference between the two runs is the privacy and
nothing else.

The same rule binds in the other direction: classical DP-SGD is the
reference every non-classical method is measured against, so *it* gets
the same primitives, tests, and documentation as anything else — rather
than being a strawman written to lose.

## Stage by stage, against DP-SGD

| Stage | DP-SGD | here |
|---|---|---|
| 1 | `sampling.poisson` | `sampling.shuffled` |
| 3 | `gradients.per_sample_grads` | `gradients.batch_grads` |
| 4 | `clipping.per_sample_clip` | dropped |
| 5 | `aggregation.sum_over_batch` | inside `batch_grads` |
| 6 | `noise.add_gaussian` | dropped |
| 7 | `updates.apply` | `updates.apply` |

Stage 5 survives as the mean inside `batch_grads`. Stage 7 is
`updates.sgd` — the *same optimizer object* DP-SGD is given, and not
`optax.sgd`, which exists and would compute the same thing. Splitting the
two sides onto different implementations of the same rule would
reintroduce, pointed the other way, exactly the uncontrolled comparison
the baseline exists to prevent. An optimizer no algorithm here implements
— Adam, for a stronger baseline — is named from `optax` at the call site
and passes through the same seam, so both sides of a comparison can still
be pinned to one optimizer.

## What dropping the privacy simplifies

Two things follow from dropping stage 6. The loop carries one random
stream instead of DP-SGD's two, so a run is reproducible from a single
seed. And `step` is a single plain function rather than a release/apply
pair: dimma splits an algorithm's step at what each mechanism *releases*,
and a non-private algorithm composes zero mechanisms — it releases
nothing, so there is no boundary to make visible. That is the
release-counting rule reaching zero, not an exception to it.

!!! note

    **Evaluated in**
    `notebooks/comparisons/02-dp-sgd-vs-sgd-baseline-on-criteo.ipynb`,
    as the anchor of the DP-SGD comparison. Every headline number in
    [the evaluation](../evaluation.md) is read as a distance to this
    baseline, not as an absolute.
