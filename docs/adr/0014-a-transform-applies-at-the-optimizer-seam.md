# A transform applies at the optimizer seam

The transforms front needed a shape before its first member, the `l_1`
projection, could land: something has to receive the projected quantity, and
what that something is decides who knows the transform exists. Three shapes
were on the table — a `projection` argument or callback on each training loop,
a projection inside each algorithm's step, and a wrapper around the
`updates.Optimizer` every loop already takes.

We took the seam. `l1_projected` wraps an optimizer and projects the point its
increment would produce, so one wrapper serves every algorithm, no `train`
signature grows, and the loops keep the property their docstrings commit to:
nothing happens inside them that the call site cannot see. An argument per
loop was rejected because the two axes multiply — every new transform would
widen every loop's signature. A callback was rejected as a hidden call site,
the same defect for which the loops already refuse metric callbacks. Projecting
only the returned parameters was rejected because it computes something else: a
projected method projects every iterate, and its trajectory never leaves the
ball.

## Consequences

The seam traffics in increments, so the wrapper re-derives the projected point
as `projected − params` for `updates.apply` to add back, and the constraint
holds to floating-point round-off rather than bit-exactly. A caller who needs
the ball exact applies `core.projection` to the returned parameters once.

What #13 deferred stays deferred. Whether an algorithm that projects *as part
of its analyzed mechanism* — SpiderBoost's projection variant is the case in
question — names a transform from `transforms` or imports `core.projection`
directly is a question about that algorithm's package, and it still has no
second call site to be decided against. This ADR fixes where a caller-side
transform composes, and nothing about what an algorithm's own step imports.

A transform still makes no privacy claim (CONTEXT.md). Wrapping an optimizer
changes what runs; whether the projected run's releases are post-processed
freely is stated where that run's accounting is stated.
