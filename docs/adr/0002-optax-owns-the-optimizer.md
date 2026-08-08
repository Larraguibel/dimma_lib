# optax owns stage 7; dimma implements stages 1 and 3 through 6

dimma implements only the stages that turn a gradient into a private one.
Stage 2 is the caller's model, which runs inside the per-sample loss they
supply, and stage 7 — the parameter update — is delegated to `optax`.
`dimma.core.updates` is a two-function seam onto it and re-exports none of
optax's optimizers.

Writing our own SGD would have been a dozen lines and removed a dependency.
We delegate because a private method and its baseline have to be pinned to
the *same* optimizer for the comparison between them to mean anything, and
"the same" is far more credible when both name `optax.adam` at the call site
than when one uses dimma's update rule and the other uses optax's. Which
optimizer a run used is part of what makes it reproducible, so it is better
read from the caller's own import than from a dimma alias.

## Consequences

An algorithm's training loop must thread an optimizer state, which a bare
`params - lr * grad` would not have needed. Learning-rate schedules come free
and index on update calls, which is the same clock privacy composes over — so
a schedule's horizon and the privacy horizon take the same number.
