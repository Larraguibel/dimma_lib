# Which optimizers dimma implements

`dimma.core.updates` implements a rule when a paper implemented here states it
and it is short enough to read against that paper, and borrows it otherwise.
Today that means `sgd`, built from `dimma.core.pytree`; an optimizer no
algorithm here needs is named at the call site from `optax` and passes through
`updates.init` and `updates.apply` unchanged.

Algorithm 1's `theta - eta_t g~_t` and SpiderBoost's constant step are two lines
of pytree arithmetic, and a run departing from either is a different algorithm
rather than a differently configured one. Adam is the other side of the rule:
bias correction, where the epsilon sits, coupled against decoupled decay are
wrong quietly rather than loudly, and a baseline that is subtly wrong does not
lose visibly — it makes every comparison against it meaningless.

This reverses delegating all of stage 7 to optax. That decision rested on
comparison control, which survives below; it simply never required optax.

## Consequences

The signature is load-bearing. `Optimizer` matches optax's `(init, update)`
pair, including the `params` argument nothing here reads, so one seam carries
both an algorithm's `sgd` and a baseline's `optax.adam`. The match is
structural, not nominal — `isinstance(updates.sgd(0.1),
optax.GradientTransformation)` is false, so an optax helper that checks the type
rather than the shape will reject dimma's optimizer.

Both sides of a comparison name the same optimizer, so a non-private SGD
baseline uses `updates.sgd`. Splitting the two would reintroduce the
uncontrolled comparison this decision's predecessor was written to prevent,
pointed the other way.

`updates` grows only when an algorithm implemented here needs a rule. If it ever
holds three or four, the rule above has stopped applying and the module has
become a worse optax.

`sgd` carries a step count even where no schedule reads it, because that count
is both the clock a schedule indexes on and the clock privacy composes over.
Private SpiderBoost will break the identity — its two branches are different
mechanisms, so the update count stops being the composition count — but that is
its accountant's to state, not this seam's.
