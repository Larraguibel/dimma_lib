# The seven-stage pipeline organizes the library

Essentially every DP-SGD-family method does the same seven things — sample,
forward, per-sample gradients, clip, aggregate, perturb, update — and differs
only in what it chooses at each. `core` is therefore organized by stage rather
than by mathematical object: `clipping`, `aggregation`, `noise` rather than
`norms`, `reductions`, `distributions`.

The alternative was to group primitives by what they operate on, which is the
more usual shape for a numerical library and would have read more naturally to
a JAX user. We took stage-grouping because it makes an algorithm expressible as
a list of choices, which is what lets two algorithms be compared stage by
stage instead of read side by side. It also makes an omission legible: `core`
names all seven stages even though it implements five, because the stage an
algorithm does not choose describes it as much as the ones it does.

## Consequences

A primitive that does not implement a stage needs a separate justification to
exist — hence the membership rule in `dimma/core/__init__.py`, which admits
stage-independent math only when it has two consumers in different modules.
`pytree` and `projection` are the two admitted that way, and both are closed
sets.

Omitting a stage is a choice the model expresses, not an exception to it.
Private SpiderBoost performs no clipping: it takes its sensitivity bound from
the function class it assumes rather than from an operation, so stage 4 is
absent by decision. An algorithm's absent stages are part of its description,
and are documented as absent rather than silently skipped — otherwise a later
reader repairs the omission.
