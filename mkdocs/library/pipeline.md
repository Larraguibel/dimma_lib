# The seven-stage pipeline

dimma is organized around one observation: essentially every method in the
DP-SGD family does the same seven things and differs only in what it
chooses at each. The library makes that structure literal. Each stage is
implemented once, as an architecture-agnostic primitive over JAX pytrees,
and an algorithm is a choice at each stage rather than a from-scratch
training loop.

| Stage | What it does | Where it lives |
|---|---|---|
| 1 | Batch generation | `dimma.core.sampling` |
| 2 | Forward pass | the caller's model |
| 3 | Per-sample gradients | `dimma.core.gradients` |
| 4 | Clipping | `dimma.core.clipping` |
| 5 | Aggregation | `dimma.core.aggregation` |
| 6 | Perturbation | `dimma.core.noise` |
| 7 | Optimization | `dimma.core.updates` |

Stage 2 is the caller's: the model runs inside the per-sample loss they
supply, and it is the only place a model appears. dimma owns the other
six — the five that turn a gradient into a private one, and the update
rule its algorithms descend with.

## Why stages, not mathematical objects

The usual shape for a numerical library groups primitives by what they
operate on — `norms`, `reductions`, `distributions` — and that shape would
read more naturally to a JAX user. dimma groups by stage —
`clipping`, `aggregation`, `noise` — because it makes an algorithm
expressible as a *list of choices*, which is what lets two algorithms be
compared stage by stage instead of read side by side.

It also makes an omission legible. `core` names all seven stages even
though it implements five, because the stage an algorithm does *not*
choose describes it as much as the ones it does.
[Private SpiderBoost](../algorithms/spiderboost.md) performs no clipping —
it takes its sensitivity bound from the function class it assumes rather
than from an operation — so stage 4 is absent from that algorithm *by
decision*, documented as absent rather than silently skipped. Absent
stages left undocumented get "repaired" by later readers, and the repair
changes the mechanism.

## The same stages, without the privacy

A non-private method is the same pipeline with stages 4 and 6 dropped,
stage 3 changed to a per-batch gradient, and stage 1 relaxed to ordinary
shuffled sampling. That is deliberate, and it is the pipeline's strongest
piece of evidence for itself: [non-private SGD](../algorithms/baselines.md)
is expressed in the same primitives, taking the same per-sample loss its
private counterpart would be given. It is what makes a private algorithm
and its baseline *comparable* rather than merely adjacent — the difference
between the two runs is the privacy and nothing else — and it is why the
baselines live inside the library instead of in a scripts folder next to
it.

## The import line names the stage

Nothing is re-exported from a package `__init__`. A caller writes

```python
from dimma.core.clipping import per_sample_clip
```

never `from dimma.core import per_sample_clip`. This is deliberately less
convenient than the usual flat re-export: the import line is the one place
a reader always sees, and making it name the stage puts the information
where it cannot be skipped.

The rule bites hardest in `dimma.core.sampling`, where the samplers stay
in separate modules — `poisson`, `poisson_truncated`, `shuffled`,
`dyadic` — rather than being flattened together. What distinguishes two
samplers is whether the standard accounting applies to what they draw,
and flattening them would hide exactly that. A test pins the rule, so a
convenience re-export added later fails loudly instead of quietly widening
the public surface.

## What `core` will not say

`core` describes and never claims. A sampler's docstring may state what it
samples, or which accounting assumption a primitive satisfies — those are
facts about code. The moment a number is called an epsilon, a scale is
calibrated to a budget, or a transformation is called free, it is a claim
about a *mechanism*, and it lives in `dimma.accounting`. Why the split is
load-bearing — a primitive cannot know the mechanism it ended up inside —
is on the [package map](map.md) page.
