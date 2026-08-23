# The transform seam

A **transform** is a change to a quantity — parameters, a privatized
gradient — that is not itself an algorithm and composes across several.
Transforms are a separate axis from algorithms: one transform can apply to
several algorithms, and an algorithm can stack more than one. A change
only one algorithm could ever make is that algorithm's, and lives in its
package.

A transform is deliberately *not* called a mechanism. In the DP
literature a mechanism is a complete randomized map carrying its own
privacy analysis; a transform carries none. Whether applying one is free
post-processing is a statement about the mechanism it sits inside —
about what the run *released* — and it is stated where that run's
accounting is stated, never as a property the transform carries around
with itself.

## One wrapper serves every algorithm

Every training loop in `dimma.algorithms` takes its optimizer through the
same structural seam, `dimma.core.updates.Optimizer`. A transform is a
wrapper around that seam:

```python
from dimma.core import updates
from dimma.transforms.projection import l1_projected

optimizer = l1_projected(updates.sgd(0.1), radius=5.0)
```

and the caller passes it to *any* `train` — DP-SGD's, SpiderBoost's, the
baseline's — none of which know it is there. This is the sense in which
projection is enabled for the whole library at once: the seam is shared,
so a transform written once composes with every algorithm, present and
future.

Three other shapes were considered and rejected, and the rejections
define the layer:

- **A projection argument on each training loop** — the two axes
  multiply, so every new transform would widen every loop's signature.
- **A callback inside the loop** — a hidden call site, the same defect
  for which the loops already refuse metric callbacks: nothing may happen
  inside a private loop that the call site cannot see.
- **Projecting only the returned parameters** — computes something else.
  A projected method projects every iterate; its trajectory never leaves
  the ball.

The geometry itself lives in `dimma.core.projection`; the transform layer
is only the application of it at the seam.

## Current members

| Transform | What it changes |
|---|---|
| [ℓ₁ projection](l1-projection.md) | the iterates, or the released estimate |
