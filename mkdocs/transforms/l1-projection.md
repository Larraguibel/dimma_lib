# ℓ₁ projection

`dimma.transforms.projection` applies the ℓ₁-ball geometry of
`dimma.core.projection` at the [optimizer seam](index.md), to either of
the two quantities the seam carries. Two wrappers, two different objects
constrained — and the distinction is the whole story of this transform.

## `l1_projected` — constrain the iterates

```python
from dimma.core import updates
from dimma.transforms.projection import l1_projected

optimizer = l1_projected(updates.sgd(0.1), radius=5.0)
```

Runs the wrapped rule, adds its increment to the current parameters,
projects the result onto `{w : ‖w‖₁ ≤ radius}` — one global ball across
every leaf of the pytree — and re-expresses the projected point as the
increment the loop adds back. Every update lands inside the ball, so the
whole trajectory stays in it. Because the seam traffics in increments,
the constraint holds to floating-point round-off rather than bit-exactly;
a caller who needs the ball exact applies `core.projection` to the
returned parameters once.

This is also how to run the containment step that some papers write into
their algorithm box — bias-reduced sparse SGD's `Π_X` among them — without
any training loop growing a projection argument.

## `l1_projected_estimate` — constrain the estimate

The second wrapper projects the *incoming estimate* before the wrapped
rule runs: the update descends along the projection of what stage 6
released, and the iterates go wherever it sends them. The motivation is
denoising — for a mechanism releasing a noisy mean of *sparse*
per-example vectors, projecting the release onto an ℓ₁ ball of the right
radius removes most of the noise, a property of the geometry and the
radius rather than of the noise, and the shape in which the sparse-DP
literature applies its projection to an already-released gradient.

The radius is the caller's number. The paper-prescribed value for the
sparse setting is the gradient bound times `√s` (`s` the sparsity); the
library states no sparsity it cannot check, so it takes a radius, not an
`s`.

!!! warning

    **The denoising presupposes sparse per-example gradients.** dimma's
    reference model — dense 39-feature logistic regression on Criteo —
    has none, and the evaluation below measured what that does: at every
    principled radius the estimate-side projection was the identity map,
    with 1.9 decades of headroom between what the mechanism releases and
    the paper's ball. The assumption is stated, and the honest outcome on
    dense data is that it does not hold. A sparse problem is where this
    wrapper would earn its place — the same reason
    [bias-reduced sparse SGD](../algorithms/bias-reduced-sparse-sgd.md),
    which builds this projection into its inner estimator, waits on a
    sparse load for its evaluation.

## No privacy claim, by construction

Both wrappers post-process. Whether the projection is free in a given run
is a statement about the mechanism it post-processes — stated where that
run's accounting is stated, not here. A projected update rule is also a
departure from any paper whose rule does not project, and as with any
other stage-7 choice, that departure is the caller's to make and to
report.

!!! note

    **Evaluated in**
    `notebooks/comparisons/dp-sgd-vs-its-projected-counterparts-on-criteo.ipynb`
    — DP-SGD against both wrappers, at matched ε. The two-sided finding is
    on the [evaluation page](../evaluation.md).
