"""Bias-reduced sparse SGD (Ghazi et al., NeurIPS 2024), Algorithms 3-4.

The method whose two transforms dimma already ships, ported whole. Its
claim is a rate that depends on the sparsity of the gradients rather
than on the dimension, so it is the first algorithm here whose point is
lost on dense data. One step draws a scale ``N`` from a truncated
geometric law, then a batch of ``2 ** (N + 1)`` examples, its two exact
halves, and one further record drawn independently; four private means
are released and combined into a nearly unbiased gradient estimate with
the variance of a much larger batch. The loop's length is not a
setting: it is where the privacy filter stops.

Algorithm 3/4 in the paper's notation, against the piece that
implements it. The paper's symbols appear here and nowhere else in the
package, so a signature can be read without the paper open:

========================================  ===================================
Algorithm 3/4                             this package
========================================  ===================================
``N ~ TGeom(M)``                          `sampling.dyadic.draw_scale`
``M = floor(log2 n) - 1``                 `dyadic.max_scale`
``B ~ Unif(binom(n, 2 ** (N + 1)))``      ``dyadic.subsample(...).whole``
``O``, ``E``, the halves of ``B``         ``.odd`` / ``.even``
``I ~ Unif([n])``                         ``.single``
``grad F_B``, ``grad F_O``, ``grad F_E``  `gradients.per_sample_grads` ->
                                          `clipping.per_sample_clip` ->
                                          `aggregation.average_over_batch`
``Gaussian l_1-Recovery(.)``, four slots  `estimators.MeanEstimator.estimate`,
                                          from `projection_estimator`
``G = (1/p_N)[G+ -(G-_O+G-_E)/2] + G_0``  `step.debiased_gradient`
``x^{t+1} = Pi_X(x^t - eta G(x^t))``      `updates.apply`; ``Pi_X`` is the
                                          caller's, see below
the ``while`` condition                   `accounting.bias_reduced_sgd.permits`
``x-bar`` / ``x_that``                    ``train.Run.average_params`` /
                                          ``.random_params``
``L``, assumption (A.5)                   ``clip_norm``, *enforced* by stage
                                          4 (ADR-0012)
``L sqrt(s)``, the radius of ``K``        ``radius``, the caller's number
                                          (ADR-0015)
========================================  ===================================

Stage 1 is not in `step`, as in the other private packages: the scale
and the draws are host-side on the NumPy generator, and `train` owns
them.

Three departures from the paper
-------------------------------
**The inner estimator is Algorithm 1, not Algorithm 2.** The pseudocode
writes Gaussian ``l_1``-Recovery in all four slots. Section 5.1's prose
calls the inner estimator "the projection mechanism", which is
Algorithm 1's name, and Section 5 opens by integrating "the mean
estimation algorithm*s*" of Section 3, plural. That the substitution
carries is *our* derivation and not the paper's:
`docs/research/algorithm-1-carries-algorithm-3.md` gives it, the
privacy analysis carries over verbatim, the accuracy bounds lose only
``ln(d/s)`` for ``ln d``, and the swapped second-moment bound is pinned
by a test rather than cited. The seam is `estimators.MeanEstimator`, so
Algorithm 2 later drops in without reopening the step.

**``Pi_X`` is absent from the loop.** ADR-0014 settled where a
caller-side projection composes — around the optimizer — and rejected a
projection argument on the loops. Wrap the optimizer in
`dimma.transforms.projection.l1_projected` to run the paper's own
containment.

**The filter check includes the current step's cost.** Algorithm 4's
printed ``while`` sums the costs of steps ``s <= t - 1``, deliberately
taking one step whose cost was never checked and absorbing it in the
``eps/2`` threshold. Theorem A.4's own stopping time is
``inf{t : eps < eps[0:t+1]}``, the check *including* step ``t``. dimma
implements the theorem, so it stops at or before Algorithm 4's ``T``
and is covered by Lemma 5.3 with slack left over.

What float32 costs, and where it ends
-------------------------------------
Releases are float32, as the device computes them; the debias combine
and the parameter update are float64 on the host, because the bracket
``G+ - (G-_O + G-_E)/2`` nearly cancels and ``1 / p_N`` then multiplies
it by about ``2 ** (N + 1)``. float64 stops the combine adding rounding
of its own. It cannot remove the rounding each release already carries
from its own sum, perturbation and projection, of order
``eps32 * clip_norm``, and that rounding is amplified by the same
``2 ** (N + 1)``::

    relative error of G  ~  2 ** (N + 1) * eps32 / (4 * noise_multiplier)

So the estimate degrades at the top of the ladder, reaching order one
around ``N = log2(4 * noise_multiplier / eps32) - 1`` — near ``N = 26``
at a multiplier of 3, and never in sight if the releases were float64.
Nothing raises on it. It is a documented and tested ceiling: a
``max_scale`` above it puts the largest and rarest scale below float32
resolution, and since ``max_scale`` is a mechanism parameter, lowering
it is an analysable change rather than a truncation.

Preconditions this package cannot check
---------------------------------------
Stated out loud because a run violating them fails silently, with a
number rather than a crash:

- **the per-sample gradients are sparse.** Assumption (A.7): every
  ``grad f(.; x)`` has at most ``s`` nonzero coordinates, for the ``s``
  the caller's ``radius = clip_norm * sqrt(s)`` was chosen from. This
  is the assumption the whole method exists for; on dense gradients the
  projection denoises nothing, the estimate is no better than DP-SGD's,
  and every bound quoted here holds vacuously. Nothing in the code
  measures ``s``, and ADR-0015 records the same caveat for the shipped
  transform;
- **the radius is the caller's number.** The library never invents a
  constraint set. Pass a ``radius`` that actually contains the mean
  gradients, or the estimate is biased toward the origin by an amount
  no bound covers;
- **the budget is small.** Lemma 5.3's amplification is stated for
  ``eps <= 1``; a larger budget makes the per-step cost the accountant
  charges an underestimate rather than an overestimate;
- **``delta`` is below ``1 / n ** 2``**, which Lemma 5.5's
  stopping-time bound assumes. Above it the privacy still holds and the
  step-count bound does not.

Our words, not the paper's
--------------------------
*Scale* for the paper's ``N`` and *whole* / *odd* / *even* for ``B``,
``O``, ``E``, are borrowed rather than coined. All four are public
names of `dimma.core.sampling.dyadic.DyadicDraw` — ``scale`` and
``whole`` its fields, ``odd`` and ``even`` its properties, with
``scale`` also what `dyadic.draw_scale` returns — and CONTEXT.md's
rule is that a word already carried by a public API means what the API
means. So this package speaks the sampler's words rather than second
ones of its own, and a draw and the release taken from it are named
the same thing on both sides of the call.

*Filter*, for Theorem A.4's stopping rule, and *debias*, for the
``1 / p_N`` combine, are this package's own. Each has one consumer, so
by the two-consumer rule they stay here rather than in CONTEXT.md.
*Release*, *expected batch size* and *post-processing* are the
glossary's.

Nothing is re-exported; import from `estimators`, `step` and `train`.
"""

__all__: list[str] = []
