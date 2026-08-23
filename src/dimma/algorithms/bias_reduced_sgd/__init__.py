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
**The inner estimator is Algorithm 1, not Algorithm 2.** The
pseudocode writes Gaussian ``l_1``-Recovery in all four slots; that
the substitution carries is *our* derivation and not the paper's —
`docs/research/algorithm-1-carries-algorithm-3.md` gives it and
ADR-0017 records it. The seam is `estimators.MeanEstimator`, so
Algorithm 2 later drops in without reopening the step.

**``Pi_X`` is absent from the loop.** ADR-0014 settled where a
caller-side projection composes — around the optimizer — and rejected a
projection argument on the loops. Wrap the optimizer in
`dimma.transforms.projection.l1_projected` to run the paper's own
containment.

**The filter check includes the current step's cost.** Algorithm 4's
printed ``while`` prices only steps ``s <= t - 1``; Theorem A.4's own
stopping time includes step ``t``, so dimma stops at or before
Algorithm 4's ``T``. ADR-0018 records the shift.

What float32 costs, and where it ends
-------------------------------------
Releases are float32; the debias combine and the parameter update are
float64 on the host, because ``G+ - (G-_O + G-_E)/2`` nearly cancels
and ``1 / p_N`` then multiplies it — and the rounding each release
already carries — by about ``2 ** (N + 1)``::

    relative error of G  ~  2 ** (N + 1) * eps32 / (4 * noise_multiplier)

That reaches order one near ``N = 26`` at a multiplier of 3, and
nothing raises on it: it is a tested ceiling on how high a
``max_scale`` is worth setting, and lowering ``max_scale`` is a
mechanism change rather than a truncation. ADR-0017 records both.

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
- **the radius is the caller's number.** Pass a ``radius`` that
  actually contains the mean gradients, or the estimate is biased
  toward the origin by an amount no bound covers;
- **the budget is small**, and **``delta`` is below ``1 / n ** 2``**.
  Both are the accountant's premises and are stated where the numbers
  are made, in `dimma.accounting.bias_reduced_sgd`'s module docstring.

Our words, not the paper's
--------------------------
*Scale* for the paper's ``N`` and *whole* / *odd* / *even* for ``B``,
``O``, ``E``, are borrowed rather than coined: all four are public
names of `dimma.core.sampling.dyadic.DyadicDraw`, and CONTEXT.md's
rule is that a word already carried by a public API means what the API
means. So a draw and the release taken from it are named the same
thing on both sides of the call.

*Filter*, for Theorem A.4's stopping rule, and *debias*, for the
``1 / p_N`` combine, are this package's own. Each has one consumer, so
by the two-consumer rule they stay here rather than in CONTEXT.md.
*Release*, *expected batch size* and *post-processing* are the
glossary's.

Nothing is re-exported; import from `estimators`, `step` and `train`.
"""

__all__: list[str] = []
