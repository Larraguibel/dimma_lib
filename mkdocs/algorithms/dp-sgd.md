# DP-SGD

Classical DP-SGD (Abadi et al., CCS 2016, Algorithm 1), in
`dimma.algorithms.dp_sgd`. The reference every non-classical method in
dimma is measured against, and the shortest path through the
[pipeline](../library/pipeline.md): Poisson subsampling, one gradient per
example, clip, sum, Gaussian noise, scale, step.

## Stage by stage

| Stage | Choice |
|---|---|
| 1 | `sampling.poisson` at sampling rate `q = L/N` (`L` the expected batch size) |
| 3 | `gradients.per_sample_grads` |
| 4 | `clipping.per_sample_clip` at clipping norm `C` |
| 5 | `aggregation.sum_over_batch` |
| 6 | `noise.add_gaussian` at scale `σ·C` (`σ` the noise multiplier) |
| 7 | `pytree.scale` by `1/L`, then `updates.apply` |

The package docstring
(`src/dimma/algorithms/dp_sgd/__init__.py`) tabulates Algorithm 1's
notation line by line against these primitives, so a signature can be read
without the paper open. Note the convention: the noise is on the **sum**,
at `σ·C`, because that is where the sensitivity bound lives — the clipping
norm bounds each per-sample gradient, so adding or removing one example
moves the sum by at most `C`.

## Where the pieces live, and why

Stage 1 is in `train`, not `step`. A Poisson draw has data-dependent
cardinality and cannot be compiled, so the draw happens host-side on a
NumPy generator, padded to a fixed cap with a mask; everything from stage
3 onward is one compiled call in `step`, split into a function returning
the *release* — the privatized gradient, the only thing an accountant
prices — and a function applying it, which is post-processing. Why every
algorithm takes that shape is on the [package map](../library/map.md).

The padding cap is a memory bound, never a privacy parameter. When a draw
exceeds it, the sampler **raises rather than truncates**: discarding
examples that were genuinely drawn would make inclusion dependent across
examples, and independence is exactly what subsampling amplification
assumes. Truncating silently would leave the standard accountant returning
a number for a mechanism that no longer ran. The failure is rare and the
cap is exposed — passing the dataset size makes it impossible, since the
draw can never exceed `n` — and a caller who genuinely wants truncation
imports `poisson_truncated`, whose different mechanism is then visible in
the import line and which no standard accountant covers.

## Accounting

`dimma.accounting.sampling.poisson_gaussian_epsilon` prices the run: a
schedule of Poisson-subsampled Gaussian releases at the run's sampling
rate and noise multiplier, composed over its steps, through Google's
`dp-accounting`. `calibrate_noise_multiplier` runs the same machinery in
the other direction, from a target budget to the multiplier.

Two properties of the reported number are deliberate:

- **It is RDP by default.** RDP is the modern form of the moments
  accountant Abadi et al. introduced — the lineage of the analysis the
  algorithm was designed under. PLD is numerically tighter for the
  subsampled Gaussian and would let the same run claim a smaller ε; it
  stays available as an argument, and which accountant produced the
  number is reported alongside it. The default is RDP because the choice
  cancels in the comparisons dimma exists to make — both algorithms
  report through the same accountant, so it moves both numbers together —
  and because RDP errs by over-reporting ε, the safe direction for a
  claim. At tight budgets on large datasets the gap is real (roughly a
  third more noise at ε = 1 and Criteo-like sampling rates), which is a
  reason to revisit per evaluation, not to hide.
- **It is a claim about a mechanism, not about the code.** The number is
  a guarantee only if the run matched the mechanism the accountant
  assumes — the loop takes noise scales and never computes an epsilon,
  precisely so that the claim sits where its assumptions are stated.

!!! note

    **Evaluated in** `notebooks/tuning/dp-sgd-on-one-hot-criteo.ipynb`
    (alone, over its hyperparameter grid) and
    `notebooks/comparisons/dp-sgd-vs-sgd-baseline-on-criteo.ipynb`
    (against the non-private baseline). Headline results are on the
    [evaluation page](../evaluation.md).
