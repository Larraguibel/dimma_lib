# Private SpiderBoost

Private SpiderBoost (Arora et al., ICML 2023, Algorithm 2), in
`dimma.algorithms.spiderboost`. A variance-reduced private method, and the
first non-classical algorithm in dimma. Its package docstring tabulates
the paper's notation, symbol by symbol, against the code that carries it.

## Two mechanisms, one loop

The algorithm composes *two* mechanisms rather than one. Every
`anchor_interval` steps, an **anchor** step draws a large batch and
releases a privatized gradient. Every other step is a **variation** step:
it draws a smaller batch, evaluates per-sample gradients at the current
*and previous* parameters, and releases the privatized **difference**. The
running estimate is the previous estimate plus that release, and the
update descends along it. The release/apply split in `step` therefore has
four functions — `anchor_release`/`anchor_step` and
`variation_release`/`variation_step` — one release per mechanism, because
sharing a release function between two mechanisms would put one
accountant's assumptions on the other's code.

Note where the noise goes: a variation release adds noise to a quantity
already divided by the batch size, so its noise scales are scales on the
*released estimate* — unlike [DP-SGD](dp-sgd.md), which noises the sum at
`σ·C`. Both are written the way their papers write them, and an accountant
must match the convention of the code that ran.

## No clipping — sensitivity by assumption

Algorithm 2 has no clipping line, and dimma adds none: stage 4 is absent
*by decision*. The privacy rests on the function class the paper assumes —
the per-sample loss is `L0`-Lipschitz and `L1`-smooth for every example —
which bounds each per-sample gradient by `L0` and each per-sample gradient
difference by `min(L1·‖w_t − w_{t−1}‖, 2·L0)` without any operation making
it so. Reading the paper's noise scales backwards recovers exactly those
two bounds; the cap on the variation noise exists because the second bound
takes over when the parameters move far.

The tempting alternative was to clip anyway, so the bounds hold by
construction whatever loss a caller supplies — free whenever the
assumptions hold, since a clip that never binds is the identity. dimma
declined it, accepting the risk, to run the algorithm as it was conceived,
and because clipping costs more here than in DP-SGD: the analysis needs
each release to be an *unbiased* estimate, and a binding clip breaks that.
In DP-SGD a biased step is one biased step; in a variance-reduced method
the bias enters the running estimate and stays there for the rest of the
phase.

!!! warning

    **The guarantee is conditional on constants the loop cannot check.**
    If the supplied loss is not Lipschitz with the `L0` used to calibrate,
    or not smooth with the `L1`, the noise is calibrated against a
    sensitivity the data can exceed and the reported `(ε, δ)` is false —
    silently, with no crash. The same holds for the step size the paper's
    analysis fixes at `1/(2·L1)`.

## Where the constants come from

For the model dimma ships, that conditional guarantee is closed by
construction rather than left to the caller's word. For logistic
regression with sigmoid BCE, both constants are known in closed form from
one number — the largest augmented feature norm — and that is exactly the
number that must not be *measured*, because a bound read off the data is a
non-private query whatever the wall-clock distance between reading it and
using it.

So dimma imposes the bound instead of measuring it. A per-record map in
`dimma.datasets.preprocessing` rescales each feature vector to norm at
most `R`, touching one record at a time — it reads no aggregate, so it
costs no budget — and afterwards `R` is a fact about the preprocessing
rather than a claim about the data.
`dimma.accounting.lipschitz.logreg_bce_constants` then produces
`L0 = R`-derived constants and the step size together, as one triple, from
`R` and a bias flag alone: the function never takes the data, and there is
deliberately no code path from a feature array to a constant. `R` itself
remains a hyperparameter with teeth — the variation noise grows like `R²`
and the prescribed step size shrinks like `1/R²` — and what `R` may *not*
be chosen by is the data.

## Accounting

`dimma.accounting.spiderboost` prices the run as two Poisson-subsampled
Gaussian schedules — the anchors at their rate and the variations at
theirs — composed together in a single accountant over `dp-accounting`.
Composing jointly rather than summing two separately-converted epsilons is
worth 46–63% of the budget in the worked configurations quoted in that
module: summing is sound, but pays to convert to ε twice.

What the accountant deliberately does not do is transcribe the paper's
closed-form noise scales. Those are asymptotic in an unspecified universal
constant and derived for fixed-size batches, while dimma samples by
Poisson; calibrating numerically against the actual mechanism gives a
number that matches what ran. Privacy is claimed; the paper's convergence
*rate* is not, since it is proved under the paper's own parameter
settings, and calibrating numerically produces different scales — the rate
does not travel with the ε.

Both this algorithm and DP-SGD report ε through the same `dp-accounting`
machinery, RDP by default, so a comparison between them is not also a
comparison between two accounting methodologies.

!!! note

    **Evaluated in**
    `notebooks/comparisons/spiderboost-vs-dp-sgd-on-criteo.ipynb`,
    head to head with DP-SGD at ε = 3. The finding — a null, and an
    instructive one — is on the [evaluation page](../evaluation.md).
