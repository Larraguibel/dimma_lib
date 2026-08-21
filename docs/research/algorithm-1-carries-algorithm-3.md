# Algorithm 1 carries Algorithm 3

**Status: internal derivation — ours, not the paper's.** Derived and then
adversarially verified against the PDF, 2026-08-19/21; every citation below
was checked against the text. The paper is Ghazi, Guzmán, Kamath, Kumar,
Manurangsi, *Differentially Private Optimization with Sparse Gradients*,
2024 — `papers/l1_projection.pdf` (gitignored).

## Claim

Algorithm 3 (Subsampled Bias-Reduced Gradient Estimator) may be instantiated
with **Algorithm 1** (the projection mechanism: add Gaussian noise to the
batch mean, project onto `K = B₁ᵈ(0, L√s)`) in the four slots its pseudocode
gives to **Algorithm 2** (Gaussian ℓ₁-recovery). The privacy analysis
(Lemma 5.3) carries over verbatim, and Lemma 5.4's accuracy bounds hold with
`ln(d/s) → ln(d)`:

```
bias          b  ≲ L·[s·ln(d)·ln(1/δ)]^{1/4} / √(nε)      (paper: ln(d/s))
second moment ν² ≲ L²·ln(n)·√(s·ln(d)·ln(1/δ)) / ε         (paper: ln(d/s))
```

Under the fourth root the change is a small constant — ≈ 1.08 at one-hot
Criteo scale (`d ≈ 6.4e5`, `s ≈ 40`). Algorithm 2 is therefore an accuracy
swap, not a prerequisite.

## Derivation

**(a) Lemma 5.4 uses only the second-moment estimate.** Theorem 3.4 is
invoked exactly twice in Lemma 5.4's proof: for the bias, via
`‖E[err]‖ ≤ √(E‖err‖²)` (Jensen) at full batch `n`; and for the per-scale
second moments `E[‖G⁺/⁻ − ∇F‖² | N=k]` at batches `2^{k+1}`, `2^k`, `1`. No
high-probability bound and no `β` appears anywhere in the lemma or its proof.

**(b) Algorithm 1's second moment, sparse branch.** Lemma 3.1 gives
`‖ẑ − z̄‖² ≤ 2L√s·‖ξ‖_∞` almost surely (the inner-product step needs
`±z̄ ∈ K`, which `K`'s symmetry and `conv(Sₛᵈ) ⊆ K` supply). With
`σ = 2L√(2·ln(1.25/δ))/(nε)` (Algorithm 1's own scale) and the Gaussian max
bound `E max_i |ξ_i| ≤ σ√(2·ln(2d))` — note `ln(2d)`, not `ln d`; absorbed
by `≲` — this chains to

```
E‖ẑ − z̄‖²  ≤  8L²·√(s·ln(2d)·ln(1.25/δ)) / (nε)  ≲  L²·√(s·ln d·ln(1/δ)) / (nε).
```

**(c) The other branch of the min.** Under (A.5)+(A.7) every per-example
gradient has `‖∇f‖₁ ≤ √s·‖∇f‖₂ ≤ L√s`, so every batch mean lies in the
convex `K`; projection onto `K` is then 1-Lipschitz about the mean and
`E‖ẑ − z̄‖² ≤ E‖ξ‖² = dσ²` — the paper's own "projection does not increase
the ℓ₂-estimation error" step from Theorems 3.2/3.3. Together, Algorithm 1
satisfies the full Theorem-3.4-shaped estimate

```
E‖ẑ − z̄‖²  ≲  L² · min{ d·ln(1/δ)/(nε)²,  √(s·ln(d)·ln(1/δ))/(nε) }
```

**unconditionally** — no regime condition on `d`, `m`, or the δ-range.

**(d) Pushing through Lemma 5.4.** The proof's geometric series needs the
inner estimator's second moment to decay as `1/batch` — `1/p_k = 2^k` cancels
against `1/2^k`, and `log₂ n` terms give the `ln n` factor; a slower rate
would leave a residual `2^{k/2}` and blow up at `k = M`. Algorithm 1's bound
has the `1/n` first power in every slot. Nothing else in the proof touches
the estimator's internals: the telescoping identity `E[G⁺_k] = E[G⁻_k]`
"follows from the uniform sampling and the cardinality of the used
datapoints", i.e. from the sampler, and needs only that the estimator is a
function of the batch's mean and the batch-size parameter — true of both
algorithms, whose input signatures are identical.

## The instantiation is cleaner than the paper's own

- Lemma 5.4's hypothesis `d ≳ nε√(s·ln(d/s))/√(ln(1/δ))` exists only to
  serve Theorem 3.4's regime conditions; the swapped lemma needs no
  condition on `d` at all.
- The paper applies Theorem 3.4 at batch size 1 (the `G₀` call), where its
  requirement `6·exp(−cm) ≤ δ` fails for any realistic `δ` — a gap in the
  paper's own instantiation that the Algorithm-1 instantiation does not have.
- Privacy: Lemma 5.3 uses only that each inner call is a Gaussian release of
  ℓ₂-sensitivity `2L/batch` plus post-processing. Theorem 3.3 gives exactly
  this for Algorithm 1 — with no random-matrix failure event folded into `δ`
  (Algorithm 2's branch 2 carries one, under `6·exp(−cm) ≤ δ`).
- The authors treat the slot as swappable: §5.1 opens "integrate the mean
  estimation algorithm**s** from Section 3" (plural), and its prose calls
  the inner estimator "the projection mechanism" — Algorithm 1's name —
  while the pseudocode writes Algorithm 2.

## What rests on this

`algorithms/bias_reduced_sgd` instantiates its inner mean-estimator seam
with Algorithm 1 first, built from shipped pieces (`core.noise` +
`core.projection`); Algorithm 2 is a later drop-in (see the accuracy-swap
ticket on the tracker). The swapped second-moment bound is pinned by a test
in the style of `tests/transforms/test_projection.py::
test_the_denoising_bound_of_lemma_31`, so the claim is held by the suite
rather than by this note alone.
