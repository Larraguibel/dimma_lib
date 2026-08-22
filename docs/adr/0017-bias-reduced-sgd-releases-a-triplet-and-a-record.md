# Bias-reduced SGD releases a triplet and a record, and its mean estimator is a seam

Ghazi et al. 2024's Algorithm 3 calls a private mean estimator four times a
step: on a batch `B` of `2^(N+1)` examples, on each of its two halves `O` and
`E`, and on one record `I` drawn independently. Counting mechanisms rather than
calls gives **two**, so per ADR-0006 the step is four functions. `batch_release`
returns all three of the batch's private means in one `BatchRelease`, because
they come out of a single draw of `B` and Lemma 5.3 amplifies them once,
jointly, at rate `2^(N+1)/n`; three release functions would invite an
accountant to compose three independent amplifications, which is not what ran.
`single_release` is the second mechanism, amplified at `1/n`. The `1/p_N`
debias combine and the parameter update are both apply-side: they are
arithmetic on already-released quantities.

The inner estimator is a seam — `estimators.MeanEstimator`, a name, a claim and
a callable — instantiated with the paper's **Algorithm 1** (perturb the mean,
project onto the `l_1` ball) and not with Algorithm 2 (Gaussian
`l_1`-recovery), which is what the pseudocode writes in all four slots. That
substitution is ours, not the paper's; `docs/research/algorithm-1-carries-algorithm-3.md`
derives it, the privacy analysis carries over verbatim, and the accuracy bounds
lose only `ln(d/s)` for `ln d`. It is also cleaner than the paper's own
instantiation: no regime condition on `d`, no random-matrix failure event
folded into `delta`, and no break at the batch of one that Algorithm 3's `G_0`
slot needs and Theorem 3.4 cannot cover. The seam exists so that Algorithm 2 is
later a drop-in rather than a rewrite, and so that an accountant reads which
estimator ran off `claim`'s type instead of inferring it.

The seam stays inside the package. ADR-0015 pre-authorised a `mechanisms/`
package for the day a complete analysed map has a caller; one caller is not
that day. Promote it when a second one exists.

Clipping supplies the paper's `L`, enforced by stage 4 — ADR-0012's pattern
rather than ADR-0009's, and the number lives in `claim.clip_norm` so the bound
and the noise calibrated against it cannot be given different values. The
radius stays the caller's number, per ADR-0015, with `clip_norm * sqrt(s)` as a
docstring mapping: the library states no sparsity it cannot check.

## Consequences

Releases are float32 and the combine is float64, on the host. `G+ - (G-_O +
G-_E)/2` nearly cancels and `1/p_N` then multiplies it by about `2^(N+1)`, so
the combine runs in NumPy — JAX offers no float64 without process-wide
`jax_enable_x64`, which would change DP-SGD's dtypes and spoil the controlled
comparison. This has an honest ceiling and it is *not* removed by the float64
combine: each float32 release already carries rounding of order `eps32 *
clip_norm` from its own sum, perturbation and projection, and the same
`2^(N+1)` amplifies it, so the relative error of `G` behaves like `2^(N+1) *
eps32 / (4 * noise_multiplier)` and reaches order one near the top of a
full-size ladder. The law is tested, the ceiling is in the package docstring,
and nothing raises on it. It also bounds usefully how high `max_scale` is worth
setting.

`max_scale` is a mechanism parameter, not a padding cap. ADR-0007's rule does
not bite — `2^(M+1) <= n` always, so no draw can be oversize — but its
principle does: lowering `max_scale` runs `TGeom(M')`, a different and fully
analysable law whose debias weights, per-step cost and bias bound all follow
along, whereas truncating a drawn batch to a memory bound would silently break
`E[G+_k] = E[G-_k]`. Per-step device memory is then exactly `2^(M'+1) * d`
floats, and a caller who sets it too high gets an OOM from XLA rather than a
quietly different mechanism.

Batches are compiled per shape, which bends ADR-0006's "a step is a single
compiled call". The batch release's leading axis is `2^(N+1)`, known host-side
before any device work, so `jax.jit`'s own cache is the per-scale table:
`max_scale + 1` programs at most — about 25 at Criteo scale — populated lazily,
and the large ones only if their scale is ever drawn. The single release has one
shape for a whole run and is bound separately so that the per-scale table never
retraces it. The alternative, a padding cap that never truncates, would have to
be about `n/2`, making every step pay `O(n)` per-example gradients including the
half of all steps that touch three records; it buys nothing, since the memory
that is scarce is the per-example gradients padding would only add to.

Deferred, and named here so it is a decision rather than an omission: a
`lax.scan` over fixed-size microbatches accumulating the three masked sums would
make device memory `O(chunk * d)` independent of the scale, restoring the
paper's full `M`. That is a second design, not this port.
