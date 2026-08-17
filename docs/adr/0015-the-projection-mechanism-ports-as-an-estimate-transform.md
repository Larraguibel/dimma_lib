# The projection mechanism ports as an estimate transform

The deprecated repo implements Ghazi et al. 2024's Algorithm 1 whole: a
`mechanisms/` package holding perturb-then-project as one function, and an
`accounting/projection.py` calibrating the noise it adds to a one-shot mean
release. Porting it raised a question the map had only half settled: the
front's name is *transforms* because a transform carries no analysis — but
the paper's Projection Mechanism carries one, so where does it land?

Only the projection step lands. It is `l1_projected_estimate`, a second
wrapper at ADR-0014's seam, projecting the incoming estimate before the
wrapped rule runs — the shape the paper itself uses in Section 5, where the
projection post-processes the noisy gradient a mechanism already released.
In dimma that mechanism is the algorithm's own; its noise is already
calibrated and accounted where that algorithm says so, and the projection
adds nothing an accountant sees.

A `mechanisms/` package was rejected: dimma has no caller that releases a
one-shot private mean, and a package with one member and no call site is
the old repo's shape, not a need. Porting the mean-release noise scales
into `accounting/` was rejected the same way — a claim with no claimant.
Splitting the wrapper from ADR-0014's seam (a projection argument on the
loops, or inside each step) was rejected in ADR-0014 already, and nothing
about the estimate side reweighs it.

## Consequences

Lemma 3.1 — the denoising bound that motivates projecting the estimate —
is a property of the geometry and the radius, not of the noise, so it is
pinned in the transform's tests on synthetic sparse estimates rather than
claimed by the code. The radius is the caller's number; the paper's `L√s`
stays a docstring mapping.

The bound's benefit presupposes sparse per-example gradients. The reference
model's 39-feature dense logistic regression has dense gradients, so on
Criteo as currently loaded the denoising may be worth ~nothing — ADR-0009's
pattern applies: the assumption is stated, and the honest outcome may be
that it does not hold here until a model with sparse gradients exists.

If a one-shot private mean release ever gets a caller, the deprecated
repo's `mechanisms/projection.py` and its calibration are the port source,
and the *mechanism* word becomes correct for that complete map.
