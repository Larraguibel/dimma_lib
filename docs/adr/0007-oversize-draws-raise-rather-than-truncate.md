# An oversize Poisson draw raises; truncation is a different mechanism

Poisson sampling produces a variable number of examples, so the draw is padded
to a fixed cap for everything downstream to be compilable. When a draw exceeds
that cap, `dimma.core.sampling.poisson` raises. Truncating instead — discarding
examples that were genuinely drawn — lives in a separate module,
`poisson_truncated`, so the choice appears in the import line.

Truncating silently would be the friendlier behaviour and is what a reader
might assume a padding cap implies. We refuse it because capping *inside* the
support of the draw makes inclusion dependent across examples, and independence
is exactly what subsampling amplification assumes. The standard accounting
would then no longer apply to the mechanism that actually ran, while continuing
to return a number.

## Consequences

A long run can fail near its end, at roughly 1e-9 per step under the default
margin. That is accepted: catching it would mean truncating or redrawing, and
both change the mechanism. The cap is exposed so a caller can raise the margin,
and passing the dataset size makes the failure impossible — the draw is
`Binomial(n, q)`, so `n` is a cap the mechanism already carries and clamping to
it removes no probability mass. That costs a batch the size of the dataset,
which is why it is not the default.

`poisson_truncated` has no accountant of its own. It is not known whether the
standard bound over- or under-states its true cost: truncation couples the
draws, which plausibly costs privacy, and lowers each example's marginal
inclusion probability, which plausibly saves some. Earlier code asserted it was
a lower bound; that claim had no proof behind it and has been withdrawn.
