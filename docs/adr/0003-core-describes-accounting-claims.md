# `core` describes, `accounting` claims

`core` may state what a sampler samples, or which assumption a primitive
satisfies — those are facts about the code. The moment a number is called an
epsilon, a noise scale is calibrated to a budget, or a transformation is called
free, that is a claim about a mechanism, and it lives in `accounting`.

The obvious alternative was to let each primitive carry its own accounting, so
that a sampler could report the epsilon it implies. We split them because a
primitive is not in a position to know the mechanism it ended up inside: the
same Poisson draw is exactly accounted for in one algorithm and outside the
standard analysis in another, depending on what the surrounding loop does with
it. Putting the claim next to the primitive would attach a guarantee to code
that cannot check the conditions the guarantee depends on.

## Consequences

Every function in `accounting` states the mechanism it assumes, because that
is the only thing making its number meaningful; a loop that departs from the
stated mechanism silently invalidates it. Standard mechanisms share
`accounting/sampling.py` over Google's `dp-accounting`. An algorithm earns its
own module beside it only when its mechanism falls outside those assumptions
or is bounded too loosely by them, and such a module travels with its
algorithm rather than being general-purpose.
