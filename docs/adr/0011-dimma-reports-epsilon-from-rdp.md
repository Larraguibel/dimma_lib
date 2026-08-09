# dimma reports epsilon from RDP

Every function in `accounting/` takes a `method` and defaults it to `"rdp"`, so
both algorithms report ε through `dp-accounting`'s Rényi accountant unless a
caller says otherwise. Which one was used is reported alongside the number.

RDP is the modern form of the moments accountant Abadi et al. introduced, and it
is what both papers dimma implements say their guarantee rests on — Arora et al.
name it explicitly before Theorem B.2. Reporting through it means the number
carries the lineage of the analysis the algorithm was designed under.

The alternative was PLD, which is numerically tighter for the subsampled
Gaussian and would let dimma claim a smaller ε for the same run. We declined it
as the default because the choice cancels in the comparison dimma exists to
make — both algorithms report through the same accountant, so it moves both
numbers together and neither ranking nor gap depends on it — and because RDP
errs by over-reporting ε, which is the safe direction for a claim. It stays
available as an argument, for a caller who wants the tighter number and will say
which accountant produced it.

## Consequences

The cost is real and regime-dependent, so it is recorded rather than assumed
small. Calibrating to a target ε, RDP needs more noise than PLD by:

| sampling rate, steps | ε\* = 1 | ε\* = 3 | ε\* = 8 |
|---|---|---|---|
| `q` = 0.001, 10000 | +32% | +8% | +4% |
| `q` = 0.0005, 20000 | +35% | +9% | +4% |
| `q` = 0.01, 10000 | +6% | +5% | +3% |

Loose budgets cost a few percent at any sampling rate. A tight budget on a large
dataset — which is where Criteo sits — costs a third more noise. If the
evaluation reports at ε = 1, this decision should be revisited there rather than
inherited; that is a question for the evaluation, not for this ADR.

Read the other way, as ε at fixed noise, the same gap is 7–13% at `q` = 0.01 and
28–79% at `q` = 0.001. The two directions are not interchangeable and a claim
about "how much tighter PLD is" has to say which one it means.

Nothing in `algorithms/` changes. Both training loops take noise scales, so the
accountant is upstream of them, and this decision is visible only where a budget
is converted.
