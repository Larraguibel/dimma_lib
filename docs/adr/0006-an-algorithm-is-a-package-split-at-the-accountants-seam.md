# An algorithm is a package, split at the accountant's seam

Each algorithm gets a package under `src/dimma/algorithms/`, containing a
`step` module and a `train` module. For each mechanism the algorithm composes,
`step` exposes a function returning that mechanism's release — everything it
makes public, and so the only thing an accountant accounts for — and a function
applying it, which is post-processing and free. An algorithm composing one
mechanism has two such functions; one composing two has four. `train` owns
stage 1, threads the state the loop carries, and returns parameters.

The seam is the point of the shape. Splitting an algorithm at "what is
released" versus "what is done with it" makes the boundary an accountant
reasons about visible in the code, instead of leaving it implicit in the middle
of a loop. Classical DP-SGD was implemented before Private SpiderBoost
specifically so this shape would be set by new code written against `core`,
rather than inherited from whichever algorithm happened to be ported first.

## Consequences

A baseline has no seam, so its `step` is one function. Count mechanisms and a
non-private algorithm has none: it releases nothing, so there is no boundary to
make visible and a release function would be a wrapper naming one that is not
there. This is the rule reaching zero rather than an exception to it.
`dimma.algorithms.sgd` is the first case.

Count mechanisms, not functions. Two mechanisms differ if they differ anywhere
— sampling rate, what is aggregated, the bound its sensitivity rests on, the
noise scale — so sharing one release function between two of them would put one
accountant's assumptions on another's code.

Everything after the release is on the apply side. Accumulating a running
estimate, projecting, updating parameters: all post-processing, all belonging
to the apply function rather than to the loop. This is what stops an algorithm
from doing its distinctive work in `train` and leaving `step` a stub, and it is
what keeps a step a single compiled call.

Hyperparameters are keyword-only arguments and there is no configuration
object. Three bare scalars — expected batch size, clipping norm, noise
multiplier — are easy to transpose, and transposing them produces a wrong
privacy guarantee rather than a crash. Whether a later algorithm earns a config
object is open; it should be argued for rather than ported in.

Training loops report no metrics. Evaluating a model on the training data is
another access to it, costing budget the algorithm does not account for, so
that call belongs at the call site where it is visible and not in a callback
inside the private loop.

For an algorithm whose estimator accumulates across steps, locating the seam is
the substantive work of implementing it, not a formality.
