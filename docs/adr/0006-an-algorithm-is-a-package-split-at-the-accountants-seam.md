# An algorithm is a package, split at the accountant's seam

Each algorithm gets a package under `src/dimma/algorithms/`, containing a
`step` module and a `train` module. `step` is split into two functions: one
returning the privatized gradient — everything the mechanism releases, and so
the only thing an accountant accounts for — and one applying it, which is
post-processing and free. `train` owns stage 1, threads the optimizer state and
the noise key, and returns parameters.

The seam is the point of the shape. Splitting an algorithm at "what is
released" versus "what is done with it" makes the boundary an accountant
reasons about visible in the code, instead of leaving it implicit in the middle
of a loop. Classical DP-SGD was implemented before Private SpiderBoost
specifically so this shape would be set by new code written against `core`,
rather than inherited from whichever algorithm happened to be ported first.

## Consequences

Hyperparameters are keyword-only arguments and there is no configuration
object. Three bare scalars — lot size, clipping norm, noise multiplier — are
easy to transpose, and transposing them produces a wrong privacy guarantee
rather than a crash. Whether a later algorithm earns a config object is open;
it should be argued for rather than ported in.

Training loops report no metrics. Evaluating a model on the training data is
another access to it, costing budget the algorithm does not account for, so
that call belongs at the call site where it is visible and not in a callback
inside the private loop.

For an algorithm whose estimator accumulates across steps, locating the seam is
the substantive work of implementing it, not a formality.
