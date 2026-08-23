# Package map and conventions

```
src/dimma/
├── core/                    the pipeline stages
│   ├── sampling/            stage 1 — one module per sampler
│   ├── gradients.py         stage 3 — per-sample and batch gradients
│   ├── clipping.py          stage 4
│   ├── aggregation.py       stage 5 — sum/average, Poisson masking
│   ├── noise.py             stage 6 — Gaussian and Laplace
│   ├── updates.py           stage 7 — sgd, and the seam optax also fits
│   ├── pytree.py            pytree vector-space ops
│   └── projection.py        ℓ₁-ball geometry
├── accounting/              where the privacy claims live
├── algorithms/              one package per algorithm
├── transforms/              post-processing that composes across algorithms
├── datasets/                loaders; no algorithm imports these
├── models/                  reference models; no algorithm imports these
└── metrics/                 evaluation; beside the loop, not inside it
```

The annotated version, file by file, is in the repository's `README.md`.
What this page adds is the reasoning that holds the shape together.

## `core` describes; `accounting` claims

The deepest seam in the library is between facts about code and claims
about mechanisms. `core` may state what a sampler samples or which
assumption a primitive satisfies. The moment a number is called an
epsilon, a noise multiplier is calibrated to a budget, or a transformation
is called free, it belongs in `accounting`.

The obvious alternative was to let each primitive carry its own
accounting, so a sampler could report the epsilon it implies. The split
exists because a primitive is not in a position to know the mechanism it
ended up inside: the same Poisson draw is exactly accounted for in one
algorithm and outside the standard analysis in another, depending on what
the surrounding loop does with what it drew. Putting the claim next to the
primitive would attach a guarantee to code that cannot check the
conditions the guarantee depends on.

The consequence runs through `accounting`: every function there states the
mechanism it assumes, because that statement is the only thing making its
number meaningful. A training loop that departs from the stated mechanism
silently invalidates the number — no crash, just a false ε. Standard
mechanisms share `accounting/sampling.py`, built over Google's
`dp-accounting`; an algorithm earns its own accounting module only when
its mechanism falls outside those assumptions or is bounded too loosely by
them, and such a module travels with its algorithm rather than being
general-purpose.

## An algorithm is a package, split at the release

Each algorithm is a package under `dimma.algorithms` with a `step` module
and a `train` module. For each mechanism the algorithm composes, `step`
exposes a function returning that mechanism's *release* — everything it
makes public, and so the only thing an accountant accounts for — and a
function applying it, which is post-processing. `train` owns stage 1,
threads the loop's state, and returns parameters.

The seam is the point. Splitting an algorithm at "what is released" versus
"what is done with it" makes the boundary an accountant reasons about
visible in the code, instead of implicit in the middle of a loop.
Everything after the release — accumulating a running estimate,
projecting, updating parameters — is on the apply side, which is also what
keeps a step a single compiled call.

Mechanisms are counted, not functions. Two mechanisms differ if they
differ anywhere — sampling rate, what is aggregated, the bound the
sensitivity rests on, the noise scale — so sharing one release function
between two mechanisms would put one accountant's assumptions on another's
code. A [baseline](../algorithms/baselines.md) composes *zero* mechanisms:
it releases nothing, so its `step` is one plain function — the rule
reaching zero, not an exception to it.

Two smaller conventions with the same motivation:

- **Hyperparameters are keyword-only, and there is no configuration
  object.** Expected batch size, clipping norm, and noise multiplier are
  three bare scalars; transposing them produces a wrong privacy guarantee
  rather than a crash, so every call site is forced to name them.
- **Training loops report no metrics.** Evaluating a model on the training
  data is another access to it, costing budget the algorithm does not
  account for. That call belongs at the call site, where it is visible —
  not in a callback inside the private loop.

## Around the loop: `datasets`, `models`, `metrics`

Three packages sit beside the pipeline rather than inside it, and no
algorithm imports any of them.

`datasets` exists so a method can run against real data without assembling
a loader, and so two compared algorithms see byte-identical inputs. One
module per dataset, each exposing a `load_*` function; what a loader did
to the data is recorded in the returned split's `metadata` rather than
implied by a mode name. Loading options are independent axes — which
columns, whether they are preprocessed, whether they are standardized —
never preset names, because a preset name reliably hides the half of the
behaviour that matters for interpreting a result.

`models` ships reference models as pairs of plain functions over plain
pytrees — `init_params` and a `forward` — with matching per-sample losses
in `dimma.models.losses`. See [working with pytrees](../pytrees.md) for
why that contract is all a model needs to be here.

`metrics` is the evaluation stack, threshold-free where it selects and
thresholded only where it reports; the reasoning is on the
[evaluation page](../evaluation.md).
