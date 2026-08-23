# One-hot Criteo is its own loader and returns index/value pairs

`load_criteo_one_hot` one-hot encodes `C1..C26` at their native train-split
cardinalities and returns a `SparseTabularSplit` — for each row, the 39 indices
it occupies and the 39 values it puts there — rather than a `TabularSplit`
holding a dense matrix.

The width is why. At the default split the vocabulary fitted on the training
rows is 551,908 distinct IDs, out of the 623,115 the file carries across all of
them, so the encoding addresses 551,947 columns. A dense float32 matrix of that
width over a million rows is 2.2 TB. The pairs are 250 MB, and the model reads
them without ever forming the row: `logreg.forward_sparse` gathers 39 weights
and dots them against 39 values, which is what `forward` computes on the row
the pair implies.

## A separate function, not a third value on the `features` axis

ADR-0008 makes loading modes independent parameters rather than preset names,
on the grounds that a name hides the half of the behaviour that does not fit in
it. A `features="one_hot"` would hide more than a name usually does: it would
change the return type. A caller reading `load_criteo(features=...)` at the call
site would have no way to see that one value of that argument hands back six
arrays and a width instead of two matrices, and the type checker would not see
it either. The encoding is therefore in the function name and nothing else is:
`preprocess`, `root`, `download`, `test_fraction`, `seed`, `device` and
`feature_norm_bound` keep their meanings, and what they did is recorded in
`metadata` and printed once, exactly as for the other eight modes.

The split is shared rather than reproduced. Both loaders read the file and cut
it through `_read_frame` and `_split_frame`, so two modes at the same seed and
fraction hold out the same rows because they ran the same code, not because two
copies of a permutation agree.

## No `standardize` axis

Centring a one-hot column gives every row a non-zero entry in it. Standardizing
this encoding turns 39 stored values per row into 551,947, which is the dense
matrix this loader exists to avoid, so the axis is not offered rather than
offered and refused. Scaling without centring would preserve sparsity but be an
axis of a different shape from `load_criteo`'s, described by different prose and
comparable with nothing; that is a mode to add when something needs it.

## The reserved unseen slot, and why `s` is exact

Each column's block is `card_j + 1` wide, the extra slot holding IDs the
training split never saw. A test row with an unfamiliar category lands there
instead of storing nothing, so every row in both splits carries exactly 39
entries with distinct indices. The weight on a reserved slot never appears in a
training gradient and so never trains, which is the whole price.

That price buys an exact `s`. For a linear model the per-example gradient is
`(σ(z) − y)·x̄`, so its support is the row's support: 39 feature coordinates and
the bias, at every example and every parameter value. Assumption (A.7) of Ghazi
et al. 2024 asks for a sparsity level, and here it is a property of the encoding
rather than a claim about the data — which is also what the shipped transform's
`L·√s` radius rests on, per ADR-0015. An encoding that dropped unseen categories
would make `s` a per-record maximum, and the bound would hold by inspection of
the data rather than by construction.

This discharges ADR-0015's caveat. That ADR recorded that the projection's
denoising benefit presupposes sparse per-example gradients, and that on Criteo
as then loaded the honest outcome might be ~nothing "until a model with sparse
gradients exists". This encoding plus `forward_sparse` is that model: the
gradient's support is 39 coordinates and the bias, so the assumption now holds
on a real dataset rather than only on the transform's synthetic tests.

## No `standardize` key in `metadata`

ADR-0013 says `metadata` carries `"standardize"` always, on the grounds that a
chain that standardized and one that did not must never be indistinguishable.
This loader departs from that letter deliberately: it has no such axis, so the
key would assert an axis that does not exist rather than record which way one
fell. `"encoding": "one_hot"` is the marker instead — it identifies the loader,
and this ADR says what that loader does and does not standardize.

## Consequences

`SparseTabularSplit` carries `num_features` as a field, not as metadata. A model
is initialised with it — `init_params(key, split.num_features)` — so it is part
of the split's shape rather than a note about how the split was made, and the
one number that must not be reconstructed at a call site. `metadata` mirrors it
for provenance alongside `"n_categories"` and `"column_offsets"`, which are what
a later reader needs to say which column an index came from.

`feature_norm_bound` applies to `val` and means what ADR-0012 says it means. The
indices within a row are distinct, so the row's ℓ₂ norm is its `val` vector's,
and capping one caps the other. It stays last in the chain for the reason
ADR-0012 gives.

`dimma.models.logreg` grows a second forward over the same parameters and the
same model. The two return the same logit and differ in what they take, so a
run switching between them changes representation and not architecture. The
model still declines to know what a feature means: `forward_sparse` is told
coordinates and values, and nothing about what sits at them.

The sparsity buys the data representation and the forward, and not the
per-example gradient buffer. `jax.grad` with respect to `w` returns a dense
`(551,947)` row per example, so the vmapped per-example clipping DP-SGD needs
allocates `B × d` float32 — 565 MB at `B = 256`, 2.3 GB at `B = 1024`. A sparse
clip-and-scatter path is its own ticket, not something this encoding delivers.

Nothing else in the library takes a `SparseTabularSplit`. The algorithms consume
per-example gradients and never a dataset, so the type reaches them only through
whatever the caller vmaps; a stage that wanted to index a split by row would
need to say which of the two shapes it means.
