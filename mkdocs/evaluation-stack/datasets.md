# Datasets

`dimma.datasets` is a convenience, and no algorithm imports it. It exists
so that a method can be run against real data without assembling a loader
first, and so that two algorithms compared against each other are
compared on byte-identical inputs. The dataframe stack it needs stays out
of the base install:

```bash
pip install 'dimma[datasets]'
```

## Criteo, and what the pinned file actually holds

The library is evaluated on the Criteo 1M click-prediction sample. It is
the workload this kind of method is normally reported against — a large,
imbalanced, real binary classification problem — which is what makes a
result on it comparable with the numbers the DP optimization literature
quotes.

`dimma.datasets.criteo.load_criteo` returns 1,000,000 rows split
800,000 train / 200,000 test, with 13 integer columns `I1..I13`, 26
categorical columns `C1..C26`, and a binary click label. The base rate —
the fraction of positive records — sits near 25%, which is the number
every metric on the [metrics page](metrics.md) is read against.

The file is pinned by URL **and SHA256**. That is what makes a dataset a
fixed input rather than whatever the upstream URL served today: two runs
agreeing on the digest were trained on the same bytes.

!!! warning

    **What is stored is not raw Criteo.** In the pinned file `I1..I13`
    arrive already scaled into `[0, 1]` against a per-column cap — the
    distinct values of `I1` are the 21 multiples of 0.05, of `I3` the 101
    multiples of 0.01. There is no missing value anywhere and no negative
    value. The upstream sample documents none of this, so the scaling is
    a property of the file rather than a step this loader can point at.
    It is the reason several steps below are no-ops here.

The data is CC-BY-NC-SA 4.0 — non-commercial, share-alike. The loader
prints the attribution once per process, to **stderr** rather than
stdout, so that a caller piping results cannot mistake the notice for
output.

## Loading options are axes, never a mode name

An earlier version of this loader had a mode called `"integer"`. The name
said which columns came back and said nothing about the median fill, the
`log1p` and the standardization it also applied. That is the failure mode
of preset names: the half of the behaviour that does not fit in the name
goes missing, and it is reliably the half that matters for interpreting a
result.

So there is no mode. There are independent axes, and the combinations are
whatever the caller asks for:

| Axis | Options | Default |
|---|---|---|
| `features` | `"numeric"` (the 13 `I*`) or `"all"` (39) | `"numeric"` |
| `preprocess` | fitted column maps on or off | `True` |
| `standardize` | centre and scale on or off | `False` |
| `feature_norm_bound` | a bound `R` on each row's ℓ₂ norm, or none | `None` |
| `test_fraction`, `seed` | how the split is cut | `0.2`, `0` |
| `device` | `"cpu"`, `"gpu"` (`"cuda"` is an alias) | `"cpu"` |

Recording is half the decision. Every load writes what it did into the
returned split's `metadata` as prose, and prints it once per distinct
combination. A split carries its own provenance rather than leaving it at
the call site, and the eight combinations produce eight distinct
descriptions — a mode described by another mode's prose is exactly the
failure the axes exist to prevent.

## Fitted maps cost budget; per-record maps do not

Every preprocessing function states whether it reads across records, and
that line organizes the rest of this page.

**Fitted maps** read the training split to produce their parameters, and
every statistic here — medians, category frequencies, means, standard
deviations — is fitted on the training split alone and then applied to
both. This is the standard benchmark convention and it is *not* a private
operation: those statistics depend on the training data and are accounted
for in no privacy budget. Any ε reported over this data inherits that
caveat and should state it, which is why every notebook states it next to
its ε rather than absorbing it.

The chain, when `preprocess=True`:

- **Missing values** are filled with the train-split median.
- **`I1..I13` are clipped at 0 and passed through `log1p`.** On this file
  both are close to no-ops, and `log1p` over values already in `[0, 1]`
  compresses them monotonically into `[0, log 2]` rather than taming a
  long tail — it is not doing the job a log transform usually does. The
  steps stay because they are what raw Criteo integers would need, and
  dropping them would make the chain wrong for any file but the one
  currently pinned.
- **Categoricals are frequency-encoded**, each ID replaced by its
  relative frequency in the training split. There is no hashing trick and
  no vocabulary cutoff: one float per column instead of one per category,
  where 26 columns whose cardinalities reach `10^5` would otherwise
  become hundreds of thousands of one-hot columns. An ID unseen in
  training encodes as `0.0`, which is the frequency it was observed with.

**Per-record maps** touch one record at a time and read nothing across
them, so they are free. `dimma.datasets.preprocessing.cap_feature_norms`
is the first one in the library.

## Standardization is its own axis, and it defaults off

Standardization was originally inside `preprocess`, and pulling it out is
the same decision one level down. The median fill, the clip, the `log1p`
and the frequency encoding all leave a column roughly where it was; the
standardization rescales every column to unit variance, and on this file
it is the entire difference between the stored data and what comes out.
Bundling the only step with an effect under a switch named for the four
without one hides exactly what a reader needs.

The default is the surprising part, and the reason is privacy arithmetic
rather than conditioning. Standardizing sets each column's scale to 1,
which lands a typical row's ℓ₂ norm near `sqrt(d)` for `d` features
regardless of where it started. On this file that is a large increase:
over the numeric chain the largest row norm goes from about 2.0 to about
15.1, and over all 39 features from about 2.7 to about 16.7.

That largest norm is not a summary statistic here — it is the quantity
[Private SpiderBoost's constants are computed from](../algorithms/spiderboost.md#where-the-constants-come-from).
A bound `R` that leaves standardized Criteo intact is roughly seven times
one that leaves the unstandardized chain intact, and the smoothness
constant goes as `R²`: the noise on the variance-reduction step grows by
a factor near fifty, and the step size shrinks by the same. The
alternative — keeping `R` small and capping anyway — discards most of
each record, because the cap now binds on nearly every row rather than on
the tail. Either way, standardization is paid for in the budget.

This is not a claim that standardizing is wrong. On a heavy-tailed
feature set with columns spanning orders of magnitude it is what makes a
run converge at all, and every executed notebook passes
`standardize=True` explicitly. It is a claim about which way a default
should fall when the parameter is invisible: the cheap chain by default,
and a caller who wants the conditioning asks for it and sees the price in
the ε they report.

## The norm cap goes last, and the ordering is load-bearing

`feature_norm_bound` applies `x / max(1, ‖x‖ / R)` to every row: rows
inside the ball are untouched, rows outside are rescaled onto it. It
returns the bound it enforced, so `R` reaches the accountant from the
operation that made it true rather than being typed a second time at the
call site.

It runs **after** every fitted map, and that is not stylistic. A fitted
map rescales columns, so it does not preserve a bound applied before it:
capping to `R = 1` and then standardizing thirteen columns of differing
scale leaves the largest norm near 8, and the accountant would be handed
a bound of `√2` for data carrying about 8. Noise calibrated at a fifth of
what the mechanism needs, an ε reported, and nothing to crash.

The cap also refuses to tell you about the data. A row whose norm is
non-finite raises a warning that such a row *exists* — never how many, so
that one such row and a thousand are indistinguishable — and a warning
rather than a refusal.

## The split

Both loaders cut the file through the same seeded permutation:
`test_fraction=0.2` and `seed=0` by default, and no stratification. Two
modes at the same seed and fraction hold out the same rows because they
ran the same code, not because two copies of a permutation agree.

Two things the loader deliberately does not do:

- **No validation split.** It returns train and test. The notebooks cut a
  validation set out of the training rows themselves — 700,000 train and
  100,000 validation, held out once and never re-drawn — which is what
  makes every sampling rate in those runs a rate over 700,000.
- **No subsampling option.** "1M" is a property of the pinned file, not a
  parameter.

## One-hot Criteo is a separate function

`load_criteo_one_hot` returns a `SparseTabularSplit`: index and value
arrays rather than a dense matrix, plus `num_features` as a field on the
split, since a model is initialised with it.

It is a separate function rather than a `features="one_hot"` because it
changes the *return type*. A caller reading `load_criteo(features=...)`
would have no way to see at the call site that one value of that argument
hands back six arrays and a width instead of two matrices, and neither
would a type checker.

The width is why the encoding has to be sparse at all: 551,947 columns
over a million rows is 2.2 TB as a dense float32 matrix, and 250 MB as
pairs. The model never forms the row —
[`forward_sparse`](models.md#two-forwards-one-model) gathers 39 weights
and dots them against 39 values.

The encoding is built for one property. Each categorical column's block
is one slot wider than its training vocabulary, and that reserved slot
absorbs IDs unseen in training. A test row with an unfamiliar category
lands there instead of storing nothing, so **every** row in both splits
carries exactly 39 entries at distinct indices. The weight on a reserved
slot never appears in a training gradient and so never trains, which is
the whole price — and it buys an exact sparsity level. For a linear model
the per-example gradient's support *is* the row's support, so the
sparsity the sparse-DP literature assumes becomes a property of the
encoding rather than a claim about the data. That is what
[bias-reduced sparse SGD](../algorithms/bias-reduced-sparse-sgd.md) needs
— its rate depends on the sparsity of the gradients rather than on the
dimension, so it is the first method here whose point is lost on dense
data — and it is what the
[ℓ₁ projection](../transforms/l1-projection.md) transform needs to
denoise anything.

There is no `standardize` axis on this loader, and it is not offered
rather than offered and refused: centring a one-hot column gives every
row a non-zero entry in it, turning 39 stored values per row into
551,947 — the dense matrix this loader exists to avoid. Its `metadata`
carries an `encoding` key instead, since a `standardize` key would assert
an axis that does not exist rather than record which way one fell.

!!! warning

    **Sparse features do not buy a sparse per-example gradient buffer.**
    `jax.grad` with respect to the weight vector returns a dense row per
    example whatever the input representation, so the vmapped per-example
    clipping DP-SGD needs allocates batch-size × width floats — 565 MB at
    a batch of 256, 2.3 GB at 1024. The sparsity buys the representation
    and the forward pass. A sparse clip-and-scatter path is separate work
    and does not exist yet.

!!! warning

    **Status: evaluation pending.** No committed notebook runs on the
    one-hot load yet; the figures above are properties of the encoding,
    not results.

## What a dataset is here

`dimma.datasets.base` names no dataset. It holds the two shapes a loader
returns — `TabularSplit` and `SparseTabularSplit` — and the two functions
that place arrays on a device, which are the only device seam.

A new dataset is one module exposing a `load_*` function that handles
cache lookup, download, checksum verification and preprocessing, and
returns one of those two shapes. The conventions that make it fit: expose
loading choices as independent parameters rather than a mode name, fit
every statistic on the training split, record what you did in `metadata`,
and put any norm cap last.

Downloads are cached under `DIMMA_HOME` if set and the platform cache
directory otherwise; `download=False` turns a missing file into an error
naming the path it wanted, which is what the notebooks pass. Nothing is
re-exported from the package:

```python
from dimma.datasets.criteo import load_criteo, load_criteo_one_hot
```
