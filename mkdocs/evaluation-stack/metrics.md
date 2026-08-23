# Metrics

`dimma.metrics` sits beside the training loop, never inside it — a
private loop returns parameters and nothing else, because evaluating a
model on the training data is another access to it, costing budget the
algorithm does not account for.

What the package contains is not the usual set, and the departures are
deliberate. This page gives the reasoning for each one.
[Evaluation on Criteo](../evaluation.md) reports what the numbers came
out to be; this page is about why those are the numbers being read.

## Nothing that decides a comparison takes a threshold

Click prediction never converts a probability into a class — the number
is multiplied into a bid, or used to order a slate. A metric that needs
an operating point lets each hyperparameter configuration pick its own,
and a set of numbers each computed at its own cut is not a comparison.

So the package splits along that line. Models are **selected** on scores
that take no threshold. They are **reported** with a confusion matrix at
one named cut, and `dimma.metrics.operating_point` is the only module in
the package that accepts one. The split is enforced by tests over the
package surface rather than left to convention: no other module may offer
a name that needs a cut.

**ROC-AUC and accuracy are offered nowhere**, and not provisionally.
Criteo's base rate — the fraction of records that are positive — sits
near 25%, so the false-positive-rate axis is dominated by the majority
class: ROC-AUC flatters every model and compresses exactly the
differences a comparison exists to show. Accuracy is worse than
uninformative at that base rate, since predicting nothing scores 75%.
Both names are barred by test, in every module including the one that
owns a cut.

## Selection runs on strictly proper scores

`dimma.metrics.scoring` offers `log_loss` and `brier_score`. Both are
**strictly proper**: each is minimized, uniquely, by reporting the true
conditional probability.

That property is the whole argument, and it is what ranking scores lack.
Every ranking score is invariant to any increasing transform of the
predictions, so a model that orders records perfectly while stating
probabilities three times too large scores exactly as well as the
calibrated one. A private run's most likely failure is precisely that —
order preserved, probabilities shifted — so selecting on a ranking score
would select against the thing most likely to have broken.

`log_loss` is the default for choosing between runs, because the model
trains on the same quantity: selection and training then agree on what
"better" means, with no proxy in between. `brier_score` is reported
beside it because it is bounded in `[0, 1]`, so a single confident
mistake moves it by at most `1/n` for `n` records — which makes it the
more stable of the two across a sweep, and the less sensitive to the tail
behaviour a private run is most likely to damage. Report both.

Two consequences worth knowing before quoting a number:

- **These take probabilities, never logits.** The round trip through a
  probability loses the tail, so when the model is confident,
  `dimma.models.losses.batch_bce_loss` — which reads the logit
  directly — is the one to quote.
- **`log_loss` clips probabilities away from 0 and 1.** That clip is a
  floor on how bad a single record is allowed to look, not a fact about
  the model.

### `normalized_entropy`, and why raw log loss does not travel

`normalized_entropy` divides log loss by the binary entropy of the base
rate. A value of 1.0 is the constant base-rate predictor; the distance
below 1.0 is the share of the available uncertainty the model removed.

It is there because raw log loss falls as the base rate approaches 0 or 1
for reasons that have nothing to do with the model — predicting a rare
event is simply cheaper to be right about on average. A log loss of 0.15
is a different achievement at a 3% base rate than at 25%, and only the
normalized form compares across datasets. This is the metric reported in
He et al., *Practical Lessons from Predicting Clicks on Ads at Facebook*
(ADKDD 2014).

## Ranking is read, but never decided on

`dimma.metrics.ranking.pr_curve` returns precision and recall at every
cut together with the average precision — the number quoted as PR-AUC.
Every cut is taken, so nothing here fixes an operating point.

What it reads is the *order* the model put the records in, and only that.
Two properties follow, and both are easy to get wrong:

- **Its floor is the base rate, not zero.** An ordering carrying no
  information scores about the fraction of records that are positive. On
  Criteo that floor is 0.252, so a PR-AUC of 0.45 is read against 0.252
  and not against 0. The floor also moves between datasets for reasons
  having nothing to do with the model.
- **It is blind to everything a proper score sees.** Read it alongside
  `dimma.metrics.scoring`, never instead of it.

Ties are broken by an unstable sort, shared deliberately between
`ranking` and `operating_point` so the two cannot drift apart on the one
thing they have in common.

## Calibration is measured directly, because DP breaks it first

The distinction between ordering records correctly and stating the right
probabilities has a name — discrimination against calibration — and it is
the one that matters for a private run.

The mechanism is specific. Clipping and noise both act on the gradient,
but a *clipped* gradient is biased toward the majority class, so the
failure a private model reaches first is usually a preserved order with
shifted probabilities. That is invisible to every ranking metric. That
per-example clipping rather than the noise is the documented cause of
miscalibration under DP is the finding of Zhang et al., *A Closer Look at
the Calibration of Differentially Private Learners*
([arXiv:2210.08248](https://arxiv.org/abs/2210.08248)) — and it is why
the tuning sweeps in `notebooks/` are run over the clipping norm rather
than around it.

`dimma.metrics.calibration` reports it in three registers, because the
cheap one can be right while the model is wrong everywhere:

| Name | What it gives |
|---|---|
| `calibration_ratio` | observed positives over predicted positives; 1.0 is calibrated |
| `reliability_curve` | the full binned picture, with a `gap` of predicted minus observed per bin |
| `expected_calibration_error` | one occupancy-weighted number, for a sweep |

`calibration_ratio` is the aggregate the ads literature reports, and it
carries a direct reading: 0.9 means the model claims about 11% more
clicks than happened, and a bid built on it overpays by roughly that
much. It can also sit at exactly 1.0 over a curve that is wrong in every
bin, which is why the other two exist.

!!! warning

    **ECE is a biased estimate, and the bias runs one way.** Each bin's
    observed rate carries its own sampling error, and the absolute value
    in the mean cannot cancel it — so a perfectly calibrated model scores
    above zero, and scores *worse* the more bins it is given. Compare
    ECEs only at equal `n_bins`, equal `strategy`, and comparable sample
    size, and treat a small difference between two of them as noise
    rather than as a result.

### Bins are a smoothing parameter, not a detail

Every binned quantity here takes `n_bins` (15 throughout the notebooks)
and `strategy`, which defaults to `"equal_mass"` — equal counts per bin
rather than equal width.

Equal-width is the wrong default for this data. Predicted click
probabilities pile up around the base rate and thin out fast on both
sides, so equal-width bins spend most of their bins on stretches holding
almost nothing and drop most of the data into two or three. Equal-mass
bins hold the same count each, which also estimates every bin's observed
rate to about the same precision — and that rate is the quantity being
compared. `"equal_width"` remains available, and is the choice when
matching a published figure.

The count itself is a real trade: too few bins and a model badly
calibrated *within* a bin looks calibrated; too many and each observed
rate is a mean over so few records that its own sampling noise dominates.
Empty bins are dropped, so a result may hold fewer bins than were asked
for — report the count alongside anything computed from these.

## The decomposition, which dissolves the choice

Choosing between a ranking score and a calibration score looks like
choosing which half of the model to care about. It is not, because a
proper score already contains both, and the containment is an identity
rather than an analogy. `dimma.metrics.decomposition` makes it explicit:

```
score = calibration - resolution + uncertainty + residual
```

`brier_decomposition` gives Murphy's three terms plus the gap left by
binning; `log_loss_decomposition` is the information-theoretic reading of
the same split. Murphy's *reliability* is renamed `calibration` so that
both decompositions speak one vocabulary.

It earns its place in a private run because **the two terms move for
unrelated reasons**. Noise added to the gradient degrades what the model
can tell apart, and resolution falls. Clipping biases the update toward
the majority class, shifting the probabilities without necessarily
disturbing their order, and calibration rises. A single number — private
log loss against non-private log loss — reports the sum of the two and
cannot say which happened; a ranking score reports a proxy for one and is
blind to the other by construction. The decomposition says which, and
that is a claim about the mechanism rather than about the score.

Under equal-mass binning the two terms land exactly on the ranking
score's blind spot. The bins are cut at quantiles, so any strictly
increasing transform of the predictions leaves every bin holding the same
records and every observed rate untouched: resolution is then rank-based,
and is precisely the part a ranking score can see, with calibration
carrying everything a ranking score is invariant to.

Two terms are read differently from the other two:

- `uncertainty` is fixed by the evaluation split, identical for every
  model scored on it, and the reason two decompositions from different
  splits do not compare.
- `residual` is a diagnostic on the binning. Near zero means the bins are
  fine enough; **negative** means real resolution the partition threw
  away, and asks for more bins.

## Reporting: one cut, named, and frozen

A confusion matrix is a quantity at a threshold, and a threshold is a
choice. Made carelessly it becomes an extra difference between the arms
of a comparison.

`dimma.metrics.operating_point` supplies the two pieces:
`best_f1_threshold` searches every observed score as a candidate — the
exact maximiser, not a grid — and `confusion_at` counts at a given cut.
The rule around them is what makes the number mean something: the
threshold is chosen by **one rule applied identically to both models, per
model, on the validation split, and frozen before the test split is
read**. Choosing it on test would make every number after it a fitted
quantity; forcing a single shared threshold on both models would penalise
whichever is differently scaled, which is not the same as being worse.
The rule is shared; the number it returns is each model's own.

The obvious cut is the wrong one. At a 25% base rate a calibrated model
rarely emits a probability above 0.5 at all, so that threshold reports
the base rate rather than the classifier. Where a notebook needs a cut
fixed before any run rather than one fitted per model, it uses the
training-split base rate instead — a training-split statistic, and so not
a further read of the test labels. Both rules appear across `notebooks/`;
what does not appear is a cut chosen after the numbers were seen.

!!! note

    **The library cannot police this.** `best_f1_threshold` returns a cut
    and knows nothing about where the probabilities came from, so a
    threshold fitted on the split it is then reported on is a defect no
    test here can see. That obligation sits with whoever reports the
    number.

## How far a difference is trusted

Selected configurations are re-run at three seeds, varying the sampling
stream and — in a private arm — the noise, with initialization held
fixed. **Three seeds is a spread, reported as min/max. It is not a
confidence interval and not an error bar.**

It is enough to do the one job it is asked to do: tell whether a gap
survives a re-run. On Criteo it repeatedly does not — a PR-AUC gap of
0.0006 whose *sign flips* under the repeat is not a ranking difference,
and a tuning step worth `1e-4` on an axis whose seed spread is `1e-4` is
a plateau rather than a peak. Numbers quoted without that spread beside
them are quoted at the wrong precision.

## Input discipline

- **Everything is coerced to float64**, though the model trains in
  float32. These are sums over hundreds of thousands of records, and a
  float32 accumulator loses the last digits of exactly the quantity the
  exercise is about: a calibration gap is a small difference between two
  numbers near the base rate, read at the fourth decimal.
- **Degenerate input raises rather than propagates.** A diverged run
  arrives as `nan`, a split can contain no positives, a label column can
  be constant — what those mean is decided at the call site, not silently
  folded into a reported number.
- **The package does not import JAX**, and a test pins that. Importing it
  would invite someone to make a metric differentiable, which is how a
  threshold-free score turns into a training objective nobody chose.

Nothing is re-exported from `dimma.metrics`; import from the module that
owns what you need:

```python
from dimma.metrics.scoring import log_loss, normalized_entropy
from dimma.metrics.calibration import expected_calibration_error
from dimma.metrics.decomposition import log_loss_decomposition
from dimma.metrics.ranking import pr_curve
from dimma.metrics.operating_point import best_f1_threshold, confusion_at
```
