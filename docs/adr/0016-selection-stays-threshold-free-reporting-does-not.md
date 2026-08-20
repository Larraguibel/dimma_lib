# Selection stays threshold-free; reporting at a chosen cut does not

`dimma.metrics` was built so that nothing in it took a threshold. That rule is
narrowed, not repealed. It now reads: nothing that lets each configuration pick
its own cut may decide a comparison, so model selection stays on the strictly
proper scores. Two modules join the package on the other side of that line —
`dimma.metrics.ranking`, which reads the order a model put the records in, and
`dimma.metrics.operating_point`, which reports at a cut the caller names.

The old guard was over-broad in one place and right in the other.

A PR curve is threshold-free. It takes precision at every cut and integrates
against recall, so it answers a ranking question and no configuration chooses
anything. `precision_recall_curve` and `average_precision` were barred by name;
they were casualties of the rule rather than targets of it, and `pr_curve` is
admitted on that ground alone.

Reporting is a different job from selection. The rule was about selection and
comparison — a metric that needs an operating point lets each configuration
pick its own, which is not a comparison — and that half survives intact. What
is admitted is the report: every comparison needs a headline confusion matrix
at one named rule (validation F1, chosen per model, frozen before the test
split is read), and both of dimma's algorithms reach for it, which is the
admission test CONTEXT.md sets. Keeping it out did not keep it from being
computed; it kept it from being tested. Notebook 01 and notebook 02 each
hand-rolled a different implementation of the number that decides the
comparison, and nothing in the suite would have noticed if one were wrong.

Two names stay barred, and not provisionally. ROC-AUC is not to be reported
anywhere in the package or in the notebooks: Criteo click prediction has a base
rate near 25%, and the false-positive-rate axis is dominated by the majority
class, so every model is flattered and the differences a comparison exists to
show are compressed. PR-AUC or a confusion matrix is the headline. `accuracy`
goes with it, degenerate at that base rate — predicting nothing scores 75%.

## Consequences

`tests/metrics/test_package_surface.py` is narrowed rather than deleted, and
that narrowing is where this decision is enforced. `roc_auc` and `accuracy`
stay barred in every metric module, the two new ones included. The names that
need a cut stay absent from every module that does not own one — and `ranking`
is such a module, since a curve over all cuts fixes none of them. A later
convenience import still fails a test rather than quietly putting a threshold
back into a comparison built not to need one.

ADR-0004 is untouched. Nothing is re-exported from `dimma.metrics`, so a caller
writes `from dimma.metrics.operating_point import confusion_at` and the import
line keeps saying which question the number answers — here, one asked at a cut.

Which split a threshold was chosen on is the caller's to state. `best_f1_threshold`
returns a cut and knows nothing about where the probabilities came from, so a
threshold fitted on the split it is then reported on is a defect the library
cannot see. That obligation sits with whoever reports the number, the same
place ADR-0012 leaves the provenance of `R`.

Parity is anchored to notebook 02. Its `pr_curve`, `best_f1_threshold` and
`confusion_at` are the implementations the promoted functions reproduce, and
the pinned tests hold their numbers. Notebook 01's `average_precision` spells
the divisor differently and computes the arithmetically identical value, so
promoting notebook 02's behaviour loses nothing notebook 01 had.

One divergence is deliberate. On labels holding no positive the notebooks
divide by zero, propagate `nan` and emit a runtime warning; `pr_curve` and
`best_f1_threshold` raise instead, following the refuse-degenerate-input
convention of `dimma.metrics._inputs` and `base_rate_entropy`. That input is
degenerate, no notebook run reaches it, and parity holds everywhere they do.

Plotting stays in the notebooks. The confusion matrix is four counts here and a
figure there; a palette is presentation, and the metrics package does not grow
a plotting dependency to hold one.
