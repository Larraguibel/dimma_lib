"""Evaluation metrics: selection is threshold-free by construction.

Not a pipeline stage. `dimma.core` is organized by the seven stages of
the training loop and ADR-0001 requires a primitive that implements none
of them to justify itself separately; this package sits beside
`dimma.models` and `dimma.datasets` as something a run needs around the
loop rather than inside it, and nothing in `core` or `algorithms`
imports it.

Nothing that decides a comparison takes a threshold, and that is the
organizing decision rather than an omission. Click prediction never
converts a probability into a class — the number is multiplied into a
bid or used to order a slate — and a metric that needs an operating
point lets each hyperparameter configuration pick its own, which is not
a comparison. So models are selected on the strictly proper scores and
read on the ranking and calibration modules beside them. Reporting is
the separate job: a headline confusion matrix is taken at one named cut,
and `dimma.metrics.operating_point` is the only module here that takes
one. ROC-AUC and `accuracy` are absent and stay absent, at every cut and
in every module — ADR-0016 narrows the rule to this shape and gives the
reasoning for both halves.

What is here, and when to reach for it:

- `dimma.metrics.scoring` — `log_loss`, `brier_score`,
  `normalized_entropy`. Strictly proper: minimized only by the true
  probability. Start model selection here.
- `dimma.metrics.calibration` — `reliability_curve`,
  `expected_calibration_error`, `calibration_ratio`. Whether the stated
  probabilities are the rates that occurred, which no ranking score
  sees.
- `dimma.metrics.decomposition` — `brier_decomposition`,
  `log_loss_decomposition`. The identity that makes the previous two
  one thing: a proper score is calibration minus discrimination plus a
  constant, so the choice between them was never a choice.
- `dimma.metrics.ranking` — `pr_curve`. Precision and recall at every
  cut, and the area under them that is quoted as PR-AUC. Reach for it to
  read the order a model put the records in, against a floor that is the
  base rate rather than zero.
- `dimma.metrics.operating_point` — `best_f1_threshold`, `confusion_at`.
  A cut, and the four counts at it. Reach for it to report a comparison
  in records rather than in rates, at a threshold chosen under a named
  rule and frozen before the split it is reported on.

These are NumPy, in float64, on arrays already off the device. They are
reported rather than differentiated, so nothing is gained by holding
them in JAX, and equal-mass binning is a sort — see
`dimma.metrics._inputs` for why the width matters.

Nothing is re-exported here; import from the module that owns what you
need::

    from dimma.metrics.scoring import log_loss, normalized_entropy
    from dimma.metrics.calibration import reliability_curve
    from dimma.metrics.decomposition import log_loss_decomposition
    from dimma.metrics.ranking import pr_curve
    from dimma.metrics.operating_point import best_f1_threshold, confusion_at
"""

__all__: list[str] = []
