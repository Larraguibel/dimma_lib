"""Evaluation metrics: threshold-free by construction.

Not a pipeline stage. `dimma.core` is organized by the seven stages of
the training loop and ADR-0001 requires a primitive that implements none
of them to justify itself separately; this package sits beside
`dimma.models` and `dimma.datasets` as something a run needs around the
loop rather than inside it, and nothing in `core` or `algorithms`
imports it.

Nothing here takes a threshold, and that is the organizing decision
rather than an omission. Click prediction never converts a probability
into a class — the number is multiplied into a bid or used to order a
slate — so a metric that needs an operating point is measuring a
decision the deployed system does not make. Worse, it lets each
hyperparameter configuration pick its own, which is not a comparison.
`accuracy`, `f1_score` and a confusion matrix are absent for that
reason, not because they were forgotten.

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

These are NumPy, in float64, on arrays already off the device. They are
reported rather than differentiated, so nothing is gained by holding
them in JAX, and equal-mass binning is a sort — see
`dimma.metrics._inputs` for why the width matters.

Nothing is re-exported here; import from the module that owns what you
need::

    from dimma.metrics.scoring import log_loss, normalized_entropy
    from dimma.metrics.calibration import reliability_curve
    from dimma.metrics.decomposition import log_loss_decomposition
"""

__all__: list[str] = []
