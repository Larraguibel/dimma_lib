# Data, models, and metrics

The stack around the training loop: the reference model and its losses,
and the evaluation metrics. How results are read — selection
threshold-free, reporting at a named cut, PR-AUC or a confusion matrix as
the headline — is on the [evaluation page](../evaluation.md).

The Criteo loader, `dimma.datasets.criteo.load_criteo`, is read in the
source:
[`src/dimma/datasets/criteo.py`](https://github.com/Larraguibel/dimma_lib/blob/main/src/dimma/datasets/criteo.py).
Its loading options are independent axes — columns, preprocessing,
standardization, the feature-norm bound — and what a load actually did is
recorded in the returned split's `metadata`.

## Models

::: dimma.models.logreg

::: dimma.models.losses

## Metrics

::: dimma.metrics.scoring

::: dimma.metrics.calibration

::: dimma.metrics.decomposition

::: dimma.metrics.operating_point

`dimma.metrics.ranking.pr_curve` — precision and recall at every cut, and
the area under them quoted as PR-AUC — is read in the source:
[`src/dimma/metrics/ranking.py`](https://github.com/Larraguibel/dimma_lib/blob/main/src/dimma/metrics/ranking.py).
