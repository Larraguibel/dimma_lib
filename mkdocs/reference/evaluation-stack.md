# Data, models, and metrics

The stack around the training loop: the reference model and its losses,
and the evaluation metrics. The choices behind them are covered in prose
on [datasets](../evaluation-stack/datasets.md),
[models](../evaluation-stack/models.md) and
[metrics](../evaluation-stack/metrics.md); what the executed comparisons
found is on the [evaluation page](../evaluation.md).

The Criteo loader, `dimma.datasets.criteo.load_criteo`, is read in the
source:
[`src/dimma/datasets/criteo.py`](https://github.com/Larraguibel/dimma_lib/blob/main/src/dimma/datasets/criteo.py).
Its loading options are independent axes — columns, preprocessing,
standardization, the feature-norm bound — and what a load actually did is
recorded in the returned split's `metadata`.

## Models

::: dimma.models.logreg
    options:
      members:
        - init_params
        - forward

`dimma.models.logreg.forward_sparse` — the same logit from index/value
pairs, for a width at which the dense row is the difficulty — is read in
the source:
[`src/dimma/models/logreg.py`](https://github.com/Larraguibel/dimma_lib/blob/main/src/dimma/models/logreg.py).
Why the sparse representation exists is on
[datasets](../evaluation-stack/datasets.md#one-hot-criteo-is-a-separate-function).

::: dimma.models.losses

## Metrics

::: dimma.metrics.scoring

::: dimma.metrics.calibration

::: dimma.metrics.decomposition

::: dimma.metrics.operating_point

`dimma.metrics.ranking.pr_curve` — precision and recall at every cut, and
the area under them quoted as PR-AUC — is read in the source:
[`src/dimma/metrics/ranking.py`](https://github.com/Larraguibel/dimma_lib/blob/main/src/dimma/metrics/ranking.py).
