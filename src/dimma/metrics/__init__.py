"""Evaluation metrics, read around the training loop rather than inside it.

NumPy in float64, on arrays already off the device. Nothing that decides
a comparison takes a threshold, and ROC-AUC and `accuracy` are offered
nowhere: ADR-0016 draws that line, ADR-0001 says why a package
implementing no pipeline stage justifies itself separately.

Nothing is re-exported here; import from the module that owns what you
need::

    from dimma.metrics.scoring import log_loss, normalized_entropy
    from dimma.metrics.calibration import reliability_curve
    from dimma.metrics.decomposition import log_loss_decomposition
    from dimma.metrics.ranking import pr_curve
    from dimma.metrics.operating_point import best_f1_threshold, confusion_at
"""

__all__: list[str] = []
