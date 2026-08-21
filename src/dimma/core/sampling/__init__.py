"""Stage 1 - batch generation.

A package rather than a module because each sampling mechanism gets its
own file: two samplers can differ in whether the standard accounting
applies to them, and that belongs in the import line.

``dyadic``
    A fixed-size draw whose size is itself drawn, on a ladder of
    powers of two. Nothing is padded; no standard
    subsampled-Gaussian accounting is stated against it.
``poisson``
    The standard mechanism. Raises on an oversize draw.
``poisson_truncated``
    Truncates oversize draws. Modified mechanism; the standard
    accounting does not apply to it.
``shuffled``
    Not a mechanism at all. Ordinary epoch shuffling, for the
    non-private baselines; no amplification and nothing to account.

Samplers state what they sample; they compute no privacy budget.
"""

from dimma.core.sampling import dyadic, poisson, poisson_truncated, shuffled

__all__ = ["dyadic", "poisson", "poisson_truncated", "shuffled"]
