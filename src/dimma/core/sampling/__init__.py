"""Stage 1 - batch generation.

A package rather than a module because each sampling mechanism gets its
own file: two samplers can differ in whether the standard accounting
applies to them, and that belongs in the import line.

``poisson``
    The standard mechanism. Raises on an oversize draw.
``poisson_truncated``
    Truncates oversize draws. Modified mechanism, lower bound rather
    than a guarantee.

Samplers state what they sample; they compute no privacy budget.
"""

from dimma.core.sampling import poisson, poisson_truncated

__all__ = ["poisson", "poisson_truncated"]
