"""Calibrating noise to a budget, and the epsilon a run spent.

Membership rule
---------------
`accounting` is where privacy *claims* live. `core` may describe what a
sampler samples or which assumption a primitive satisfies; the moment a
number is called an epsilon, a scale is calibrated to a budget, or a
transformation is called free, it belongs here.

Three kinds of module, and the split is not stylistic:

`sampling` and its neighbours wrap Google's `dp-accounting` for the
standard mechanisms, where the analysis is settled and shared. An
algorithm gets its own module here - `spiderboost`, `bias_reduced_sgd`,
and others as they land - only when its mechanism falls outside those
assumptions or is bounded too loosely by them. Such a module travels
with its algorithm and is not general-purpose. `bias_reduced_sgd` is
the one whose accountant is not a composition at all but a filter,
because the schedule it prices is chosen by the run: ADR-0018.

The third kind supplies a *premise* an accountant takes rather than an
epsilon: `lipschitz` produces the constants Private SpiderBoost assumes
about the loss, from an enforced bound rather than from the data. Here
and not in `models` per ADR-0003; ADR-0012 records the rest.

An accountant is only as good as the match between its assumptions and
the code that ran. Every function here states the mechanism it assumes;
if the training loop departs from it, the number is not a guarantee.

Imports
-------
No functions are re-exported. Import from the module, so the import
line says which analysis is being invoked:

    from dimma.accounting.sampling import poisson_gaussian_epsilon
"""

from dimma.accounting import (
    bias_reduced_sgd,
    lipschitz,
    sampling,
    spiderboost,
)

__all__ = ["bias_reduced_sgd", "lipschitz", "sampling", "spiderboost"]
