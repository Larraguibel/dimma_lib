"""Reference models.

Convenience only: no algorithm imports this package. It exists so that a
method can be run without writing a model first, and so that two
algorithms compared against each other are compared on the same one.

A model here is a pair of plain functions over a plain JAX pytree —
``init_params`` and a ``forward`` that maps one feature vector to one
scalar. `dimma.models.losses` supplies the matching per-sample loss.

Nothing is re-exported here; import from the module that owns what you
need::

    from dimma.models.logreg import init_params, forward
    from dimma.models.losses import per_sample_bce_loss
"""

__all__: list[str] = []
