"""Reference models.

Convenience only: no algorithm imports this package. It exists so that a
method can be run without writing a model first, and so that two
algorithms compared against each other are compared on the same one.

A model here is a pair of plain functions over a plain JAX pytree —
``init_params`` and a ``forward`` that maps one feature vector to one
scalar. It becomes stage 2 of the pipeline by being called inside the
per-sample loss the caller hands to an algorithm; `dimma.models.losses`
supplies that loss for the models shipped here.

Nothing is re-exported here; import from the module that owns what you
need::

    from dimma.models.logreg import init_params, forward
    from dimma.models.losses import per_sample_bce_loss
"""

__all__: list[str] = []
