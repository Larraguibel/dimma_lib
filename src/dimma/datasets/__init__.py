"""Dataset loaders.

Convenience only: the algorithms never import this package. It exists so
that a method can be run against real data without assembling a loader
first, and so that two algorithms compared against each other are
compared on byte-identical inputs.

One module per dataset, each exposing a ``load_*`` function that handles
cache lookup, download, checksum verification, and preprocessing. What a
loader did to the data is recorded in the split's ``metadata`` rather than
left implicit in the mode name.

Nothing is re-exported here; import from the module that owns what you
need::

    from dimma.datasets.criteo import load_criteo
"""

__all__: list[str] = []
