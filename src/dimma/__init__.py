"""dimma - a JAX library for differentially private optimization.

The pipeline is a fixed sequence of stages, each implemented once as an
architecture-agnostic primitive over JAX pytrees. An algorithm is a
choice at each stage rather than a from-scratch training loop.

See `dimma.core` for the stages and the rule governing them. Nothing is
re-exported here; import from the module that owns what you need.
"""

__version__ = "0.1.0"

__all__: list[str] = []
