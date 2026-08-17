"""Post-processing that composes across algorithms.

Membership rule
---------------
A transform changes a quantity — parameters, a privatized gradient —
without being an algorithm, and it earns a module here only by
composing across several: one transform applies to more than one
algorithm, and an algorithm can stack more than one. A change only one
algorithm could ever make is that algorithm's, and lives in its
package.

A transform makes no privacy claim. Whether applying one is free
post-processing is a statement about the mechanism the run releases
from, and belongs where that mechanism's accounting is stated — not
here, and not in a property the transform carries around.

The layer applies; the geometry lives in `core`. Whether an
*algorithm's own step* consumes a transform by name from here, or
imports the geometry from `dimma.core` directly, is deliberately
undecided until a second call site exists; ADR-0014.

Imports
-------
No functions are re-exported. Import from the transform's module::

    from dimma.transforms.projection import l1_projected
"""

from dimma.transforms import projection

__all__ = [
    "projection",
]
