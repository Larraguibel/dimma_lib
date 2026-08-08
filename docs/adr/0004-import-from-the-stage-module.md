# Nothing is re-exported from a package `__init__`

A caller writes `from dimma.core.clipping import per_sample_clip`, never
`from dimma.core import per_sample_clip`. Packages export their submodules and
nothing else, and the same holds for the two samplers, which stay in separate
modules rather than being flattened into `dimma.core.sampling`.

This is deliberately less convenient than the usual flat re-export. The import
line is the one place a reader always sees, and making it name the stage — or
name *which* sampling mechanism was chosen — puts the information where it
cannot be skipped. Flattening the samplers in particular would hide the only
thing that distinguishes them, which is whether the standard accounting
applies.

## Consequences

Nothing enforces this at runtime, so `tests/core/test_package_surface.py`
pins it: a convenience re-export added later fails a test rather than quietly
widening a surface the README commits to keeping stable.
