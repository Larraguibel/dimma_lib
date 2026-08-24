# Notebooks

Grouped by what the run is for.

- **`tuning/`** — one algorithm run by itself, swept over its hyperparameters.
- **`comparisons/`** — two or more algorithms run head to head under a shared
  protocol.
- **`explorations/`** — one-off exploratory work.

## Citing a notebook

A notebook's **name** is its citation key, and the citation carries the kind the
folder carries: a notebook under `tuning/` is cited as a *sweep*, one under
`comparisons/` as a *comparison* — *the DP-SGD sweep*, *the SpiderBoost-vs-DP-SGD
comparison*. Prose cites a notebook this way and never by path; documentation
outside `notebooks/` cites the repo path, which is `docs/writing-guide.md`
rule 6. Two notebooks about one algorithm are told apart by kind, which is why
the folder split has to stay.

Renaming breaks every citation, so don't. Nothing is allocated and nothing is
reserved, so a notebook on a branch is cited by name before it merges.

**Some files here still carry a leading number.** That is the older rule, under
which the number rather than the name was the identity. The numbers are being
dropped as each notebook is next touched; they are not part of any citation, and
a file that still has one is cited by its name like every other. Do not give a
new notebook a number.

## Explorations are disposable

Anything in `explorations/` may be deleted without ceremony. A result worth
keeping does not stay there: it graduates into `comparisons/` as a notebook run
under the shared protocol, or into `docs/` as prose.
