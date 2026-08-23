# Notebooks

Grouped by what the run is for.

- **`tuning/`** — one algorithm run by itself, swept over its hyperparameters.
- **`comparisons/`** — two or more algorithms run head to head under a shared
  protocol.
- **`explorations/`** — one-off exploratory work.

## Numbering

A notebook's number is a global citation key. The ADRs, the tests, the issues
and the notebooks' own prose all cite a notebook by number and never by path, so
the number is the notebook's identity and the folder is only where it currently
sits. Numbers are unique across all three folders, are never reused, and are
never renamed. A new notebook takes the next free number whichever folder it
lands in — the sequence is one sequence, not one per folder.

## Explorations are disposable

Anything in `explorations/` may be deleted without ceremony. A result worth
keeping does not stay there: it graduates into `comparisons/` as a notebook run
under the shared protocol, or into `docs/` as prose.
