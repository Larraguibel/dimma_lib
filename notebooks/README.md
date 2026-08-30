# Notebooks

Grouped by what the run is for.

- **`tuning/`** — one algorithm run by itself, swept over its hyperparameters.
- **`comparisons/`** — two or more algorithms run head to head under a shared
  protocol.
- **`explorations/`** — one-off exploratory work.

## Citing a notebook

A notebook is cited by what it is, not by where it sits or by a number: *the
DP-SGD sweep*, *the SpiderBoost-vs-DP-SGD comparison*. The citation carries the
kind the folder carries — a notebook under `tuning/` is cited as a sweep, one
under `comparisons/` as a comparison — so two notebooks about one algorithm are
told apart by what they do with it. Prose never cites a path; documentation
outside `notebooks/` does, which is `docs/writing-guide.md` rule 6.

Nothing is allocated and nothing is reserved: a new notebook takes a name that
says what it runs on what data, and a notebook on a branch is cited by name
before it merges. Renaming breaks citations exactly the way renumbering used to,
so don't. ADR-0022 has the reasoning; ADR-0019 has the numbering scheme this
replaced.

## Explorations are disposable

Anything in `explorations/` may be deleted without ceremony. A result worth
keeping does not stay there: it graduates into `comparisons/` as a notebook run
under the shared protocol, or into `docs/` as prose. The one exception is a run
an ADR cites by path — the citation is the decision record, so that file is not
deletable without another one.
