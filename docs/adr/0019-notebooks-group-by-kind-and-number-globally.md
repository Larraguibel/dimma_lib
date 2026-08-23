# Notebooks are grouped by kind; their numbers stay global

`notebooks/` splits into three folders named for what a run is for — `tuning/`
for one algorithm swept over its hyperparameters, `comparisons/` for two or more
run head to head under a shared protocol, `explorations/` for one-off work. The
numbers do not move with the files. They stay global across all three folders,
unique, and permanent.

The split and the numbering answer different questions, and the mistake would be
to let one answer both. A flat folder said nothing about whether a notebook
tuned one algorithm or arbitrated between two, which is the first thing a reader
needs and the thing that decides whether a result is comparable to another. That
is an organization problem, and a folder solves it.

Identity is the other question, and folders are the wrong instrument for it.
Every reference to a notebook in this repository is by number — ADR-0016 pins
metric parity to notebook 02 and contrasts it with notebook 01, a test class is
called `TestNotebook02Parity`, issues name the notebook they ask for, and the
notebooks cross-reference each other in prose. None of those references carry a
path, and none would survive a renumber. So a number is a citation key:
it identifies one run and one set of conclusions for as long as anything cites
them, and a key that can be reassigned is not a key.

## Consequences

Moving a notebook between folders is allowed and costs nothing, because no
reference names a folder. Renumbering and renaming are not allowed. A notebook
whose kind was misjudged gets moved; a notebook that is superseded keeps its
number and the superseding one takes a new one.

Numbers are allocated from one sequence, not one per folder. The next notebook
takes the next free number whichever folder it lands in, so a number never
implies a kind and reading the sequence never implies reading a folder.

A number is reserved by whatever claimed it, including work not on `main`. 03 is
absent from `main` and reserved anyway: it lives on an unmerged branch, where it
closed issue #9, and it is already cited from there. A gap in the sequence is
expected and is not an invitation to fill it.

`explorations/` is disposable by construction — its contents may be deleted
without a decision record — and that is the reason work there does not earn a
citation until it graduates into `comparisons/` or into `docs/`.

No path-based reference to a notebook exists to update, and this decision is why
none should be introduced. Cite the number.
