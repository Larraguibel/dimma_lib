# A notebook's name is its citation key

Notebook filenames carry no number. `notebooks/tuning/dp-sgd-on-criteo.ipynb`,
not `01-dp-sgd-on-criteo.ipynb`. Prose cites a notebook by what it is — *the
DP-SGD sweep*, *the SpiderBoost-vs-DP-SGD comparison* — and the citation carries
the kind the folder carries: a notebook under `tuning/` is cited as a sweep, one
under `comparisons/` as a comparison. Documentation outside `notebooks/` cites
the repo path, which is `docs/writing-guide.md` rule 6 and is unchanged.

This reverses the numbering half of ADR-0019. The folder split that ADR made
stands; only the claim that identity needs a number goes.

ADR-0019 argued that a citation key must survive a move, and a number does. That
much was right, and it is why nothing here reintroduces paths into prose. What
it did not weigh is the other half of the trade: a number is stable *and*
opaque. It identifies without describing, so every reader of "notebook 05" was
carrying a nine-entry lookup table or going to fetch one, and the notebooks lean
on cross-references heavily enough — one file cited another twenty-four times —
that the cost was paid on nearly every page. The names beside the numbers were
already descriptive. The number was a second identity layered over a working
first one.

The sequence also turned out to cost what ADR-0019 said renumbering would. A
number is allocated before the work lands, so it can be claimed from a branch,
and one notebook spent a paragraph of its own opening cell explaining why it was
`05` and not `03` — a defence of its position in a sequence, in the file a reader
opens to learn about SpiderBoost. A key that has to justify itself in prose is
not doing the silent work a key is for.

## The stability argument survives, unchanged

A name is exactly as stable as a number for as long as nobody renames it, and
the rule is the same rule: don't. A renamed notebook is a renumbered notebook,
and it breaks every citation the same way. What changes is only that the thing
held fixed now says what it is.

Two notebooks about one algorithm are distinguished by kind, not by digit — *the
bias-reduced-SGD sweep* tunes it, *the bias-reduced-SGD comparison* runs it
against DP-SGD — so the citation vocabulary needs the folder split to work, and
that is the second reason the split stays.

## Consequences

Nothing is allocated and nothing is reserved. There is no sequence, so there are
no gaps in it, and a notebook on a branch cites the notebook it inherits from by
name before either has merged. Work in `explorations/` earns no prose citation:
the citation names a kind, and `explorations/` is not one. Documentation outside
`notebooks/` cites it by repo path, which is rule 6, and an ADR that does so has
graduated the finding in ADR-0019's sense — the prose is the record, and the run
behind it is no longer deletable without one. ADR-0009 cites the feature-norm
sweep that way.

A notebook that is superseded keeps its name and the superseding one takes its
own; where two would collide, the newer one says in its name what makes it
different — the dataset mode, or the axis it moves — which is what the names
already did.

Test names follow. A parity class is named for the notebook it pins, so
`TestNotebook02Parity` is `TestDpSgdVsSgdBaselineParity`, and a test that cites a
notebook in a docstring cites it the way prose does.

Moving a notebook between folders now costs something it did not cost under
ADR-0019: the citation names the kind, so a file that moves from `tuning/` to
`comparisons/` is cited differently afterwards and the citations have to move
with it. That is the price of a citation that describes, it is paid only when a
notebook's kind was misjudged, and a misjudged kind was already worth a
correction.
