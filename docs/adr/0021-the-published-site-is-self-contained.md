# The published site is self-contained

The MkDocs site — `mkdocs/`, published at the repo's GitHub Pages URL —
never mentions ADRs, `CONTEXT.md`, or anything under `docs/`: not as links,
not by number, not by name. When a page needs the reasoning behind a
decision it re-explains that reasoning in its own voice, and the only repo
artifacts it points at are code under `src/`, `README.md`, and notebooks.

The alternative was to publish the decision record — a nav section over
`docs/adr/`, or design pages citing ADRs by number. We declined it because
the two surfaces serve different readers under different contracts. `docs/`
is agent-facing: terse, rule-shaped, and dense with internal
cross-references that assume the repo's context is loaded. The site
explains to a DP researcher who has none of that loaded. Citing across the
boundary would couple the site's prose to a numbering and register it does
not control, and would turn the ADRs into public API the moment a reader
bookmarks one.

## Consequences

Reasoning is woven into the page it justifies — there is no Design section
— so a decision worth publishing is written twice: once as a rule here,
once as explanation on the site. The two must agree in substance while
sharing no citation, and a change to a decision touches both.

Module docstrings cite ADRs freely, because code documentation is
agent-facing. The mkdocstrings reference must therefore curate around
them: render a member rather than its module where the module docstring
cites agent docs. `docs/writing-guide.md` carries the working rules,
including grepping the built site for "ADR" before publishing.
