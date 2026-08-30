# MkDocs writing guide

Rules for adding or editing pages under `mkdocs/`. Agents must read this
before touching any file in `mkdocs/`. `mkdocs/` is the published site;
nothing under `docs/` is ever published.

---

## 1. Know the reader

The reader works in DP — often on the theory or statistics side. Do not
define baseline DP vocabulary (ε, δ, sensitivity, the Gaussian mechanism).
Do explain the practice of DP optimization, even when it feels basic
within applied work: the distinctions practice forces (batches versus
per-sample gradients, clipping as an enforced operation, what each
hyperparameter buys) are exactly what a theory-side reader has not
needed before.

## 2. Define every algorithm-specific symbol on first use

Any letter or identifier naming an algorithm parameter, mathematical
quantity, or code variable is defined the first time it appears on a page,
with a short parenthetical:

> `q` (the anchor interval — an anchor step fires every `q` steps)

Each page must be interpretable on its own. Use the repo's canonical
vocabulary: `CONTEXT.md` fixes the terms, and packages carry a
paper-symbol-to-code table in their `__init__` docstring; prose reuses
those names and never invents a third. Consult `CONTEXT.md` while
writing — but never cite it on a page (rule 3). One phrase is enough —
a foothold, not a duplicate of the paper.

## 3. The site never mentions agent docs

ADRs, `CONTEXT.md`, and anything under `docs/` do not appear on a published
page — not as links, not by number, not by name. When a page needs the
reasoning behind a decision, it re-explains that reasoning in its own
voice. The only repo artifacts a page may point at are code under `src/`,
`README.md`, and notebooks (by path, per rule 6).

## 4. Weave the reasoning in

The "why it's built this way" lives in the section describing the thing it
justifies. There is no separate design section, and no page collects
decisions for their own sake.

## 5. One concern per section; one page per algorithm

Each algorithm gets one page. Each `##` section on it answers one question.
Split an algorithm into a folder (`algorithms/<name>/`) only when a section
outgrows the page — a genuinely separate concern with its own audience.

## 6. Cite the evaluating notebook

Every algorithm page names the notebook that evaluates it, by repo path
(e.g. `notebooks/comparisons/spiderboost-vs-dp-sgd-on-criteo.ipynb`).
Notebooks are never rendered into the site. If the evaluation does not
exist yet:

    !!! warning
        **Status: evaluation pending.** No notebook evaluates this
        algorithm yet.

## 7. Plots are exported, with provenance

A plot on a page is a static image committed under `mkdocs/assets/`,
exported from a notebook. Its caption names the producing notebook.

## 8. Use admonitions for status

Signal the epistemic status of a claim with admonitions (`warning` for
open questions, `note` for background). Do not bury open questions in
prose where they look like settled facts.

## 9. Examples use plain pytrees

dimma models are `init_params` plus a `forward` over a plain pytree. Code
examples never import Flax or any model framework.

## 10. Prose cites live code

Any code identifier in prose must exist under `src/` at the time of
writing. Grep before publishing; a dangling identifier is a broken link.

## 11. The reference is curated

A page under `reference/` exists only by deliberate addition, for a
people-facing module. Never auto-walk the package. Docstrings render
onto the site as written, and many module docstrings cite ADRs — which
rule 3 bans from the site — so curate around them: render a member
rather than its module where the module docstring cites agent docs, and
grep the built `site/` for "ADR" before publishing.

## 12. Language and links

All pages are in English. Links between pages are relative. Never
hardcode the published site URL.

---

## When adding a new algorithm

1. Create `mkdocs/algorithms/<name>.md`: what it does in prose, its entry
   point under `dimma.algorithms.<name>`, the reasoning woven in (rule 4),
   the evaluating notebook or the pending admonition (rule 6).
2. Add a row to the table in `mkdocs/algorithms/index.md`
   (columns: Algorithm / Paper).
3. Add the page to `nav` in `mkdocs.yml` under `Algorithms`.
4. Update `mkdocs/index.md` → "Structure" to mention it.
5. Only if its people-facing surface warrants: add a curated
   `reference/` page (rule 11).
