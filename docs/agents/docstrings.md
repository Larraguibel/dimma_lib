# Docstrings

How Python docstrings are written in this repo. **NumPy/numpydoc style** — the
scientific-Python standard, and what the libraries around us already use. Never
mix in Google-style (`Args:`, `Returns:`) or bare reST field lists (`:param x:`).

## Two tiers of depth

Depth tracks audience, not line count.

- **People-facing** — anything on the public API, the functions and classes a
  caller imports and reads: a **full** numpydoc docstring.
- **Private helpers** — leading underscore, or reachable only from inside its
  own module: **one imperative line**, or nothing when the name already says it.

Never stamp a full `Parameters`/`Returns` template onto a trivial helper. The
template is for readers who can't see the body; a helper's readers can.

## Every docstring

- The first line is **one imperative sentence ending in a period** — `Return the
  clipped gradient.`, not `Returns...` and not `This function...`. It stands
  alone on its line, and a blank line separates it from any body.
- Document the **contract**: inputs, outputs, errors raised, side effects,
  assumptions the caller must meet. Not the implementation — that's the code.
- **Rationale lives in `docs/adr/`.** Reference an ADR by number; don't restate
  its argument. Docstrings stay file-local: they carry what's relevant to *that*
  file only.

## Full docstrings

Use numpydoc sections, in this order, skipping any that don't apply:

`Parameters`, `Returns`, `Yields`, `Raises`, `See Also`, `Notes`, `References`,
`Examples`.

In the `name : type` slot, write the **semantic** type — shape, dtype, valid
range, default, units. Type hints already carry the bare type, so the slot is
where `ndarray of shape (n,)`, `float in [0, 1]`, or `float > 0, in units of the
sensitivity` earns its keep.

`Examples` are doctest format with deterministic output, and only on the main
entry points users actually call. `References` for anything implementing a paper.

```python
def rdp_gaussian(alpha: float, sigma: float) -> float:
    """Return the Renyi divergence of the Gaussian mechanism at order ``alpha``.

    Parameters
    ----------
    alpha : float > 1
        Renyi order.
    sigma : float > 0
        Noise multiplier, in units of the sensitivity.

    Returns
    -------
    float
        RDP epsilon at order ``alpha``, in nats.

    Raises
    ------
    ValueError
        If ``alpha <= 1`` or ``sigma <= 0``.

    References
    ----------
    .. [1] Mironov, "Renyi Differential Privacy", CSF 2017, Prop. 7.
    """
```

## Classes

- The **class** docstring says what an instance represents, then carries the
  constructor's `Parameters` section and an `Attributes` section for the public
  attributes.
- **`__init__` never has a docstring.** Its parameters are documented on the class.
- **Properties** are documented like attributes: a summary line, no `Returns`.

## Modules

A short docstring above the imports: one line of purpose, at most a sentence of
orientation. No API listings, no rationale.

## Write nothing for

- Standard dunders doing the obvious thing (`__repr__`, `__len__`, `__eq__`).
- Overrides that don't change the parent's contract.
- One-line helpers whose name is the documentation.

A docstring that adds no information is noise, and noise drifts out of sync.

## Tests

Test functions and test modules need no docstrings. Add a one-liner only when a
test's intent isn't clear from its name.
