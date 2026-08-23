"""Logistic regression - one linear layer, no hashing.

The logit is ``dot(w, x) + b``. That is the whole model: the smallest
thing that still trains, so that a run which goes wrong is easier to
attribute to the algorithm than to the architecture.

Parameter pytree
----------------
``{"w": (d,), "b": ()}`` - a plain dict of JAX arrays, which
``jax.grad`` / ``jax.vmap`` / ``jax.jit`` traverse without help, and
which stages 4 and 5 treat like any other pytree.

Feature layout
--------------
None. ``x`` is a feature vector of length ``d`` and this module never
asks what its entries mean, so encoding a dataset into one - which
columns are dense, how categoricals are represented, what is normalized
- stays with the caller.

There are two forwards over the same parameters and the same model.
``forward`` takes the dense vector; ``forward_sparse`` takes the
coordinates a row occupies and the values it puts there, for a ``d``
wide enough that the dense vector is the problem. They return the same
logit and differ in nothing else.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# Scale of the Gaussian the weights start at. A linear model has no
# symmetry between units to break, so the randomness buys variation
# between runs rather than trainability; the small scale is what keeps
# the initial logits near zero, where the sigmoid is steepest.
_INIT_STD = 0.01


def init_params(key: jax.Array, num_features: int) -> dict:
    """Initialise parameters for a ``num_features``-dimensional input.

    Parameters
    ----------
    key : jax.Array
        A PRNG key, consumed by the weight draw.
    num_features : int > 0
        ``d``, the width of the feature vector :func:`forward` takes.

    Returns
    -------
    dict
        ``{"w": (num_features,), "b": ()}``, the weights drawn from
        ``N(0, 0.01 ** 2)`` (``_INIT_STD``) and the bias at zero.
    """
    w = jax.random.normal(key, (num_features,)) * _INIT_STD
    return {"w": w, "b": jnp.array(0.0)}


def forward(params: dict, x: jax.Array) -> jax.Array:
    """Return the logit for a **single** example. ``vmap`` it for a batch.

    Parameters
    ----------
    params : dict, ``{"w": (d,), "b": ()}``
        Pytree as returned by :func:`init_params`.
    x : jax.Array, shape ``(d,)``
        One feature vector.

    Returns
    -------
    logit : jax.Array, shape ``()``
        Unsquashed; the sigmoid is applied by the loss, not here.
    """
    return jnp.dot(params["w"], x) + params["b"]


def forward_sparse(params: dict, idx: jax.Array, val: jax.Array) -> jax.Array:
    """Return the logit for a **single** example held as index/value pairs.

    Computes exactly what :func:`forward` computes on the dense row the
    pair implies - zeros everywhere except ``val[k]`` at ``idx[k]`` -
    without materialising it. At the widths this is for, materialising
    it is the whole difficulty; ADR-0020 records the numbers. ``vmap``
    it for a batch, the same way as :func:`forward`.

    Parameters
    ----------
    params : dict, ``{"w": (d,), "b": ()}``
        Pytree as returned by :func:`init_params`, whose ``w`` is
        ``(num_features,)`` - the width the indices address, not their
        count.
    idx : jax.Array, shape ``(s,)``
        Integer coordinates into ``w``. Out-of-range entries clamp
        rather than raise, which is JAX's indexing rule and not a check
        this function adds.
    val : jax.Array, shape ``(s,)``
        The value at each coordinate.

    Returns
    -------
    logit : jax.Array, shape ``()``
        Unsquashed; the sigmoid is applied by the loss, not here.
    """
    return jnp.dot(params["w"][idx], val) + params["b"]
