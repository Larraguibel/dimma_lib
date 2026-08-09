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

    Returns ``{"w": (num_features,), "b": ()}``, weights drawn from a
    Gaussian of scale ``_INIT_STD`` and the bias at zero.
    """
    w = jax.random.normal(key, (num_features,)) * _INIT_STD
    return {"w": w, "b": jnp.array(0.0)}


def forward(params: dict, x: jax.Array) -> jax.Array:
    """The logit for a **single** example. ``vmap`` it for a batch.

    Parameters
    ----------
    params
        Pytree as returned by :func:`init_params`.
    x : jax.Array, shape ``(d,)``
        One feature vector.

    Returns
    -------
    logit : jax.Array, shape ``()``
        Unsquashed; the sigmoid is applied by the loss, not here.
    """
    return jnp.dot(params["w"], x) + params["b"]
