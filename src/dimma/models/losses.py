"""Sigmoid binary cross-entropy, for the models shipped here.

`per_sample_bce_loss` is what an algorithm is handed: it takes one
example, which is what makes a per-sample gradient definable at all.
`batch_bce_loss` is a number to report, not something to differentiate.

Both evaluate `_stable_bce`, so the loss is written down once.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dimma.models.logreg import forward


def _stable_bce(logit: jax.Array, y: jax.Array) -> jax.Array:
    """Evaluate sigmoid BCE from a logit, elementwise and overflow-safe.

    Notes
    -----
    ``max(z, 0) - z*y + log1p(exp(-|z|))`` is the direct form
    rearranged so the only exponential has a non-positive argument. Do
    not restore the direct form: in float32 it loses the leading digits
    of ``1 - sigmoid(z)`` from ``|z|`` of about 15 and takes ``log(0)``
    from about 17, returning a plausible wrong number well before
    ``exp`` itself overflows at 88.
    """
    return jnp.maximum(logit, 0.0) - logit * y + jnp.log1p(jnp.exp(-jnp.abs(logit)))


def per_sample_bce_loss(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
    """Return the BCE of a **single** example under `dimma.models.logreg`.

    This is the ``(params, x_single, y_single) -> scalar`` an algorithm
    takes. Composed with the linear logit, its gradient is
    ``(sigmoid(z) - y) * x`` in ``w`` and ``sigmoid(z) - y`` in ``b``.

    Parameters
    ----------
    params : dict, ``{"w": (d,), "b": ()}``
        Pytree as returned by `dimma.models.logreg.init_params`.
    x : jax.Array, shape ``(d,)``
        One feature vector.
    y : jax.Array, shape ``()``
        Binary label in ``{0., 1.}``.

    Returns
    -------
    jax.Array, shape ``()``
        The example's loss, in nats.
    """
    return _stable_bce(forward(params, x), y)


def batch_bce_loss(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
    """Return the mean BCE over a batch, a number to report at the call site.

    Parameters
    ----------
    params : dict, ``{"w": (d,), "b": ()}``
        Pytree as returned by `dimma.models.logreg.init_params`.
    x : jax.Array, shape ``(B, d)``
        A batch of feature vectors.
    y : jax.Array, shape ``(B,)``
        Binary labels in ``{0., 1.}``.

    Returns
    -------
    jax.Array, shape ``()``
        The batch's mean loss, in nats.
    """
    logits = jax.vmap(forward, in_axes=(None, 0))(params, x)
    return jnp.mean(_stable_bce(logits, y))
