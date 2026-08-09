"""Sigmoid binary cross-entropy, for the models shipped here.

`per_sample_bce_loss` is what an algorithm is handed: it takes one
example, which is what makes a per-sample gradient definable at all.
`batch_bce_loss` is a number to report, not something to differentiate -
`dimma.core.gradients` builds every gradient function the pipeline needs
out of the per-sample form, private and non-private alike.

Both evaluate `_stable_bce`, so there is one objective defined once.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dimma.models.logreg import forward


def _stable_bce(logit: jax.Array, y: jax.Array) -> jax.Array:
    """Sigmoid BCE from a logit, elementwise and overflow-safe.

    Evaluates ``max(z, 0) - z*y + log1p(exp(-|z|))``, which is
    ``-[y*log(sigmoid(z)) + (1-y)*log(1-sigmoid(z))]`` rearranged so
    that the only exponential taken has a non-positive argument. In
    float32 the direct form loses the leading digits of
    ``1 - sigmoid(z)`` from ``|z|`` of about 14 and takes ``log(0)``
    from about 17, so it returns a plausible wrong number before it
    returns ``inf`` or ``nan`` - both well short of the 88 where
    ``exp`` itself overflows. This form is finite across the whole
    float32 range, tending to ``max(z, 0) - z*y`` as ``|z|`` grows.

    Broadcasts over any shape: a scalar for the per-sample loss, a
    ``(B,)`` vector for the batch one.
    """
    return jnp.maximum(logit, 0.0) - logit * y + jnp.log1p(jnp.exp(-jnp.abs(logit)))


def per_sample_bce_loss(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
    """BCE for a **single** example under `dimma.models.logreg`.

    This is the ``(params, x_single, y_single) -> scalar`` an algorithm
    takes. Composed with the linear logit, its gradient is
    ``(sigmoid(z) - y) * x`` in ``w`` and ``sigmoid(z) - y`` in ``b``.

    Parameters
    ----------
    params
        Pytree as returned by `dimma.models.logreg.init_params`.
    x : jax.Array, shape ``(d,)``
        One feature vector.
    y : jax.Array, shape ``()``
        Binary label in ``{0., 1.}``.
    """
    return _stable_bce(forward(params, x), y)


def batch_bce_loss(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
    """Mean BCE over a batch, a number to report at the call site.

    Parameters
    ----------
    params
        Pytree as returned by `dimma.models.logreg.init_params`.
    x : jax.Array, shape ``(B, d)``
        A batch of feature vectors.
    y : jax.Array, shape ``(B,)``
        Binary labels in ``{0., 1.}``.
    """
    logits = jax.vmap(forward, in_axes=(None, 0))(params, x)
    return jnp.mean(_stable_bce(logits, y))
