"""Stage 3 - gradients.

Turn a per-sample loss into a gradient function. Stage 2 has no
primitive here: the model is the caller's and runs inside the loss.

A dimma per-sample loss is ``(params, x_single, y_single) -> scalar``,
taking one example rather than a batch. Both factories vectorize it
with ``in_axes=(None, 0, 0)``.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def per_sample_grads(per_sample_loss_fn: Callable) -> Callable:
    """Return a function giving one gradient per example.

    What the per-example DP algorithms differentiate with.

    Parameters
    ----------
    per_sample_loss_fn : callable
        ``(params, x_single, y_single) -> scalar``, taking one example
        rather than a batch.

    Returns
    -------
    callable
        ``(params, x_batch, y_batch) -> pytree`` matching ``params``,
        with leaves of shape ``(B, *param_shape)`` — the layout stages
        4 and 5 expect. ``x_batch`` and ``y_batch`` are mapped over
        their leading axis; ``params`` is not.

    Notes
    -----
    Build it once outside the training loop so ``jax.jit`` sees a
    stable compilation key.
    """
    return jax.vmap(jax.grad(per_sample_loss_fn), in_axes=(None, 0, 0))


def batch_grads(per_sample_loss_fn: Callable) -> Callable:
    """Return a function giving one gradient for the mean loss.

    What the non-private baselines differentiate with: stages 4 and 6
    are dropped and stage 5 is already done. Takes the same loss a
    private algorithm would be given, which is what keeps the
    comparison controlled.

    Parameters
    ----------
    per_sample_loss_fn : callable
        ``(params, x_single, y_single) -> scalar``, the same loss
        :func:`per_sample_grads` takes.

    Returns
    -------
    callable
        ``(params, x_batch, y_batch) -> pytree`` matching ``params``,
        with no leading batch axis: the gradient of the batch's mean
        loss.
    """
    batched_loss = jax.vmap(per_sample_loss_fn, in_axes=(None, 0, 0))

    def mean_loss(params, x_batch, y_batch):
        return jnp.mean(batched_loss(params, x_batch, y_batch))

    return jax.grad(mean_loss)
