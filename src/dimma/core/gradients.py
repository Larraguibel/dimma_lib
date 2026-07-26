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
    """One gradient per example, for per-example DP algorithms.

    Returns ``(params, x_batch, y_batch) -> pytree`` with leaves of
    shape ``(B, *param_shape)``, the layout stages 4 and 5 expect.

    Build it once outside the training loop so ``jax.jit`` sees a stable
    compilation key.
    """
    return jax.vmap(jax.grad(per_sample_loss_fn), in_axes=(None, 0, 0))


def batch_grads(per_sample_loss_fn: Callable) -> Callable:
    """One gradient for the mean loss, for the non-private baselines.

    Returns ``(params, x_batch, y_batch) -> pytree`` with no leading
    batch axis. Stages 4 and 6 are dropped and stage 5 is already done.
    Takes the same loss a private algorithm would be given, which is
    what keeps the comparison controlled.
    """
    batched_loss = jax.vmap(per_sample_loss_fn, in_axes=(None, 0, 0))

    def mean_loss(params, x_batch, y_batch):
        return jnp.mean(batched_loss(params, x_batch, y_batch))

    return jax.grad(mean_loss)
