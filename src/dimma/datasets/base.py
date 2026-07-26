"""Dataset-agnostic types, shared by every loader.

Nothing here names a dataset.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dimma.datasets._device import resolve_device


class TabularSplit(NamedTuple):
    """A train/test split of tabular data, on one device.

    Attributes
    ----------
    x_train, y_train, x_test, y_test : jnp.ndarray
        Features and labels, ``float32``, on the requested device.
    metadata : dict
        Per-dataset extras. Loaders record here what they did to the
        data — the preprocessing chain, the statistics it was fitted
        with, the column names, the license — so that a split carries
        its own provenance instead of leaving it in the call site.
    """

    x_train: jnp.ndarray
    y_train: jnp.ndarray
    x_test: jnp.ndarray
    y_test: jnp.ndarray
    metadata: dict


def arrays_to_split(
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    x_test_np: np.ndarray,
    y_test_np: np.ndarray,
    device: str = "cpu",
    metadata: dict | None = None,
) -> TabularSplit:
    """Move four pre-split NumPy arrays onto ``device``."""
    dev = resolve_device(device)
    return TabularSplit(
        x_train=jax.device_put(jnp.asarray(x_train_np), dev),
        y_train=jax.device_put(jnp.asarray(y_train_np), dev),
        x_test=jax.device_put(jnp.asarray(x_test_np), dev),
        y_test=jax.device_put(jnp.asarray(y_test_np), dev),
        metadata=metadata if metadata is not None else {},
    )
