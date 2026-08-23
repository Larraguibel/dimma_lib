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


class SparseTabularSplit(NamedTuple):
    """A train/test split whose rows are index/value pairs, on one device.

    For a feature space too wide to hold densely — a one-hot encoding at
    native cardinalities is the case this exists for — a row is stored
    as the coordinates it occupies and the values it puts there, rather
    than as ``num_features`` floats most of which are zero. Row ``i`` of
    the dense matrix this implies is zeros except

        ``dense[i, idx[i, k]] = val[i, k]``   for ``k`` in ``0..s-1``

    where ``s = idx.shape[1]``. Every row has exactly ``s`` stored
    entries and the indices within a row are distinct, so a row's ℓ₂
    norm is its ``val`` vector's and the count is a property of the
    encoding rather than of the record. ADR-0019 records what that
    exactness is for.

    Attributes
    ----------
    idx_train, idx_test : jnp.ndarray, shape ``(n, s)``
        ``int32`` coordinates into ``0..num_features-1``.
    val_train, val_test : jnp.ndarray, shape ``(n, s)``
        ``float32`` values at those coordinates.
    y_train, y_test : jnp.ndarray, shape ``(n,)``
        Labels, ``float32``.
    num_features : int
        The width the indices address. A field rather than metadata
        because a model is initialised with it: it is part of the
        split's shape, not a note about how the split was made.
    metadata : dict
        Per-dataset extras, as on :class:`TabularSplit` — what the
        loader did to the data, and the statistics it fitted.
    """

    idx_train: jnp.ndarray
    val_train: jnp.ndarray
    y_train: jnp.ndarray
    idx_test: jnp.ndarray
    val_test: jnp.ndarray
    y_test: jnp.ndarray
    num_features: int
    metadata: dict


def arrays_to_sparse_split(
    idx_train_np: np.ndarray,
    val_train_np: np.ndarray,
    y_train_np: np.ndarray,
    idx_test_np: np.ndarray,
    val_test_np: np.ndarray,
    y_test_np: np.ndarray,
    num_features: int,
    device: str = "cpu",
    metadata: dict | None = None,
) -> SparseTabularSplit:
    """Move six pre-split NumPy arrays onto ``device``. Indices int32."""
    dev = resolve_device(device)
    idx_train = jnp.asarray(idx_train_np, dtype=jnp.int32)
    val_train = jnp.asarray(val_train_np, dtype=jnp.float32)
    y_train = jnp.asarray(y_train_np, dtype=jnp.float32)
    idx_test = jnp.asarray(idx_test_np, dtype=jnp.int32)
    val_test = jnp.asarray(val_test_np, dtype=jnp.float32)
    y_test = jnp.asarray(y_test_np, dtype=jnp.float32)
    return SparseTabularSplit(
        idx_train=jax.device_put(idx_train, dev),
        val_train=jax.device_put(val_train, dev),
        y_train=jax.device_put(y_train, dev),
        idx_test=jax.device_put(idx_test, dev),
        val_test=jax.device_put(val_test, dev),
        y_test=jax.device_put(y_test, dev),
        num_features=int(num_features),
        metadata=metadata if metadata is not None else {},
    )
