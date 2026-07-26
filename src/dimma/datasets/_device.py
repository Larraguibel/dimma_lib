"""Device selection from a friendly name.

Lives here rather than in a ``utils`` package because ``datasets.base``
is its only caller: `core`'s membership rule says a thing with one
consumer belongs with that consumer. Promote it when a second one shows
up.
"""

from __future__ import annotations

import jax

_ALIASES = {"cpu": "cpu", "gpu": "gpu", "cuda": "gpu"}


def resolve_device(name: str) -> jax.Device:
    """Return the first ``jax.Device`` matching ``name``.

    Parameters
    ----------
    name : str
        ``"cpu"``, ``"gpu"``, or ``"cuda"`` (case-insensitive). ``"cuda"``
        is an alias for ``"gpu"`` — JAX's backend is named ``gpu``
        whatever the underlying driver.

    Returns
    -------
    jax.Device

    Raises
    ------
    ValueError
        If ``name`` is not one of the three.
    RuntimeError
        If the requested backend has no devices, e.g. ``"cuda"`` on a
        machine whose jaxlib is CPU-only.
    """
    key = name.lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown device {name!r}. Expected one of: cpu, gpu, cuda."
        )
    backend = _ALIASES[key]
    try:
        devices = jax.devices(backend)
    except RuntimeError as e:
        raise RuntimeError(
            f"Requested device {name!r} but JAX backend {backend!r} is "
            f"unavailable. Install a CUDA-enabled jaxlib to use the GPU, or "
            f"pass device='cpu'. Original error: {e}"
        ) from e
    if not devices:
        raise RuntimeError(f"No devices found for backend {backend!r}.")
    return devices[0]
