"""Device selection from a friendly name.

Lives here rather than in a ``utils`` package because ``datasets.base``
is its only caller; the membership rule is `dimma.core`'s, per ADR-0001.
"""

from __future__ import annotations

import jax

_ALIASES = {"cpu": "cpu", "gpu": "gpu", "cuda": "gpu"}


def resolve_device(name: str) -> jax.Device:
    """Return the first ``jax.Device`` matching ``name``.

    ``name`` is ``"cpu"``, ``"gpu"`` or ``"cuda"``, case-insensitive,
    and ``"cuda"`` is an alias for ``"gpu"`` — JAX's backend is named
    ``gpu`` whatever the driver. Raises ``ValueError`` on any other name
    and ``RuntimeError`` when the backend has no devices, e.g. ``"cuda"``
    on a CPU-only jaxlib.
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
