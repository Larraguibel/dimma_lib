"""Device names, and what happens when the backend is not there."""

from __future__ import annotations

import jax
import pytest

from dimma.datasets._device import resolve_device


def test_cpu_always_resolves():
    assert resolve_device("cpu") in jax.devices("cpu")


def test_name_is_case_insensitive():
    assert resolve_device("CPU") == resolve_device("cpu")


def test_cuda_is_an_alias_for_gpu():
    """JAX names the backend ``gpu`` whatever the driver underneath is."""
    try:
        expected = resolve_device("gpu")
    except RuntimeError:
        with pytest.raises(RuntimeError, match="gpu"):
            resolve_device("cuda")
    else:
        assert resolve_device("cuda") == expected


def test_unknown_name_raises_and_lists_the_options():
    with pytest.raises(ValueError, match="cpu, gpu, cuda"):
        resolve_device("xpu")
