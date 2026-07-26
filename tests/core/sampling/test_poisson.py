"""Stage 1 - Poisson subsampling, the standard mechanism."""

from __future__ import annotations

import numpy as np
import pytest

from dimma.core.sampling import poisson


def test_padded_batch_size_exceeds_the_expected_batch():
    assert poisson.padded_batch_size(100, 10_000) > 100


def test_padded_batch_size_grows_with_the_margin():
    assert poisson.padded_batch_size(100, 10_000, margin_sigmas=6.0) > \
        poisson.padded_batch_size(100, 10_000, margin_sigmas=2.0)


def test_padded_batch_size_handles_p_equal_to_one():
    """p = 1 draws every example, so the cap is exactly n."""
    assert poisson.padded_batch_size(50, 50) == 50


def test_padded_batch_size_never_exceeds_the_dataset():
    """No draw can exceed n, so no cap above it is meaningful."""
    for b_expected, n in [(50, 60), (900, 1000), (5, 6), (1, 2)]:
        assert poisson.padded_batch_size(b_expected, n) <= n


def test_a_cap_of_n_cannot_raise(rng):
    """Clamped to n the cap is exact: Binomial(n, p) has no mass above n."""
    n = 200
    for _ in range(50):
        indices, mask = poisson.subsample(rng, n, 0.95, n)
        assert indices.shape == (n,)


def test_returns_padded_indices_and_mask_of_fixed_shape(rng):
    b_max = poisson.padded_batch_size(20, 1000)
    indices, mask = poisson.subsample(rng, 1000, 0.02, b_max)
    assert indices.shape == (b_max,)
    assert mask.shape == (b_max,)


def test_dtypes_are_stable(rng):
    """Downstream indexing and masking depend on these."""
    b_max = poisson.padded_batch_size(20, 1000)
    indices, mask = poisson.subsample(rng, 1000, 0.02, b_max)
    assert indices.dtype == np.int64
    assert mask.dtype == np.float32


def test_mask_is_ones_then_zeros(rng):
    b_max = poisson.padded_batch_size(20, 1000)
    _, mask = poisson.subsample(rng, 1000, 0.02, b_max)
    k = int(mask.sum())
    assert np.all(mask[:k] == 1.0)
    assert np.all(mask[k:] == 0.0)


def test_padding_slots_hold_index_zero(rng):
    """Index 0 is a real row; only the mask stops it contributing."""
    b_max = poisson.padded_batch_size(20, 1000)
    indices, mask = poisson.subsample(rng, 1000, 0.02, b_max)
    k = int(mask.sum())
    assert np.all(indices[k:] == 0)


def test_selected_indices_are_in_range_and_distinct(rng):
    n = 1000
    b_max = poisson.padded_batch_size(20, n)
    indices, mask = poisson.subsample(rng, n, 0.02, b_max)
    k = int(mask.sum())
    real = indices[:k]
    assert np.all((real >= 0) & (real < n))
    assert len(np.unique(real)) == k


def test_oversize_draw_raises_rather_than_truncating(rng):
    """Truncating would change the mechanism the accounting assumes."""
    with pytest.raises(RuntimeError, match="b_max"):
        poisson.subsample(rng, 1000, 0.9, 10)


def test_draws_are_reproducible_from_the_generator():
    a = poisson.subsample(np.random.default_rng(7), 1000, 0.02, 60)
    b = poisson.subsample(np.random.default_rng(7), 1000, 0.02, 60)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_the_generator_advances_between_calls(rng):
    """One generator across all steps is what keeps the draws independent."""
    first = poisson.subsample(rng, 1000, 0.02, 60)
    second = poisson.subsample(rng, 1000, 0.02, 60)
    assert not np.array_equal(first[0], second[0])


def test_inclusion_rate_matches_p(rng):
    """Each example is included independently with probability p."""
    n, p = 2000, 0.05
    b_max = poisson.padded_batch_size(int(n * p), n)
    counts = [
        int(poisson.subsample(rng, n, p, b_max)[1].sum()) for _ in range(200)
    ]
    assert np.isclose(np.mean(counts) / n, p, rtol=0.05)
