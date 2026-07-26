"""Stage 1 - Poisson subsampling with truncation, a modified mechanism."""

from __future__ import annotations

import numpy as np

from dimma.core.sampling import poisson, poisson_truncated


def test_oversize_draw_truncates_instead_of_raising():
    """The whole difference from the standard mechanism."""
    rng = np.random.default_rng(0)
    indices, mask = poisson_truncated.subsample(rng, 1000, 0.9, 10)
    assert indices.shape == (10,)
    assert int(mask.sum()) == 10


def test_truncated_batch_has_no_padding():
    """Truncation fills every slot, so the mask is all ones."""
    rng = np.random.default_rng(0)
    _, mask = poisson_truncated.subsample(rng, 1000, 0.9, 10)
    assert np.all(mask == 1.0)


def test_truncated_indices_stay_distinct_and_in_range():
    rng = np.random.default_rng(0)
    n = 1000
    indices, _ = poisson_truncated.subsample(rng, n, 0.9, 10)
    assert np.all((indices >= 0) & (indices < n))
    assert len(np.unique(indices)) == 10


def test_matches_the_standard_sampler_when_nothing_is_truncated():
    """Below the cap the two mechanisms are the same draw."""
    n, p, b_max = 1000, 0.02, 60
    strict = poisson.subsample(np.random.default_rng(3), n, p, b_max)
    truncated = poisson_truncated.subsample(np.random.default_rng(3), n, p, b_max)
    assert np.array_equal(strict[0], truncated[0])
    assert np.array_equal(strict[1], truncated[1])


def test_returns_fixed_shapes_and_stable_dtypes():
    rng = np.random.default_rng(0)
    b_max = poisson.padded_batch_size(20, 1000)
    indices, mask = poisson_truncated.subsample(rng, 1000, 0.02, b_max)
    assert indices.shape == (b_max,)
    assert mask.shape == (b_max,)
    assert indices.dtype == np.int64
    assert mask.dtype == np.float32


def test_draws_are_reproducible_from_the_generator():
    a = poisson_truncated.subsample(np.random.default_rng(7), 1000, 0.9, 10)
    b = poisson_truncated.subsample(np.random.default_rng(7), 1000, 0.9, 10)
    assert np.array_equal(a[0], b[0])
