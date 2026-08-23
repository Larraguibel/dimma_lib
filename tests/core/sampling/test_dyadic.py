"""Stage 1 - dyadic subsampling, a fixed-size draw at a random scale."""

from __future__ import annotations

import collections
import math

import numpy as np
import pytest

from dimma.core.sampling import dyadic, poisson


def test_max_scale_is_floor_log2_n_minus_one():
    """Definition 5.1's M, by bit length rather than by ``math.log2``."""
    table = {2: 0, 3: 0, 4: 1, 7: 1, 8: 2, 1000: 8, 45_000_000: 24}
    assert {n: dyadic.max_scale(n) for n in table} == table


def test_the_largest_batch_fits_the_dataset():
    """Why no oversize draw is possible, and so nothing is capped."""
    for n in range(2, 500):
        assert 2 ** (dyadic.max_scale(n) + 1) <= n


def test_a_dataset_below_two_is_rejected(rng):
    """The smallest batch on the ladder holds two examples."""
    for n in [1, 0, -3]:
        with pytest.raises(ValueError, match="n="):
            dyadic.max_scale(n)
        with pytest.raises(ValueError, match="n="):
            dyadic.subsample(rng, n, 0)


def test_a_negative_max_scale_has_no_law():
    """The law is over 0..M, which is empty below zero."""
    with pytest.raises(ValueError, match="max_scale="):
        dyadic.scale_probabilities(-1)


def test_the_scale_probabilities_are_definition_51():
    """``p_k = C_M / 2 ** k``, summing to one, with ``C_M`` in [0.5, 1]
    as Ghazi et al.'s Remark 5.2 records."""
    for m in [0, 1, 5, 24]:
        p = dyadic.scale_probabilities(m)
        c = 1.0 / (2.0 * (1.0 - 2.0 ** -(m + 1)))
        assert p.shape == (m + 1,)
        assert np.allclose(p, [c / 2 ** k for k in range(m + 1)])
        assert math.isclose(p.sum(), 1.0, rel_tol=1e-12)
        assert 0.5 <= c <= 1.0


def test_the_scale_distribution_matches_its_pmf(rng):
    """The coin realizes the law, so the debias weight ``1 / p_N`` is
    the weight of the scale that actually ran."""
    m, draws = 5, 200_000
    counts = collections.Counter(
        dyadic.draw_scale(rng, m) for _ in range(draws)
    )
    expected = dyadic.scale_probabilities(m)
    for k, p in enumerate(expected):
        standard_error = math.sqrt(draws * p * (1.0 - p))
        assert abs(counts[k] - draws * p) <= 4.0 * standard_error


def test_the_draw_has_exactly_two_to_the_scale_plus_one_indices(rng):
    """Cardinality is fixed by the scale, not by the data."""
    for scale in range(0, dyadic.max_scale(1024) + 1):
        draw = dyadic.subsample(rng, 1024, scale)
        assert draw.whole.shape == (2 ** (scale + 1),)
        assert draw.single.shape == (1,)


def test_the_draw_is_without_replacement(rng):
    for _ in range(50):
        draw = dyadic.subsample(rng, 64, 3)
        assert len(np.unique(draw.whole)) == draw.whole.size
        assert np.all((draw.whole >= 0) & (draw.whole < 64))
        assert 0 <= int(draw.single[0]) < 64


def test_the_halves_are_exact_and_disjoint_and_cover_the_whole(rng):
    """The debiasing identity rests on O and E being an exact equal
    partition of B, at every scale."""
    n = 1024
    for scale in range(0, dyadic.max_scale(n) + 1):
        draw = dyadic.subsample(rng, n, scale)
        assert draw.odd.size == draw.even.size == 2 ** scale
        assert np.intersect1d(draw.odd, draw.even).size == 0
        assert np.array_equal(np.concatenate([draw.odd, draw.even]),
                              draw.whole)


def test_every_subset_of_the_right_size_is_equally_likely(rng):
    """Exactness, not an approximation: the batch is uniform over the
    15 four-element subsets of six examples.

    This is what a Poisson stand-in for the fixed-size draw would fail,
    and Lemma 5.3's amplification is stated for the exact draw.
    """
    n, scale, draws = 6, 1, 60_000
    counts = collections.Counter(
        frozenset(dyadic.subsample(rng, n, scale).whole.tolist())
        for _ in range(draws)
    )
    subsets = math.comb(n, 4)
    assert len(counts) == subsets
    p = 1.0 / subsets
    standard_error = math.sqrt(draws * p * (1.0 - p))
    for count in counts.values():
        assert abs(count - draws * p) <= 4.0 * standard_error


def test_the_partition_is_uniform_over_splits(rng):
    """The split is ours, not an ordering NumPy happens to return: at
    n = 4 and scale 0 all 12 ordered (odd, even) pairs are equally
    likely."""
    n, scale, draws = 4, 0, 60_000
    counts = collections.Counter(
        (int(draw.odd[0]), int(draw.even[0]))
        for draw in (dyadic.subsample(rng, n, scale) for _ in range(draws))
    )
    pairs = n * (n - 1)
    assert len(counts) == pairs
    p = 1.0 / pairs
    standard_error = math.sqrt(draws * p * (1.0 - p))
    for count in counts.values():
        assert abs(count - draws * p) <= 4.0 * standard_error


def test_the_single_index_is_uniform(rng):
    n, draws = 10, 20_000
    counts = collections.Counter(
        int(dyadic.subsample(rng, n, 0).single[0]) for _ in range(draws)
    )
    p = 1.0 / n
    standard_error = math.sqrt(draws * p * (1.0 - p))
    for value in range(n):
        assert abs(counts[value] - draws * p) <= 4.0 * standard_error


def test_the_single_index_may_collide_with_the_batch(rng):
    """No rejection: I is uniform over [n] independently of B, so it
    lands in B at exactly the rate ``2 ** (scale + 1) / n``. Rejecting
    collisions would drive this to zero and change the distribution the
    sensitivity argument assumes."""
    n, scale, draws = 16, 2, 20_000
    hits = sum(
        int(draw.single[0]) in set(draw.whole.tolist())
        for draw in (dyadic.subsample(rng, n, scale) for _ in range(draws))
    )
    p = 2 ** (scale + 1) / n
    standard_error = math.sqrt(draws * p * (1.0 - p))
    assert abs(hits - draws * p) <= 4.0 * standard_error


def test_draws_are_reproducible_from_the_generator():
    def draw(seed):
        rng = np.random.default_rng(seed)
        scale = dyadic.draw_scale(rng, dyadic.max_scale(256))
        return dyadic.subsample(rng, 256, scale)

    first, second = draw(7), draw(7)
    assert first.scale == second.scale
    assert np.array_equal(first.whole, second.whole)
    assert np.array_equal(first.single, second.single)


def test_the_generator_advances_between_calls(rng):
    """One generator across all steps is what keeps the draws
    independent."""
    first = dyadic.subsample(rng, 256, 3)
    second = dyadic.subsample(rng, 256, 3)
    assert not np.array_equal(first.whole, second.whole)


def test_dtypes_are_stable(rng):
    """Downstream indexing depends on these, and the scale sizes a
    compiled call, so it stays a host `int`."""
    draw = dyadic.subsample(rng, 256, 3)
    assert draw.whole.dtype == np.int64
    assert draw.single.dtype == np.int64
    assert isinstance(draw.scale, int)
    assert isinstance(dyadic.draw_scale(rng, 4), int)


def test_a_scale_above_the_maximum_is_rejected(rng):
    """Above the maximum the batch would not fit the training set."""
    n = 100  # max_scale 5, so a batch of 64
    with pytest.raises(ValueError, match="scale="):
        dyadic.subsample(rng, n, dyadic.max_scale(n) + 1)
    with pytest.raises(ValueError, match="scale="):
        dyadic.subsample(rng, n, -1)


def test_nothing_is_padded_and_no_mask_is_returned(rng):
    """The contrast with `poisson.subsample`, whose cardinality is
    data-dependent and which returns an ``(indices, mask)`` pair."""
    draw = dyadic.subsample(rng, 1000, 4)
    assert draw._fields == ("scale", "whole", "single")
    assert not hasattr(draw, "mask")
    assert draw.whole.size == 2 ** 5

    indices, mask = poisson.subsample(rng, 1000, 0.02, 60)
    assert indices.size == mask.size == 60
