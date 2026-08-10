"""Stage 1 - shuffled epoch sampling, which is not a mechanism."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from dimma.core.sampling import shuffled


def take(stream, count):
    return list(itertools.islice(stream, count))


def test_every_batch_has_the_requested_size(rng):
    """Fixed cardinality is the property the compiled step rests on."""
    for batch in take(shuffled.batches(rng, 100, 16), 20):
        assert batch.shape == (16,)


def test_a_batch_is_indices_and_not_a_pair(rng):
    """No mask: there is no padding here to mask out."""
    batch = next(shuffled.batches(rng, 100, 10))
    assert isinstance(batch, np.ndarray)
    assert np.issubdtype(batch.dtype, np.integer)


def test_indices_are_in_range(rng):
    for batch in take(shuffled.batches(rng, 50, 7), 30):
        assert batch.min() >= 0 and batch.max() < 50


def test_an_epoch_visits_every_example_exactly_once(rng):
    """What makes this ordinary sampling rather than a subsample."""
    n, batch_size = 60, 12
    epoch = np.concatenate(take(shuffled.batches(rng, n, batch_size),
                                n // batch_size))
    assert np.array_equal(np.sort(epoch), np.arange(n))


def test_batches_within_an_epoch_are_disjoint(rng):
    batches = take(shuffled.batches(rng, 60, 12), 5)
    seen = np.concatenate(batches)
    assert len(np.unique(seen)) == seen.size


def test_the_epoch_remainder_is_dropped(rng):
    """A short batch is a second shape and would retrace the step.

    n = 10 at a batch of 3 yields three full batches and discards one
    index: nine distinct indices out, one left over, and no fourth
    short batch.
    """
    epoch = np.concatenate(take(shuffled.batches(rng, 10, 3), 3))
    assert epoch.size == 9
    assert len(np.unique(epoch)) == 9
    assert np.setdiff1d(np.arange(10), epoch).size == 1


def test_the_stream_does_not_end_at_an_epoch_boundary(rng):
    """A loop above is counted in steps, not epochs."""
    assert len(take(shuffled.batches(rng, 20, 5), 41)) == 41


def test_consecutive_epochs_are_shuffled_differently(rng):
    n, batch_size = 60, 12
    stream = shuffled.batches(rng, n, batch_size)
    first = np.concatenate(take(stream, n // batch_size))
    second = np.concatenate(take(stream, n // batch_size))
    assert not np.array_equal(first, second)


def test_a_stream_is_reproducible_from_its_seed():
    def epoch(seed):
        return np.concatenate(
            take(shuffled.batches(np.random.default_rng(seed), 40, 10), 4)
        )

    assert np.array_equal(epoch(7), epoch(7))


def test_different_seeds_give_different_orders():
    def epoch(seed):
        return np.concatenate(
            take(shuffled.batches(np.random.default_rng(seed), 40, 10), 4)
        )

    assert not np.array_equal(epoch(0), epoch(1))


@pytest.mark.parametrize("batch_size", [0, -1, 41])
def test_a_batch_size_outside_the_dataset_is_rejected(rng, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        shuffled.batches(rng, 40, batch_size)


def test_the_rejection_happens_when_the_stream_is_built(rng):
    """Not on first advance: a deferred raise inside a generator body
    would surface a step into the loop rather than at the call."""
    with pytest.raises(ValueError, match="batch_size"):
        shuffled.batches(rng, 40, 100)


def test_a_batch_of_the_whole_dataset_is_allowed(rng):
    """Full-batch descent is the degenerate end of the range."""
    batch = next(shuffled.batches(rng, 40, 40))
    assert np.array_equal(np.sort(batch), np.arange(40))
