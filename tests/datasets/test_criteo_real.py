"""The claims that only the real 1M sample can settle.

Skipped unless the file is already cached, so a fresh checkout does not
pull 45 MB to run the suite. Everything about the *logic* is covered
against the synthetic fixture in ``test_criteo.py``; what is here is what
depends on the actual data — its width, its category cardinalities, and
whether float32 is wide enough for the IDs it happens to contain.

To populate the cache::

    python -c "from dimma.datasets.criteo import load_criteo; load_criteo()"

— and what standardizing does to the row norms, which is a fact about
the real distribution.

Each mode is loaded once for the module. Preprocessing a million rows is
seconds of work, and repeating it per test would make the suite slower
than the thing it is testing.
"""

from __future__ import annotations

import numpy as np
import pytest

from dimma.datasets._cache import get_cache_dir
from dimma.datasets.criteo import CAT_COLS, INT_COLS, load_criteo

CACHED = get_cache_dir("datasets") / "criteo_1M.parquet"

pytestmark = pytest.mark.skipif(
    not CACHED.exists(), reason=f"Criteo sample not cached at {CACHED}"
)

#: Not all eight: `standardize` is a column map whose logic the fixture
#: settles, and a million rows say nothing more about it than 200 do.
#: The one combination worth the seconds is the standardized numeric
#: chain, because what it does to the row norms is a property of the
#: real distribution and of nothing smaller.
MODES = [
    ("numeric", False, False),
    ("numeric", True, False),
    ("numeric", True, True),
    ("all", False, False),
    ("all", True, False),
]
WIDTH = {"numeric": 13, "all": 39}


@pytest.fixture(scope="module")
def splits() -> dict:
    return {
        mode: load_criteo(
            features=mode[0],
            preprocess=mode[1],
            standardize=mode[2],
            download=False,
        )
        for mode in MODES
    }


@pytest.mark.parametrize("mode", MODES)
def test_shape(splits, mode):
    split = splits[mode]
    assert split.x_train.shape == (800_000, WIDTH[mode[0]])
    assert split.x_test.shape == (200_000, WIDTH[mode[0]])


def test_train_split_is_standardized_when_asked(splits):
    x = np.asarray(splits[("numeric", True, True)].x_train)
    assert np.allclose(x.mean(axis=0), 0.0, atol=1e-2)
    assert np.allclose(x.std(axis=0), 1.0, atol=1e-2)


def test_the_default_chain_is_not_standardized(splits):
    """`preprocess=True` on this file is close to a no-op — log1p on
    values already inside [0, 1] — so the columns come out small and
    uncentred, which is the state ADR-0013 chose as the default."""
    x = np.asarray(splits[("numeric", True, False)].x_train)
    assert not np.allclose(x.mean(axis=0), 0.0, atol=1e-2)
    assert x.max() <= np.log1p(1.0) + 1e-6


def test_standardizing_multiplies_the_largest_row_norm(splits):
    """The cost the default exists to avoid, on the real distribution.
    `L0` is the largest augmented row norm, so the smallest honest `R` is
    the largest norm here; standardizing sends it from about 2 to about
    15. `L1 = R**2 / 4` therefore grows by a factor near 50, and the step
    size Theorem B.3 prescribes shrinks by the same. ADR-0013 declines to
    spend that by default."""
    plain = np.linalg.norm(np.asarray(splits[("numeric", True, False)].x_train), axis=1)
    scaled = np.linalg.norm(np.asarray(splits[("numeric", True, True)].x_train), axis=1)

    assert plain.max() < 3.0
    assert scaled.max() > 10.0
    assert np.median(plain) < 1.0 < np.median(scaled)


def test_raw_category_ids_survive_float32(splits):
    """They are integers below 2**24, so the cast is exact, not approximate."""
    x = np.asarray(splits[("all", False, False)].x_train)
    categories = x[:, len(INT_COLS):]
    assert categories.max() > 1000
    assert categories.max() < 2**24
    assert np.array_equal(categories, np.round(categories))


def test_frequency_encoding_compresses_the_categorical_block(splits):
    """Hundreds of thousands of IDs become 26 floats, which is the point."""
    split = splits[("all", True, False)]
    assert split.x_train.shape[1] == len(INT_COLS) + len(CAT_COLS)
    assert max(split.metadata["n_categories"]) > 10_000


def test_label_rate_matches_the_published_sample(splits):
    y = np.asarray(splits[("numeric", True, False)].y_train)
    assert 0.24 < y.mean() < 0.26


def test_every_mode_splits_the_same_rows(splits):
    labels = [np.asarray(splits[mode].y_train) for mode in MODES]
    for other in labels[1:]:
        assert np.array_equal(labels[0], other)
