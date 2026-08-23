"""A synthetic Criteo file, so the eight modes are testable without 45 MB.

The real schema and shape (label, I1..I13, C1..C26), with the NaN, the
negative value and the unseen category planted where the split will
send them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dimma.datasets._attribution import reset_emitted
from dimma.datasets.criteo import CAT_COLS, INT_COLS, LABEL_COL

N_ROWS = 200
SEED = 0
TEST_FRACTION = 0.2

#: A category ID planted in a test-split row and nowhere else.
UNSEEN_CATEGORY = 999_999


@pytest.fixture(autouse=True)
def quiet_notices():
    """Each test starts with no notice emitted, and leaves none behind."""
    reset_emitted()
    yield
    reset_emitted()


def split_indices(
    n: int = N_ROWS, seed: int = SEED, test_fraction: float = TEST_FRACTION
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the loader's permutation, pinning the split rule."""
    perm = np.random.default_rng(seed).permutation(n)
    n_test = int(round(n * test_fraction))
    return perm[n_test:], perm[:n_test]


@pytest.fixture
def criteo_frame() -> pd.DataFrame:
    """The source frame the fixture file is written from."""
    rng = np.random.default_rng(1234)
    train_idx, test_idx = split_indices()

    data = {LABEL_COL: rng.integers(0, 2, N_ROWS).astype(np.int64)}
    for j, col in enumerate(INT_COLS):
        values = rng.integers(0, 500, N_ROWS).astype(np.float64)
        values[j] = np.nan          # one NaN per column, in a distinct row
        values[j + 20] = -7.0       # and one negative, for the clip
        data[col] = values
    for col in CAT_COLS:
        # Few enough distinct IDs that some appear many times: frequency
        # encoding is only meaningful when frequencies differ.
        data[col] = rng.integers(1000, 1012, N_ROWS).astype(np.int64)

    frame = pd.DataFrame(data)
    frame.loc[test_idx[0], "C1"] = UNSEEN_CATEGORY
    assert UNSEEN_CATEGORY not in set(frame.loc[train_idx, "C1"])
    return frame


@pytest.fixture
def criteo_root(tmp_path, criteo_frame) -> "pytest.TempPathFactory":
    """A cache directory holding the synthetic ``criteo_1M.parquet``."""
    criteo_frame.to_parquet(tmp_path / "criteo_1M.parquet")
    return tmp_path
