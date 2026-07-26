"""The four modes, and the line between them.

Two claims are worth more than the shapes. First, that ``preprocess`` is
honest in both directions: ``False`` returns the stored bytes and
``True`` returns exactly the chain the metadata says it returned.
Second, that every fitted statistic comes from the training split, so
that a test-split row cannot influence the features of a training-split
row.
"""

from __future__ import annotations

import numpy as np
import pytest

from dimma.datasets.criteo import (
    CAT_COLS,
    INT_COLS,
    _frequency_encode,
    load_criteo,
)

from .conftest import N_ROWS, TEST_FRACTION, UNSEEN_CATEGORY, split_indices

MODES = [("numeric", False), ("numeric", True), ("all", False), ("all", True)]
WIDTH = {"numeric": 13, "all": 39}


def load(criteo_root, features, preprocess):
    return load_criteo(
        features=features,
        preprocess=preprocess,
        root=criteo_root,
        download=False,
    )


# --------------------------------------------------------------------
# Argument handling
# --------------------------------------------------------------------


def test_unknown_features_raises(criteo_root):
    with pytest.raises(ValueError, match="numeric"):
        load(criteo_root, "integer", True)


def test_missing_file_without_download_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_criteo(root=tmp_path, download=False)


def test_default_mode_is_numeric_and_preprocessed(criteo_root):
    split = load_criteo(root=criteo_root, download=False)
    assert split.metadata["features"] == "numeric"
    assert split.metadata["preprocess"] is True


# --------------------------------------------------------------------
# Shape and split
# --------------------------------------------------------------------


@pytest.mark.parametrize("features,preprocess", MODES)
def test_column_count(criteo_root, features, preprocess):
    split = load(criteo_root, features, preprocess)
    assert split.x_train.shape[1] == WIDTH[features]
    assert split.x_test.shape[1] == WIDTH[features]


@pytest.mark.parametrize("features,preprocess", MODES)
def test_row_counts_follow_test_fraction(criteo_root, features, preprocess):
    split = load(criteo_root, features, preprocess)
    n_test = int(round(N_ROWS * TEST_FRACTION))
    assert split.x_train.shape[0] == split.y_train.shape[0] == N_ROWS - n_test
    assert split.x_test.shape[0] == split.y_test.shape[0] == n_test


@pytest.mark.parametrize("features,preprocess", MODES)
def test_everything_is_float32(criteo_root, features, preprocess):
    split = load(criteo_root, features, preprocess)
    for array in (split.x_train, split.y_train, split.x_test, split.y_test):
        assert array.dtype == np.float32


def test_all_four_modes_split_the_same_rows(criteo_root):
    """Otherwise two modes could not be compared on the same benchmark."""
    labels = [np.asarray(load(criteo_root, *mode).y_train) for mode in MODES]
    for other in labels[1:]:
        assert np.array_equal(labels[0], other)


def test_seed_selects_the_split(criteo_root, criteo_frame):
    train_idx, _ = split_indices()
    split = load(criteo_root, "numeric", False)
    expected = criteo_frame.iloc[train_idx]["label"].to_numpy(dtype=np.float32)
    assert np.array_equal(np.asarray(split.y_train), expected)


# --------------------------------------------------------------------
# preprocess=False returns what is stored
# --------------------------------------------------------------------


@pytest.mark.parametrize("features", ["numeric", "all"])
def test_raw_mode_is_the_stored_values(criteo_root, criteo_frame, features):
    columns = INT_COLS if features == "numeric" else INT_COLS + CAT_COLS
    train_idx, test_idx = split_indices()
    split = load(criteo_root, features, False)

    expected_train = criteo_frame.iloc[train_idx][columns].to_numpy(np.float32)
    expected_test = criteo_frame.iloc[test_idx][columns].to_numpy(np.float32)
    assert np.array_equal(np.asarray(split.x_train), expected_train, equal_nan=True)
    assert np.array_equal(np.asarray(split.x_test), expected_test, equal_nan=True)


@pytest.mark.parametrize("features", ["numeric", "all"])
def test_raw_mode_preserves_nan(criteo_root, features):
    split = load(criteo_root, features, False)
    assert np.isnan(np.asarray(split.x_train)).any()


@pytest.mark.parametrize("features", ["numeric", "all"])
def test_raw_mode_preserves_negative_values(criteo_root, features):
    """The clip belongs to preprocessing, not to the loader."""
    assert np.nanmin(np.asarray(load(criteo_root, features, False).x_train)) < 0


# --------------------------------------------------------------------
# preprocess=True applies the chain it advertises
# --------------------------------------------------------------------


@pytest.mark.parametrize("features", ["numeric", "all"])
def test_preprocessing_removes_nan(criteo_root, features):
    split = load(criteo_root, features, True)
    assert not np.isnan(np.asarray(split.x_train)).any()
    assert not np.isnan(np.asarray(split.x_test)).any()


@pytest.mark.parametrize("features", ["numeric", "all"])
def test_train_split_is_standardized(criteo_root, features):
    x = np.asarray(load(criteo_root, features, True).x_train)
    assert np.allclose(x.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(x.std(axis=0), 1.0, atol=1e-5)


@pytest.mark.parametrize("features", ["numeric", "all"])
def test_test_split_uses_the_train_statistics(criteo_root, features):
    """Standardizing the test split by its own moments would leak it."""
    x = np.asarray(load(criteo_root, features, True).x_test)
    assert not np.allclose(x.mean(axis=0), 0.0, atol=1e-6)


def test_integer_chain_is_median_fill_clip_log1p(criteo_root, criteo_frame):
    train_idx, _ = split_indices()
    split = load(criteo_root, "numeric", True)

    raw = criteo_frame.iloc[train_idx][INT_COLS]
    expected = raw.fillna(raw.median(numeric_only=True)).to_numpy(np.float32)
    expected = np.log1p(np.clip(expected, 0.0, None))
    means, stds = split.metadata["feature_means"], split.metadata["feature_stds"]
    expected = (expected - means) / stds

    assert np.allclose(np.asarray(split.x_train), expected, atol=1e-5)


def test_median_comes_from_the_train_split_only(criteo_root, criteo_frame):
    """A test-split value must not move a training-split feature."""
    baseline = np.asarray(load(criteo_root, "numeric", True).x_train)

    _, test_idx = split_indices()
    perturbed = criteo_frame.copy()
    perturbed.loc[test_idx, INT_COLS] = 1e6
    perturbed.to_parquet(criteo_root / "criteo_1M.parquet")

    after = np.asarray(load(criteo_root, "numeric", True).x_train)
    assert np.array_equal(baseline, after)


# --------------------------------------------------------------------
# Frequency encoding
# --------------------------------------------------------------------


def test_frequency_encode_returns_relative_frequencies():
    train = np.array([10, 10, 10, 20])
    encoded, n_categories = _frequency_encode(train, np.array([10, 20]))
    assert np.allclose(encoded, [0.75, 0.25])
    assert n_categories == 2


def test_frequency_encode_maps_unseen_categories_to_zero():
    train = np.array([10, 20])
    encoded, _ = _frequency_encode(train, np.array([10, 30, 5]))
    assert np.allclose(encoded, [0.5, 0.0, 0.0])


def test_frequency_encode_is_exact_at_the_vocabulary_edges():
    """searchsorted is off by one at both ends unless it is guarded."""
    train = np.array([2, 5, 9])
    encoded, _ = _frequency_encode(train, np.array([2, 9, 1, 10]))
    assert np.allclose(encoded, [1 / 3, 1 / 3, 0.0, 0.0])


def test_categorical_block_collapses_to_one_column_each(criteo_root):
    split = load(criteo_root, "all", True)
    assert split.x_train.shape[1] == len(INT_COLS) + len(CAT_COLS)


def test_unseen_test_category_does_not_produce_nan(criteo_root, criteo_frame):
    _, test_idx = split_indices()
    assert criteo_frame.loc[test_idx[0], "C1"] == UNSEEN_CATEGORY
    x = np.asarray(load(criteo_root, "all", True).x_test)
    assert np.isfinite(x).all()


# --------------------------------------------------------------------
# Metadata and the printed notice
# --------------------------------------------------------------------


@pytest.mark.parametrize("features,preprocess", MODES)
def test_metadata_always_records_the_mode(criteo_root, features, preprocess):
    metadata = load(criteo_root, features, preprocess).metadata
    assert metadata["features"] == features
    assert metadata["preprocess"] is preprocess
    assert metadata["columns"] == (
        INT_COLS if features == "numeric" else INT_COLS + CAT_COLS
    )
    assert metadata["license"] == "CC-BY-NC-SA 4.0"
    assert metadata["source"].startswith("https://")


@pytest.mark.parametrize("features,preprocess", MODES)
def test_every_mode_describes_its_preprocessing(criteo_root, features, preprocess):
    description = load(criteo_root, features, preprocess).metadata["preprocessing"]
    assert description
    assert ("No fill" in description) is not preprocess


@pytest.mark.parametrize("features,preprocess", MODES)
def test_standardization_statistics_are_exposed_when_fitted(
    criteo_root, features, preprocess
):
    metadata = load(criteo_root, features, preprocess).metadata
    if not preprocess:
        assert "feature_means" not in metadata
        assert "feature_stds" not in metadata
    else:
        assert len(metadata["feature_means"]) == WIDTH[features]
        assert len(metadata["feature_stds"]) == WIDTH[features]


def test_category_counts_are_exposed_only_where_they_exist(criteo_root):
    assert "n_categories" not in load(criteo_root, "all", False).metadata
    assert "n_categories" not in load(criteo_root, "numeric", True).metadata
    counts = load(criteo_root, "all", True).metadata["n_categories"]
    assert len(counts) == len(CAT_COLS)
    assert all(n > 0 for n in counts)


def test_notice_goes_to_stderr_and_matches_the_metadata(criteo_root, capsys):
    split = load(criteo_root, "all", True)
    captured = capsys.readouterr()
    assert split.metadata["preprocessing"] in captured.err
    assert captured.out == ""


def test_notice_prints_once_per_mode(criteo_root, capsys):
    load(criteo_root, "numeric", True)
    capsys.readouterr()

    load(criteo_root, "numeric", True)
    assert capsys.readouterr().err == ""

    load(criteo_root, "numeric", False)
    assert capsys.readouterr().err != ""
