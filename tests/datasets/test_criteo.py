"""The eight modes, and the lines between them.

Three claims worth more than the shapes: that ``preprocess`` is honest
in both directions, that ``standardize`` is a separate axis, and that
every fitted statistic comes from the training split alone.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dimma.datasets.criteo import (
    CAT_COLS,
    INT_COLS,
    _frequency_encode,
    load_criteo,
)
from dimma.datasets.preprocessing import cap_feature_norms

from .conftest import N_ROWS, TEST_FRACTION, UNSEEN_CATEGORY, split_indices

#: Every combination of the three axes, so the invariants that hold
#: across all of them are asserted across all of them.
MODES = [
    (features, preprocess, standardize)
    for features in ("numeric", "all")
    for preprocess in (False, True)
    for standardize in (False, True)
]
CHAINS = [("numeric", False), ("numeric", True), ("all", False), ("all", True)]
WIDTH = {"numeric": 13, "all": 39}


def load(criteo_root, features, preprocess, standardize=False):
    return load_criteo(
        features=features,
        preprocess=preprocess,
        standardize=standardize,
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


def test_default_mode_is_numeric_preprocessed_and_unstandardized(criteo_root):
    """The last of the three is the one that would be assumed wrongly:
    standardization is a fitted map no budget charges for, so a caller
    who did not ask for it does not get it."""
    split = load_criteo(root=criteo_root, download=False)
    assert split.metadata["features"] == "numeric"
    assert split.metadata["preprocess"] is True
    assert split.metadata["standardize"] is False


# --------------------------------------------------------------------
# Shape and split
# --------------------------------------------------------------------


@pytest.mark.parametrize("features,preprocess,standardize", MODES)
def test_column_count(criteo_root, features, preprocess, standardize):
    split = load(criteo_root, features, preprocess, standardize)
    assert split.x_train.shape[1] == WIDTH[features]
    assert split.x_test.shape[1] == WIDTH[features]


@pytest.mark.parametrize("features,preprocess,standardize", MODES)
def test_row_counts_follow_test_fraction(
    criteo_root, features, preprocess, standardize
):
    split = load(criteo_root, features, preprocess, standardize)
    n_test = int(round(N_ROWS * TEST_FRACTION))
    assert split.x_train.shape[0] == split.y_train.shape[0] == N_ROWS - n_test
    assert split.x_test.shape[0] == split.y_test.shape[0] == n_test


@pytest.mark.parametrize("features,preprocess,standardize", MODES)
def test_everything_is_float32(criteo_root, features, preprocess, standardize):
    split = load(criteo_root, features, preprocess, standardize)
    for array in (split.x_train, split.y_train, split.x_test, split.y_test):
        assert array.dtype == np.float32


def test_all_eight_modes_split_the_same_rows(criteo_root):
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


def test_integer_chain_is_median_fill_clip_log1p(criteo_root, criteo_frame):
    """And nothing else: with `standardize=False` the chain ends here."""
    train_idx, _ = split_indices()
    split = load(criteo_root, "numeric", True)

    raw = criteo_frame.iloc[train_idx][INT_COLS]
    expected = raw.fillna(raw.median(numeric_only=True)).to_numpy(np.float32)
    expected = np.log1p(np.clip(expected, 0.0, None))

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
# standardize is its own axis (ADR-0013)
# --------------------------------------------------------------------


@pytest.mark.parametrize("features", ["numeric", "all"])
def test_standardizing_centres_and_scales_the_train_split(
    criteo_root, features
):
    x = np.asarray(load(criteo_root, features, True, True).x_train)
    assert np.allclose(x.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(x.std(axis=0), 1.0, atol=1e-5)


@pytest.mark.parametrize("features,preprocess", CHAINS)
def test_not_standardizing_leaves_the_columns_where_they_were(
    criteo_root, features, preprocess
):
    """The default must not centre anything behind the caller's back."""
    x = np.asarray(load(criteo_root, features, preprocess).x_train)
    assert not np.allclose(np.nanmean(x, axis=0), 0.0, atol=1e-3)


@pytest.mark.parametrize("features", ["numeric", "all"])
def test_standardizing_composes_onto_the_chain_it_follows(
    criteo_root, features
):
    """`standardize` is applied to whatever preceded it rather than
    selecting a different chain."""
    plain = np.asarray(load(criteo_root, features, True).x_train)
    split = load(criteo_root, features, True, True)
    means = split.metadata["feature_means"]
    stds = split.metadata["feature_stds"]

    expected = (plain - means) / stds
    assert np.allclose(np.asarray(split.x_train), expected, atol=1e-5)


def test_standardizing_does_not_require_preprocessing(criteo_root):
    """The two axes are independent: the stored values can be
    standardized without the fill, the clip, the log1p or the encoding.
    Checked on the categorical block, the 26 of the 39 stored columns
    that the fixture leaves free of NaN."""
    x = np.asarray(load(criteo_root, "all", False, True).x_train)
    categorical = x[:, len(INT_COLS):]
    assert np.allclose(categorical.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(categorical.std(axis=0), 1.0, atol=1e-5)


def test_standardizing_stored_values_propagates_their_nan(criteo_root):
    """A fitted mean reads the whole column, so one stored NaN takes the
    whole column with it — the fill that would have removed it belongs to
    `preprocess`, which this caller declined. The pinned file carries no
    NaN; a caller who supplies their own file may, and imputing behind
    their back is not the loader's to do."""
    x = np.asarray(load(criteo_root, "numeric", False, True).x_train)
    nan_columns = np.isnan(x).any(axis=0)
    assert nan_columns.any()
    assert np.isnan(x[:, nan_columns]).all()


@pytest.mark.parametrize("features", ["numeric", "all"])
def test_test_split_uses_the_train_statistics(criteo_root, features):
    """Standardizing the test split by its own moments would leak it."""
    x = np.asarray(load(criteo_root, features, True, True).x_test)
    assert not np.allclose(x.mean(axis=0), 0.0, atol=1e-6)


def test_standardizing_pulls_the_row_norms_towards_sqrt_d(criteo_root):
    """Which is the reason the default is off, though the direction it
    moves them depends on the file. d columns each rescaled to unit
    variance put a typical row near sqrt(d) whatever it was before, so
    the norms a `feature_norm_bound` has to bound afterwards are a
    property of the standardization and not of the data. What that costs
    on the pinned file — where they move *up* — is in `test_criteo_real`;
    this fixture's columns are 0-500 rather than [0, 1], so here the same
    step moves them down to the same place."""
    scaled = np.asarray(load(criteo_root, "numeric", True, True).x_train)
    assert np.median(np.linalg.norm(scaled, axis=1)) == pytest.approx(
        np.sqrt(13), rel=0.2
    )


# --------------------------------------------------------------------
# Metadata and the printed notice
# --------------------------------------------------------------------


@pytest.mark.parametrize("features,preprocess,standardize", MODES)
def test_metadata_always_records_the_mode(
    criteo_root, features, preprocess, standardize
):
    metadata = load(criteo_root, features, preprocess, standardize).metadata
    assert metadata["features"] == features
    assert metadata["preprocess"] is preprocess
    assert metadata["standardize"] is standardize
    assert metadata["columns"] == (
        INT_COLS if features == "numeric" else INT_COLS + CAT_COLS
    )
    assert metadata["license"] == "CC-BY-NC-SA 4.0"
    assert metadata["source"].startswith("https://")


@pytest.mark.parametrize("features,preprocess,standardize", MODES)
def test_every_mode_describes_its_preprocessing(
    criteo_root, features, preprocess, standardize
):
    description = load(
        criteo_root, features, preprocess, standardize
    ).metadata["preprocessing"]
    assert description
    assert ("No fill" in description) is not preprocess
    assert ("Not standardized" in description) is not standardize


def test_the_eight_modes_are_eight_descriptions(criteo_root):
    """A mode described by another mode's prose is the failure ADR-0008
    exists to prevent, and the third axis must not reintroduce it."""
    described = {
        load(criteo_root, *mode).metadata["preprocessing"] for mode in MODES
    }
    assert len(described) == len(MODES)


@pytest.mark.parametrize("features,preprocess,standardize", MODES)
def test_standardization_statistics_are_exposed_when_fitted(
    criteo_root, features, preprocess, standardize
):
    """They track `standardize` alone — `preprocess` fits medians and
    frequencies, and neither of those is a mean or a deviation."""
    metadata = load(criteo_root, features, preprocess, standardize).metadata
    if not standardize:
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


# --------------------------------------------------------------------
# The feature-norm bound (ADR-0012)
# --------------------------------------------------------------------


@pytest.mark.parametrize("features,preprocess,standardize", MODES)
def test_no_bound_is_the_default_and_leaves_no_trace(
    criteo_root, features, preprocess, standardize
):
    assert "feature_norm_bound" not in load(
        criteo_root, features, preprocess, standardize).metadata


def test_the_bound_is_enforced_on_both_splits(criteo_root):
    split = load_criteo(root=criteo_root, download=False,
                        feature_norm_bound=1.0)
    for x in (split.x_train, split.x_test):
        norms = np.linalg.norm(np.asarray(x), axis=1)
        assert np.all(norms <= 1.0 + 1e-5)


def test_the_bound_reaches_the_metadata(criteo_root):
    """So it can be handed to an accountant without being typed twice."""
    split = load_criteo(root=criteo_root, download=False,
                        feature_norm_bound=0.5)
    assert split.metadata["feature_norm_bound"] == 0.5


@pytest.mark.parametrize("features,preprocess,standardize", MODES)
def test_the_bound_is_recorded_whatever_the_other_axes_are(
    criteo_root, features, preprocess, standardize
):
    """It is not conditional on any of the other three, which is what
    the docstring promises."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        split = load_criteo(features=features, preprocess=preprocess,
                            standardize=standardize,
                            root=criteo_root, download=False,
                            feature_norm_bound=1.0)
    assert split.metadata["feature_norm_bound"] == 1.0


def test_the_cap_goes_after_the_fitted_maps(criteo_root):
    """Capping first and standardizing after would leave the largest
    norm far above R, and the accountant would never know."""
    uncapped = np.asarray(load(criteo_root, "numeric", True, True).x_train)
    expected, _ = cap_feature_norms(uncapped, 1.0)

    capped = load_criteo(root=criteo_root, download=False, standardize=True,
                         feature_norm_bound=1.0).x_train
    assert np.allclose(np.asarray(capped), expected, atol=1e-6)


def test_capping_before_standardizing_would_not_have_bounded_anything(
    criteo_root,
):
    """The ordering is load-bearing, not stylistic: this is what the
    other order — cap the preprocessed features, standardize after —
    would have produced."""
    preprocessed = np.asarray(load(criteo_root, "numeric", True).x_train)
    wrong_order, _ = cap_feature_norms(preprocessed, 1.0)
    restandardized = (wrong_order - wrong_order.mean(0)) / wrong_order.std(0)
    # Not marginally over: thirteen columns rescaled independently land
    # near sqrt(13), so the accountant would be handed a constant about
    # a quarter of the truth.
    assert np.linalg.norm(restandardized, axis=1).max() > 3.0


def test_the_bound_is_in_the_recorded_preprocessing(criteo_root):
    metadata = load_criteo(root=criteo_root, download=False,
                           feature_norm_bound=1.0).metadata
    assert "1" in metadata["preprocessing"]
    assert "norm" in metadata["preprocessing"]


def test_a_different_bound_is_a_different_notice(criteo_root, capsys):
    """The printed line has to track the bound, or the second run is
    described by the first run's chain."""
    load_criteo(root=criteo_root, download=False, feature_norm_bound=1.0)
    capsys.readouterr()
    load_criteo(root=criteo_root, download=False, feature_norm_bound=2.0)
    assert capsys.readouterr().err != ""


def test_a_bound_that_bounds_nothing_is_rejected(criteo_root):
    with pytest.raises(ValueError):
        load_criteo(root=criteo_root, download=False, feature_norm_bound=0.0)


def test_capping_unpreprocessed_data_says_it_could_not(criteo_root):
    """The raw mode keeps its NaN, and a NaN row is not bounded."""
    with pytest.warns(UserWarning, match="finite"):
        load_criteo(root=criteo_root, download=False, preprocess=False,
                    feature_norm_bound=1.0)


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


def test_flipping_standardize_is_a_new_notice(criteo_root, capsys):
    """Otherwise the second run is described by the first run's chain."""
    load(criteo_root, "numeric", True)
    capsys.readouterr()
    load(criteo_root, "numeric", True, True)
    assert capsys.readouterr().err != ""
