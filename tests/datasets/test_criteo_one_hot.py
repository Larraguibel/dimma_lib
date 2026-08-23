"""The ninth mode: one-hot, returned as index/value pairs.

These tests pin the entry count — exactly 39 per row, with distinct
indices, in both splits and whatever a row holds — the blocks those
indices fall in, the reserved unseen slot, the integer chain, the
metadata and the prose. They also pin the two claims that make the
representation worth having: that the pairs reconstruct the one-hot
matrix nobody built, and that `forward_sparse` returns what `forward`
would have returned on it. And the seam with `load_criteo`: the same
seed holds out the same rows. ADR-0020 records why each of these is
the thing to pin.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from dimma.datasets.criteo import (
    CAT_COLS,
    INT_COLS,
    load_criteo,
    load_criteo_one_hot,
)
from dimma.models.logreg import forward, forward_sparse, init_params

from .conftest import N_ROWS, TEST_FRACTION, UNSEEN_CATEGORY, split_indices

WIDTH = len(INT_COLS) + len(CAT_COLS)
N_INT = len(INT_COLS)


def load_one_hot(criteo_root, preprocess=True, **kwargs):
    return load_criteo_one_hot(
        preprocess=preprocess, root=criteo_root, download=False, **kwargs
    )


def densify(idx: np.ndarray, val: np.ndarray, num_features: int) -> np.ndarray:
    """The matrix the pairs stand for. Only ever built at fixture size."""
    dense = np.zeros((idx.shape[0], num_features), dtype=np.float32)
    rows = np.repeat(np.arange(idx.shape[0]), idx.shape[1])
    dense[rows, idx.ravel()] = val.ravel()
    return dense


# --------------------------------------------------------------------
# The invariant the representation exists for
# --------------------------------------------------------------------


@pytest.mark.parametrize("preprocess", [False, True])
def test_every_row_holds_exactly_thirty_nine_entries(criteo_root, preprocess):
    """s is a property of the encoding, not of the record: a row with a
    category nobody saw in training stores the same 39 as any other."""
    split = load_one_hot(criteo_root, preprocess)
    n_test = int(round(N_ROWS * TEST_FRACTION))
    assert split.idx_train.shape == (N_ROWS - n_test, WIDTH)
    assert split.val_train.shape == (N_ROWS - n_test, WIDTH)
    assert split.idx_test.shape == (n_test, WIDTH)
    assert split.val_test.shape == (n_test, WIDTH)


@pytest.mark.parametrize("part", ["train", "test"])
def test_the_indices_within_a_row_are_distinct(criteo_root, part):
    """Which is what makes a row's l2 norm its `val` vector's, and the
    count of stored entries the count of active coordinates."""
    split = load_one_hot(criteo_root)
    idx = np.asarray(getattr(split, f"idx_{part}"))
    for row in idx:
        assert len(set(row.tolist())) == WIDTH


@pytest.mark.parametrize("part", ["train", "test"])
def test_every_index_addresses_a_real_column(criteo_root, part):
    split = load_one_hot(criteo_root)
    idx = np.asarray(getattr(split, f"idx_{part}"))
    assert idx.min() >= 0
    assert idx.max() < split.num_features


@pytest.mark.parametrize("part", ["train", "test"])
def test_each_entry_falls_in_its_own_column_block(criteo_root, part):
    """A categorical column's index says which column it came from, so
    no two columns can share a weight."""
    split = load_one_hot(criteo_root)
    idx = np.asarray(getattr(split, f"idx_{part}"))
    offsets = split.metadata["column_offsets"]
    cardinalities = split.metadata["n_categories"]

    numeric = np.tile(np.arange(N_INT), (len(idx), 1))
    assert np.array_equal(idx[:, :N_INT], numeric)
    for j, (offset, cardinality) in enumerate(zip(offsets, cardinalities)):
        column = idx[:, N_INT + j]
        assert column.min() >= offset
        assert column.max() <= offset + cardinality


def test_num_features_is_the_blocks_end_to_end(criteo_root):
    split = load_one_hot(criteo_root)
    cardinalities = split.metadata["n_categories"]
    assert split.num_features == N_INT + sum(n + 1 for n in cardinalities)
    assert split.metadata["num_features"] == split.num_features


# --------------------------------------------------------------------
# The reserved unseen slot
# --------------------------------------------------------------------


def test_an_unseen_test_category_goes_to_its_columns_reserved_slot(
    criteo_root, criteo_frame
):
    _, test_idx = split_indices()
    assert criteo_frame.loc[test_idx[0], "C1"] == UNSEEN_CATEGORY

    split = load_one_hot(criteo_root)
    offset = split.metadata["column_offsets"][0]
    cardinality = split.metadata["n_categories"][0]
    assert int(np.asarray(split.idx_test)[0, N_INT]) == offset + cardinality


def test_the_reserved_slot_is_untouched_by_the_training_split(criteo_root):
    """So the weight on it never trains, which is the whole cost of
    keeping the entry count fixed."""
    split = load_one_hot(criteo_root)
    idx_train = np.asarray(split.idx_train)
    offsets = split.metadata["column_offsets"]
    cardinalities = split.metadata["n_categories"]
    for offset, cardinality in zip(offsets, cardinalities):
        assert offset + cardinality not in set(idx_train.ravel().tolist())


# --------------------------------------------------------------------
# What the pairs stand for
# --------------------------------------------------------------------


def test_the_pairs_reconstruct_the_one_hot_matrix(criteo_root, criteo_frame):
    """Against a get_dummies reference, on the only data small enough
    for the dense matrix to exist at all."""
    train_idx, _ = split_indices()
    train_df = criteo_frame.iloc[train_idx]
    split = load_one_hot(criteo_root, preprocess=False)

    blocks = [train_df[INT_COLS].to_numpy(dtype=np.float32)]
    for col in CAT_COLS:
        dummies = pd.get_dummies(train_df[col]).to_numpy(dtype=np.float32)
        unseen = np.zeros((len(train_df), 1), dtype=np.float32)
        blocks.append(np.concatenate([dummies, unseen], axis=1))
    expected = np.concatenate(blocks, axis=1)

    dense = densify(
        np.asarray(split.idx_train),
        np.asarray(split.val_train),
        split.num_features,
    )
    assert dense.shape == expected.shape
    assert np.array_equal(dense, expected, equal_nan=True)


def test_the_categorical_values_are_all_one(criteo_root):
    split = load_one_hot(criteo_root)
    assert np.array_equal(
        np.asarray(split.val_train)[:, N_INT:],
        np.ones((split.val_train.shape[0], len(CAT_COLS)), dtype=np.float32),
    )


def test_forward_sparse_is_forward_on_the_row_it_implies(criteo_root):
    """The reason nothing has to be densified: the same logit, computed
    from the pairs."""
    split = load_one_hot(criteo_root)
    params = init_params(jax.random.key(0), split.num_features)

    idx, val = split.idx_train[:16], split.val_train[:16]
    dense = jnp.asarray(densify(np.asarray(idx), np.asarray(val),
                                split.num_features))
    sparse_logits = jax.vmap(forward_sparse, in_axes=(None, 0, 0))(
        params, idx, val
    )
    dense_logits = jax.vmap(forward, in_axes=(None, 0))(params, dense)
    assert jnp.allclose(sparse_logits, dense_logits, atol=1e-5)


# --------------------------------------------------------------------
# The seam with `load_criteo`
# --------------------------------------------------------------------


def test_the_same_seed_holds_out_the_same_rows(criteo_root):
    """Otherwise a one-hot run and a frequency-encoded one are two
    benchmarks rather than one comparison."""
    sparse = load_one_hot(criteo_root)
    dense = load_criteo(
        features="all", preprocess=True, root=criteo_root, download=False
    )
    for a, b in ((sparse.y_train, dense.y_train),
                 (sparse.y_test, dense.y_test)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_the_seed_selects_the_split(criteo_root, criteo_frame):
    train_idx, _ = split_indices()
    expected = criteo_frame.iloc[train_idx]["label"].to_numpy(dtype=np.float32)
    labels = np.asarray(load_one_hot(criteo_root).y_train)
    assert np.array_equal(labels, expected)


def test_the_category_counts_agree_with_the_frequency_encoding(criteo_root):
    """Same vocabulary, fitted the same way on the same rows."""
    sparse = load_one_hot(criteo_root)
    dense = load_criteo(
        features="all", preprocess=True, root=criteo_root, download=False
    )
    assert sparse.metadata["n_categories"] == dense.metadata["n_categories"]


# --------------------------------------------------------------------
# `preprocess` governs the integer chain and nothing else
# --------------------------------------------------------------------


def test_preprocessing_applies_the_integer_chain(criteo_root):
    numeric = load_criteo(
        features="numeric", preprocess=True, root=criteo_root, download=False
    )
    values = np.asarray(load_one_hot(criteo_root, True).val_train)[:, :N_INT]
    assert np.allclose(values, np.asarray(numeric.x_train), atol=1e-6)


def test_not_preprocessing_returns_the_stored_values(criteo_root):
    numeric = load_criteo(
        features="numeric", preprocess=False, root=criteo_root, download=False
    )
    values = np.asarray(load_one_hot(criteo_root, False).val_train)[:, :N_INT]
    assert np.array_equal(values, np.asarray(numeric.x_train), equal_nan=True)
    assert np.isnan(values).any()


def test_the_categoricals_are_one_hot_either_way(criteo_root):
    """`preprocess` does not reach them: that is what the name is for."""
    on = np.asarray(load_one_hot(criteo_root, True).idx_train)
    off = np.asarray(load_one_hot(criteo_root, False).idx_train)
    assert np.array_equal(on[:, N_INT:], off[:, N_INT:])


# --------------------------------------------------------------------
# Types, metadata, and the printed notice
# --------------------------------------------------------------------


@pytest.mark.parametrize("preprocess", [False, True])
def test_indices_are_int32_and_everything_else_float32(
    criteo_root, preprocess
):
    split = load_one_hot(criteo_root, preprocess)
    assert split.idx_train.dtype == np.int32
    assert split.idx_test.dtype == np.int32
    for array in (split.val_train, split.y_train, split.val_test,
                  split.y_test):
        assert array.dtype == np.float32


def test_metadata_records_what_the_loader_did(criteo_root):
    metadata = load_one_hot(criteo_root).metadata
    assert metadata["encoding"] == "one_hot"
    assert metadata["preprocess"] is True
    assert metadata["columns"] == INT_COLS + CAT_COLS
    assert metadata["int_cols"] == INT_COLS
    assert metadata["cat_cols"] == CAT_COLS
    assert len(metadata["n_categories"]) == len(CAT_COLS)
    assert len(metadata["column_offsets"]) == len(CAT_COLS)
    assert metadata["license"] == "CC-BY-NC-SA 4.0"
    assert metadata["source"].startswith("https://")


@pytest.mark.parametrize("preprocess", [False, True])
def test_the_chain_is_described_in_prose(criteo_root, preprocess):
    split = load_one_hot(criteo_root, preprocess)
    description = split.metadata["preprocessing"]
    assert "one-hot" in description
    assert "unseen" in description
    assert "39" in description
    assert ("no log1p" in description) is not preprocess


def test_the_one_hot_prose_is_nobody_elses(criteo_root):
    """ADR-0008 again: a mode described by another mode's chain."""
    modes = [
        load_criteo(features=features, preprocess=preprocess,
                    root=criteo_root, download=False).metadata["preprocessing"]
        for features in ("numeric", "all")
        for preprocess in (False, True)
    ]
    for preprocess in (False, True):
        split = load_one_hot(criteo_root, preprocess)
        assert split.metadata["preprocessing"] not in modes


def test_the_notice_goes_to_stderr_once_per_mode(criteo_root, capsys):
    split = load_one_hot(criteo_root)
    assert split.metadata["preprocessing"] in capsys.readouterr().err

    load_one_hot(criteo_root)
    assert capsys.readouterr().err == ""

    load_one_hot(criteo_root, preprocess=False)
    assert capsys.readouterr().err != ""


# --------------------------------------------------------------------
# The feature-norm bound (ADR-0012)
# --------------------------------------------------------------------


def test_no_bound_is_the_default_and_leaves_no_trace(criteo_root):
    assert "feature_norm_bound" not in load_one_hot(criteo_root).metadata


def test_the_bound_is_enforced_on_both_splits(criteo_root):
    """On `val`, which is where a sparse row's norm lives."""
    split = load_one_hot(criteo_root, feature_norm_bound=1.0)
    for val in (split.val_train, split.val_test):
        norms = np.linalg.norm(np.asarray(val), axis=1)
        assert np.all(norms <= 1.0 + 1e-5)


def test_the_bound_reaches_the_metadata_and_the_prose(criteo_root):
    split = load_one_hot(criteo_root, feature_norm_bound=0.5)
    assert split.metadata["feature_norm_bound"] == 0.5
    assert "0.5" in split.metadata["preprocessing"]


def test_the_bound_leaves_the_indices_alone(criteo_root):
    """It rescales what is stored, not where it is stored."""
    bounded = load_one_hot(criteo_root, feature_norm_bound=1.0)
    plain = np.asarray(load_one_hot(criteo_root).idx_train)
    capped = np.asarray(bounded.idx_train)
    assert np.array_equal(plain, capped)


def test_a_bound_that_bounds_nothing_is_rejected(criteo_root):
    with pytest.raises(ValueError):
        load_one_hot(criteo_root, feature_norm_bound=0.0)


def test_capping_unpreprocessed_data_says_it_could_not(criteo_root):
    """The stored integers keep their NaN, and a NaN row is not bounded."""
    with pytest.warns(UserWarning, match="finite"):
        load_one_hot(criteo_root, preprocess=False, feature_norm_bound=1.0)


def test_missing_file_without_download_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_criteo_one_hot(root=tmp_path, download=False)
