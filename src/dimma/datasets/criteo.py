"""The Criteo 1M sample: a click-prediction benchmark, nine ways.

`load_criteo` has three independent choices, so eight modes. *Which
columns* — the 13 integer features, or all 39 including the 26
categorical ones. *Whether they are preprocessed* — the stored values
as they come, or the fill, clip, log1p and frequency encoding described
in ``metadata["preprocessing"]``. *Whether they are standardized* —
column means and standard deviations fitted on the training split, or
the scales the columns already had. Stored is not raw; see below.

All three are parameters rather than one mode name, because a mode name
is where the interesting part goes missing: the earlier version of this
loader called its preprocessed mode ``"integer"``, which says which
columns it returns and nothing about the log1p and standardization it
also applies. Here each is unmissable at the call site, and whatever
they did is recorded in the returned metadata and printed once.

``standardize`` is its own axis, and off by default. It sets every
column's scale to 1, so it puts a typical row's ℓ₂ norm near
``sqrt(d)`` whatever it was before — and on this file that is a move
upwards: the largest norm over the numeric chain goes from about 2 to
about 15. That number is what a ``feature_norm_bound`` afterwards has to
bound, and per ADR-0012 the constants derived from it grow like ``R``
and ``R²``. Standardizing therefore buys conditioning with roughly fifty
times the noise, which ADR-0013 records as a trade the caller makes
explicitly rather than one a default makes for them.

Every statistic — medians, category frequencies, means, standard
deviations — is fitted on the training split alone and applied to the
test split, so the test set leaks nothing into the features. Note that
this is not itself a private operation: the statistics depend on the
training data and are not accounted for in any privacy budget. That is
the standard benchmark convention, not a claim about the pipeline.

What the pinned file actually holds
-----------------------------------
Not raw Criteo. ``I1..I13`` arrive already scaled into ``[0, 1]``
against a per-column cap — the distinct values of ``I1`` are the 21
multiples of 0.05, of ``I3`` the 101 multiples of 0.01, of ``I6`` the
501 multiples of 0.002 — and there is no NaN anywhere, in the integer
block or the categorical one, and no negative value. The upstream
sample documents none of this and ``dimma`` pins it only by SHA256, so
the scaling is a property of the file rather than a step this loader
can point at. Getting raw integer-with-NaN semantics would mean
re-deriving the sample from the original Criteo dump.

Two things follow for ``preprocess=True``. The median fill has nothing
to fill and the clip at 0 has nothing to clip; both are no-ops here,
kept because they are what the chain owes raw Criteo. And ``log1p``
runs on values already in ``[0, 1]``, so it compresses them into
``[0, log 2]`` monotonically rather than taming a long tail. So on this
file ``preprocess=True`` alone is close to a no-op for the integer
block, and ``standardize=True`` is the step that does what its name
suggests — which is the other half of why the two are separate axes.

The ninth mode
--------------
``load_criteo_one_hot`` is a separate function, not a third value on
the ``features`` axis. It one-hot encodes ``C1..C26`` at their native
train-split cardinalities — 552k columns at the default split, out of
the 623k distinct IDs the whole file carries — and hands back the
indices each row occupies and the values it puts there rather than a
matrix, because the dense one would be 2.2 TB. ADR-0020 records why it
is its own function, and what the exact entry count per row is for.

License
-------
The Criteo data is CC-BY-NC-SA 4.0. ``dimma`` is not; only the data
carries the restriction, and the library does not enforce it. An
attribution notice prints to stderr once per process on first download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from dimma.datasets._attribution import emit_once
from dimma.datasets._cache import get_cache_dir
from dimma.datasets._download import download_with_checksum
from dimma.datasets.base import (
    SparseTabularSplit,
    TabularSplit,
    arrays_to_sparse_split,
    arrays_to_split,
)
from dimma.datasets.preprocessing import cap_feature_norms

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover - exercised by the install, not tests
    raise ImportError(
        "dimma.datasets.criteo needs pandas and pyarrow, which are not part of "
        "the base install. Get them with: pip install 'dimma[datasets]'"
    ) from e


_URL = (
    "https://huggingface.co/datasets/eldieguinpo/criteo-1M/"
    "resolve/main/criteo_1M.parquet"
)
_SHA256 = "0b468148aecf6fa9464def4f2ad075b8843874f3469239afd3504602579f767a"
_FILENAME = "criteo_1M.parquet"

INT_COLS: list[str] = [f"I{i}" for i in range(1, 14)]
CAT_COLS: list[str] = [f"C{i}" for i in range(1, 27)]
LABEL_COL = "label"

_LICENSE_NOTICE = (
    "Downloaded Criteo 1M sample (CC-BY-NC-SA 4.0).\n"
    "Original data: Criteo Labs (https://ailab.criteo.com).\n"
    "Non-commercial use only. "
    "Derivative works must be shared under the same license."
)

_DESCRIPTIONS = {
    ("numeric", False): (
        "13 integer features I1-I13 exactly as stored, cast to float32 — "
        "already scaled into [0, 1] upstream, not raw Criteo integers, and "
        "with no NaN to preserve. No fill, no log1p."
    ),
    ("numeric", True): (
        "13 integer features I1-I13: NaN filled with the train-split median, "
        "clipped below at 0, then log1p. The stored values carry no NaN and "
        "no negatives, so the fill and the clip are no-ops here and log1p "
        "acts on [0, 1]."
    ),
    ("all", False): (
        "39 features I1-I13 and C1-C26 exactly as stored, cast to float32. "
        "I* are already scaled into [0, 1] upstream, not raw Criteo integers; "
        "C* are integer category IDs, which float32 represents exactly below "
        "2**24. No NaN in either block. No fill, no log1p, no encoding."
    ),
    ("one_hot", False): (
        "39 features as index/value pairs. I1-I13 exactly as stored, cast "
        "to float32, at indices 0-12 — already scaled into [0, 1] upstream, "
        "not raw Criteo integers, and with no NaN to preserve; no fill, no "
        "log1p."
    ),
    ("one_hot", True): (
        "39 features as index/value pairs. I1-I13 at indices 0-12: NaN filled "
        "with the train-split median, clipped below at 0, then log1p — the "
        "fill and the clip are no-ops on stored values that carry neither NaN "
        "nor negatives, and log1p acts on [0, 1]."
    ),
    ("all", True): (
        "39 features. I1-I13: NaN filled with the train-split median, clipped "
        "below at 0, log1p — the fill and the clip are no-ops on stored values "
        "that carry neither NaN nor negatives, and log1p acts on [0, 1]. "
        "C1-C26: each category ID replaced by that category's relative "
        "frequency in the train split, with categories unseen in training "
        "encoded as 0.0."
    ),
}

#: Appended to the sentence above. Its own clause because `standardize`
#: is its own axis: a chain that standardized and one that did not must
#: never be described by the same prose, whichever way the default falls.
_STANDARDIZATION = {
    True: (
        " Every column then standardized by the train-split mean and "
        "standard deviation."
    ),
    False: (
        " Not standardized: the columns keep the scales above, and no "
        "mean or standard deviation was fitted."
    ),
}

#: Appended to whichever one-hot entry above `preprocess` chose. Written
#: once because `preprocess` governs the integer chain alone: the
#: categorical half of the sentence is the same either way.
_ONE_HOT_CATEGORICAL = (
    " C1-C26 one-hot at their native train-split cardinalities, each "
    "column holding its own block of the index space and one slot "
    "reserved for IDs unseen in the training split, so every row carries "
    "exactly 39 entries whatever it holds. Not standardized: centring a "
    "one-hot column fills it, and the sparsity is the point."
)


def _read_frame(
    columns: list[str], root: str | Path | None, download: bool
) -> pd.DataFrame:
    """Find or fetch the cached parquet and read ``columns`` and the label."""
    if root is None:
        root = get_cache_dir("datasets")
    else:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
    parquet_path = root / _FILENAME

    if not parquet_path.exists():
        if not download:
            raise FileNotFoundError(
                f"Criteo parquet not found at {parquet_path} and "
                f"download=False. Pass download=True, or put the file there "
                f"yourself."
            )
        download_with_checksum(_URL, parquet_path, _SHA256)
        emit_once("criteo:license", _LICENSE_NOTICE)

    return pd.read_parquet(parquet_path, columns=columns + [LABEL_COL])


def _split_frame(
    df: pd.DataFrame, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut the frame in two by a seeded permutation of its rows.

    Every loader in this module splits here, which is the shared-split
    guarantee ADR-0020 records.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_test = int(round(len(df) * test_fraction))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return df.iloc[train_idx], df.iloc[test_idx]


def _norm_bound_clause(feature_norm_bound: float | None) -> str:
    """The sentence a cap adds to the chain's prose, or nothing."""
    if feature_norm_bound is None:
        return ""
    return (
        f" Every row then rescaled to l2 norm at most "
        f"{feature_norm_bound}, one record at a time."
    )


def _locate_in_vocabulary(
    vocabulary: np.ndarray, codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Where each code sits in the sorted ``vocabulary``, and whether it is
    there at all — an absent code's position is clipped into range, so the
    mask is what says it means nothing."""
    pos = np.searchsorted(vocabulary, codes)
    pos_safe = np.clip(pos, 0, vocabulary.shape[0] - 1)
    return pos_safe, vocabulary[pos_safe] == codes


def _frequency_encode(
    train_codes: np.ndarray, codes: np.ndarray
) -> tuple[np.ndarray, int]:
    """Replace each category ID by its relative frequency in ``train_codes``.

    One float per column instead of one per category: 26 columns whose
    cardinalities reach ~10^5 would otherwise become ~640k one-hot
    columns, and the hashed model that consumed the raw IDs is not part
    of this library. Categories absent from the training split encode as
    0.0, which is the frequency they were observed with.
    """
    vocabulary, counts = np.unique(train_codes, return_counts=True)
    freq = counts / train_codes.shape[0]

    pos_safe, seen = _locate_in_vocabulary(vocabulary, codes)
    encoded = np.where(seen, freq[pos_safe], 0.0).astype(np.float32)
    return encoded, vocabulary.shape[0]


def _one_hot_column(
    vocabulary: np.ndarray, codes: np.ndarray, offset: int
) -> np.ndarray:
    """Send each category ID to its index in this column's block.

    ``vocabulary`` is the sorted distinct IDs of the training split, the
    same fitted-on-train convention `_frequency_encode` uses, so local ID
    ``l`` sits at ``offset + l`` and an ID the training split never saw
    goes to the slot reserved at ``offset + cardinality``; ADR-0020
    records what that slot is for.
    """
    pos_safe, seen = _locate_in_vocabulary(vocabulary, codes)
    local = np.where(seen, pos_safe, vocabulary.shape[0])
    return (offset + local).astype(np.int32)


def _prepare_integers(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Median-fill, clip at 0, log1p. Statistics from the train split only.

    The first two are no-ops on the pinned file, which stores I1-I13
    already scaled into ``[0, 1]`` with no NaN and no negatives. They
    stay because they are what raw Criteo integers would need, and
    dropping them would make this chain wrong for any file but the one
    currently pinned.
    """
    medians = train_df.median(numeric_only=True)
    train = train_df.fillna(medians).to_numpy(dtype=np.float32)
    test = test_df.fillna(medians).to_numpy(dtype=np.float32)
    return (
        np.log1p(np.clip(train, a_min=0.0, a_max=None)),
        np.log1p(np.clip(test, a_min=0.0, a_max=None)),
    )


def _standardize(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Centre and scale by the train-split statistics.

    A column with no variance is left alone rather than divided by
    something near zero.
    """
    means = train.mean(axis=0)
    stds = train.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return (train - means) / stds, (test - means) / stds, means, stds


def load_criteo(
    features: Literal["numeric", "all"] = "numeric",
    preprocess: bool = True,
    standardize: bool = False,
    root: str | Path | None = None,
    download: bool = True,
    test_fraction: float = 0.2,
    seed: int = 0,
    device: str = "cpu",
    feature_norm_bound: float | None = None,
) -> TabularSplit:
    """Load the Criteo 1M sample as a train/test split.

    On first call, downloads ``criteo_1M.parquet`` (~45 MB) from
    https://huggingface.co/datasets/eldieguinpo/criteo-1M into the cache
    directory and verifies its SHA256. Later calls reuse the cached file.

    Parameters
    ----------
    features : {"numeric", "all"}, default "numeric"
        ``"numeric"`` returns the 13 integer features ``I1..I13``, which
        the pinned file stores already scaled into ``[0, 1]`` rather
        than as raw Criteo integers. ``"all"`` returns those plus the 26
        categorical features ``C1..C26``, which are integer category IDs
        in the source file.
    preprocess : bool, default True
        Whether to apply the column chain recorded in
        ``metadata["preprocessing"]`` and printed once per process:
        median fill, clip at 0 and ``log1p`` on ``I1..I13``, and
        frequency encoding on ``C1..C26``. With ``False`` the stored
        values are returned untouched beyond a cast to float32, and the
        caller owns everything downstream. Untouched is not raw: see the
        module docstring for what the pinned file has already had done
        to it. This axis does not standardize; ``standardize`` does.
    standardize : bool, default False
        Whether to centre and scale every column by the train-split mean
        and standard deviation, last before any ``feature_norm_bound``.
        Independent of ``preprocess``, so either chain can be
        standardized or not. Off by default: it is the step that pushes
        row norms towards ``sqrt(d)``, and the cost of the wider ``R``
        that then has to bound them is quadratic in the noise — ADR-0013
        records the trade, ADR-0012 the constants. With ``True`` the
        fitted arrays come back as ``metadata["feature_means"]`` and
        ``metadata["feature_stds"]``; with ``False`` neither key exists.
    root : str | Path | None, default None
        Cache directory. ``None`` uses ``get_cache_dir("datasets")``.
    download : bool, default True
        With ``False``, a missing file raises instead of being fetched.
    test_fraction : float, default 0.2
        Fraction of rows held out for testing.
    seed : int, default 0
        Seed for the train/test permutation. Two calls with the same
        seed and fraction split identically, whatever the other
        arguments — so the four modes are directly comparable.
    device : str, default "cpu"
        Target JAX device: ``"cpu"``, ``"gpu"``, or ``"cuda"``.
    feature_norm_bound : float | None, default None
        With a value, every row is rescaled to ``l_2`` norm at most that
        much, last in the chain and one record at a time, and the bound
        is recorded in ``metadata["feature_norm_bound"]``. It is what
        `dimma.accounting.lipschitz` turns into a Lipschitz constant, so
        it must be chosen before the data is looked at rather than read
        off it; ADR-0012 records why, and what a wider one costs. With
        ``None`` no cap is applied and no constant is implied.

    Returns
    -------
    TabularSplit
        ``metadata`` always carries ``"features"``, ``"preprocess"``,
        ``"standardize"``, ``"preprocessing"`` (the chain in prose),
        ``"columns"``, ``"license"``, and ``"source"``. With
        ``standardize=True`` it also carries ``"feature_means"`` and
        ``"feature_stds"``, the arrays the columns were standardized by;
        with ``features="all"`` it carries ``"int_cols"`` and
        ``"cat_cols"``, and when ``preprocess=True`` as well,
        ``"n_categories"`` — the distinct IDs each ``C*`` column had in
        the training split, before frequency encoding collapsed it to one
        float. Whatever the three axes are, a ``feature_norm_bound`` that
        was enforced is carried back under that name.

    Raises
    ------
    ValueError
        If ``features`` is not ``"numeric"`` or ``"all"``, or if
        ``feature_norm_bound`` is not finite and positive.
    FileNotFoundError
        If ``download=False`` and the file is not in the cache.
    RuntimeError
        If a downloaded file's SHA256 does not match the pinned digest.

    Notes
    -----
    The fitted statistics come from the training split only, but they are
    not privatized: they are ordinary data-dependent preprocessing, and
    a run that accounts for its own gradient noise has still not
    accounted for them. Standard for this benchmark, and worth stating
    rather than assuming.
    """
    if features not in ("numeric", "all"):
        raise ValueError(
            f"Unknown features {features!r}. Expected 'numeric' or 'all'."
        )

    columns = INT_COLS if features == "numeric" else INT_COLS + CAT_COLS
    df = _read_frame(columns, root, download)
    train_df, test_df = _split_frame(df, test_fraction, seed)

    y_train = train_df[LABEL_COL].to_numpy(dtype=np.float32)
    y_test = test_df[LABEL_COL].to_numpy(dtype=np.float32)

    description = (
        _DESCRIPTIONS[(features, preprocess)]
        + _STANDARDIZATION[standardize]
        + _norm_bound_clause(feature_norm_bound)
    )
    metadata: dict = {
        "features": features,
        "preprocess": preprocess,
        "standardize": standardize,
        "preprocessing": description,
        "columns": list(columns),
        "license": "CC-BY-NC-SA 4.0",
        "source": _URL,
    }
    if features == "all":
        metadata["int_cols"] = list(INT_COLS)
        metadata["cat_cols"] = list(CAT_COLS)

    if not preprocess:
        x_train = train_df[columns].to_numpy(dtype=np.float32)
        x_test = test_df[columns].to_numpy(dtype=np.float32)
    else:
        x_train, x_test = _prepare_integers(
            train_df[INT_COLS], test_df[INT_COLS]
        )
        if features == "all":
            blocks_train, blocks_test, cardinalities = [x_train], [x_test], []
            for col in CAT_COLS:
                train_codes = train_df[col].to_numpy()
                enc_train, n_seen = _frequency_encode(train_codes, train_codes)
                enc_test, _ = _frequency_encode(
                    train_codes, test_df[col].to_numpy()
                )
                blocks_train.append(enc_train[:, None])
                blocks_test.append(enc_test[:, None])
                cardinalities.append(n_seen)
            x_train = np.concatenate(blocks_train, axis=1)
            x_test = np.concatenate(blocks_test, axis=1)
            metadata["n_categories"] = cardinalities

    if standardize:
        x_train, x_test, means, stds = _standardize(x_train, x_test)
        metadata["feature_means"] = means
        metadata["feature_stds"] = stds

    if feature_norm_bound is not None:
        # Last, after every fitted map. ADR-0012 records why the order
        # is load-bearing rather than incidental.
        x_train, bound = cap_feature_norms(x_train, feature_norm_bound)
        x_test, _ = cap_feature_norms(x_test, feature_norm_bound)
        metadata["feature_norm_bound"] = bound

    emit_once(
        f"criteo:{features}:{preprocess}:{standardize}:{feature_norm_bound}",
        f"criteo: {description}",
    )

    return arrays_to_split(
        x_train, y_train, x_test, y_test, device=device, metadata=metadata
    )


def load_criteo_one_hot(
    preprocess: bool = True,
    root: str | Path | None = None,
    download: bool = True,
    test_fraction: float = 0.2,
    seed: int = 0,
    device: str = "cpu",
    feature_norm_bound: float | None = None,
) -> SparseTabularSplit:
    """Load the Criteo 1M sample one-hot encoded, as a sparse split.

    All 39 features, with ``C1..C26`` one-hot at the cardinalities they
    have in the training split: 551,947 columns on the real file at the
    default split. The dense matrix would be 2.2 TB in float32, so a row
    comes back as the 39 indices it occupies and the 39 values it puts
    there, and the dense one is never built.
    `dimma.models.logreg.forward_sparse` consumes the pair directly.

    A separate function rather than a third value on `load_criteo`'s
    ``features`` axis, because the return type is what changes:
    `SparseTabularSplit`, not `TabularSplit`. Everything else stays a
    parameter and is recorded in metadata, as ADR-0008 requires;
    ADR-0020 records why the encoding is the one thing that is not.

    On first call, downloads ``criteo_1M.parquet`` (~45 MB) from
    https://huggingface.co/datasets/eldieguinpo/criteo-1M into the cache
    directory and verifies its SHA256. Later calls reuse the cached file.

    Parameters
    ----------
    preprocess : bool, default True
        Governs the integer chain alone: median fill, clip at 0 and
        ``log1p`` on ``I1..I13``, as in `load_criteo` and recorded in
        ``metadata["preprocessing"]``. With ``False`` the stored values
        go into ``val`` cast to float32 and nothing else. It does not
        reach ``C1..C26``, which are one-hot either way — that is what
        this function is named for, and an axis that could turn it off
        would be a different loader.
    root : str | Path | None, default None
        Cache directory. ``None`` uses ``get_cache_dir("datasets")``.
    download : bool, default True
        With ``False``, a missing file raises instead of being fetched.
    test_fraction : float, default 0.2
        Fraction of rows held out for testing.
    seed : int, default 0
        Seed for the train/test permutation. The same seed and fraction
        hold out the same rows here as in `load_criteo`, so a one-hot
        run is comparable with the other eight modes rather than merely
        run on the same file.
    device : str, default "cpu"
        Target JAX device: ``"cpu"``, ``"gpu"``, or ``"cuda"``.
    feature_norm_bound : float | None, default None
        With a value, every row is rescaled to ``l_2`` norm at most that
        much, last in the chain and one record at a time, and the bound
        is recorded in ``metadata["feature_norm_bound"]``. A row's norm
        is its ``val`` vector's, since the indices within a row are
        distinct, so the cap applies to ``val`` and means what it means
        in `load_criteo`. ADR-0012 records what it is for and what a
        wider one costs.

    Returns
    -------
    SparseTabularSplit
        ``idx_*`` are ``(n, 39)`` int32 and ``val_*`` ``(n, 39)``
        float32: columns 0-12 are the numeric features at indices 0-12,
        columns 13-38 the one-hot entry for ``C1..C26`` with value 1.0
        — the value before any cap, since a ``feature_norm_bound``
        rescales the whole ``val`` row afterwards.
        ``num_features`` is the width the indices address.
        ``metadata`` carries ``"encoding"``, ``"preprocess"``,
        ``"preprocessing"`` (the chain in prose), ``"columns"``,
        ``"int_cols"``, ``"cat_cols"``, ``"n_categories"`` — the distinct
        IDs each ``C*`` column had in the training split, as in
        `load_criteo` — ``"column_offsets"``, where each column's block
        begins, ``"num_features"`` again for provenance,
        ``"license"``, and ``"source"``. A ``feature_norm_bound`` that
        was enforced is carried back under that name.

    Raises
    ------
    ValueError
        If ``feature_norm_bound`` is not finite and positive.
    FileNotFoundError
        If ``download=False`` and the file is not in the cache.
    RuntimeError
        If a downloaded file's SHA256 does not match the pinned digest.

    Notes
    -----
    There is no ``standardize`` axis: centring a one-hot column would
    fill it, turning 39 stored values back into 551,947. ADR-0020
    records why the axis is absent rather than present and refused.

    Every row has exactly 39 entries — 13 numeric and one per
    categorical column, the reserved unseen slot included — so for a
    linear model the per-example gradient's sparsity ``s`` is known by
    construction rather than assumed, which is what assumption (A.7) of
    Ghazi et al. 2024 asks for; ADR-0020 records what rests on it.

    The vocabulary and the median are fitted on the training split
    alone, and like every fitted statistic in this module they are not
    privatized; ADR-0008 records the caveat that puts on a reported ε.
    """
    columns = INT_COLS + CAT_COLS
    df = _read_frame(columns, root, download)
    train_df, test_df = _split_frame(df, test_fraction, seed)

    y_train = train_df[LABEL_COL].to_numpy(dtype=np.float32)
    y_test = test_df[LABEL_COL].to_numpy(dtype=np.float32)

    if preprocess:
        int_train, int_test = _prepare_integers(
            train_df[INT_COLS], test_df[INT_COLS]
        )
    else:
        int_train = train_df[INT_COLS].to_numpy(dtype=np.float32)
        int_test = test_df[INT_COLS].to_numpy(dtype=np.float32)

    n_int = len(INT_COLS)
    width = len(columns)
    idx_train = np.empty((int_train.shape[0], width), dtype=np.int32)
    idx_test = np.empty((int_test.shape[0], width), dtype=np.int32)
    # Ones, because a one-hot entry's value is 1.0 and only the numeric
    # block below overwrites what it finds here.
    val_train = np.ones((int_train.shape[0], width), dtype=np.float32)
    val_test = np.ones((int_test.shape[0], width), dtype=np.float32)

    numeric_idx = np.arange(n_int, dtype=np.int32)
    idx_train[:, :n_int] = numeric_idx
    idx_test[:, :n_int] = numeric_idx
    val_train[:, :n_int] = int_train
    val_test[:, :n_int] = int_test

    offsets: list[int] = []
    cardinalities: list[int] = []
    offset = n_int
    for j, col in enumerate(CAT_COLS):
        train_codes = train_df[col].to_numpy()
        test_codes = test_df[col].to_numpy()
        vocabulary = np.unique(train_codes)
        cardinality = vocabulary.shape[0]
        block_train = _one_hot_column(vocabulary, train_codes, offset)
        block_test = _one_hot_column(vocabulary, test_codes, offset)
        idx_train[:, n_int + j] = block_train
        idx_test[:, n_int + j] = block_test
        offsets.append(offset)
        cardinalities.append(cardinality)
        # The + 1 is the slot held for IDs this column never saw in
        # training, which is what keeps every row at `width` entries.
        offset += cardinality + 1
    num_features = offset

    description = (
        _DESCRIPTIONS[("one_hot", preprocess)]
        + _ONE_HOT_CATEGORICAL
        + _norm_bound_clause(feature_norm_bound)
    )
    metadata: dict = {
        "encoding": "one_hot",
        "preprocess": preprocess,
        "preprocessing": description,
        "columns": list(columns),
        "int_cols": list(INT_COLS),
        "cat_cols": list(CAT_COLS),
        "n_categories": cardinalities,
        "column_offsets": offsets,
        "num_features": num_features,
        "license": "CC-BY-NC-SA 4.0",
        "source": _URL,
    }

    if feature_norm_bound is not None:
        # Last, as in `load_criteo`. A row's norm is its `val` vector's,
        # because the indices within a row are distinct.
        val_train, bound = cap_feature_norms(val_train, feature_norm_bound)
        val_test, _ = cap_feature_norms(val_test, feature_norm_bound)
        metadata["feature_norm_bound"] = bound

    emit_once(
        f"criteo:one_hot:{preprocess}:{feature_norm_bound}",
        f"criteo: {description}",
    )

    return arrays_to_sparse_split(
        idx_train,
        val_train,
        y_train,
        idx_test,
        val_test,
        y_test,
        num_features=num_features,
        device=device,
        metadata=metadata,
    )
