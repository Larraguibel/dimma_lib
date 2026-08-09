"""The Criteo 1M sample: a click-prediction benchmark, four ways.

Two independent choices, so four modes. *Which columns* — the 13 integer
features, or all 39 including the 26 categorical ones. *Whether they are
preprocessed* — the raw stored values, or the chain described in
``metadata["preprocessing"]``.

Both are parameters rather than one mode name, because a mode name is
where the interesting part goes missing: the earlier version of this
loader called its preprocessed mode ``"integer"``, which says which
columns it returns and nothing about the log1p and standardization it
also applies. Here ``preprocess`` is unmissable, and whatever it did is
recorded in the returned metadata and printed once.

Every statistic — medians, category frequencies, means, standard
deviations — is fitted on the training split alone and applied to the
test split, so the test set leaks nothing into the features. Note that
this is not itself a private operation: the statistics depend on the
training data and are not accounted for in any privacy budget. That is
the standard benchmark convention, not a claim about the pipeline.

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
from dimma.datasets.base import TabularSplit, arrays_to_split
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
        "13 integer features I1-I13 exactly as stored, cast to float32. "
        "NaN preserved. No fill, no log1p, no standardization."
    ),
    ("numeric", True): (
        "13 integer features I1-I13: NaN filled with the train-split median, "
        "clipped below at 0, log1p, then standardized by the train-split mean "
        "and standard deviation."
    ),
    ("all", False): (
        "39 features I1-I13 and C1-C26 exactly as stored, cast to float32. "
        "NaN preserved. C* are integer category IDs, which float32 represents "
        "exactly below 2**24. No fill, no log1p, no encoding, no "
        "standardization."
    ),
    ("all", True): (
        "39 features. I1-I13: NaN filled with the train-split median, clipped "
        "below at 0, log1p. C1-C26: each category ID replaced by that "
        "category's relative frequency in the train split, with categories "
        "unseen in training encoded as 0.0. All 39 columns then standardized "
        "by the train-split mean and standard deviation."
    ),
}


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
    uniq, counts = np.unique(train_codes, return_counts=True)
    freq = counts / train_codes.shape[0]

    pos = np.searchsorted(uniq, codes)
    pos_safe = np.clip(pos, 0, uniq.shape[0] - 1)
    seen = uniq[pos_safe] == codes
    return np.where(seen, freq[pos_safe], 0.0).astype(np.float32), uniq.shape[0]


def _prepare_integers(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Median-fill, clip at 0, log1p. Statistics from the train split only."""
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
        ``"numeric"`` returns the 13 integer features ``I1..I13``.
        ``"all"`` returns those plus the 26 categorical features
        ``C1..C26``, which are integer category IDs in the source file.
    preprocess : bool, default True
        Whether to apply the chain recorded in
        ``metadata["preprocessing"]`` and printed once per process. With
        ``False`` the stored values are returned untouched beyond a cast
        to float32, NaN included, and the caller owns everything
        downstream.
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
        ``"preprocessing"`` (the chain in prose), ``"columns"``,
        ``"license"``, and ``"source"``. With ``preprocess=True`` it also
        carries ``"feature_means"`` and ``"feature_stds"``, the arrays the
        columns were standardized by; with ``features="all"`` it carries
        ``"int_cols"`` and ``"cat_cols"``, and when both hold,
        ``"n_categories"`` — the distinct IDs each ``C*`` column had in
        the training split, before frequency encoding collapsed it to one
        float. Whatever ``features`` and ``preprocess`` are, a
        ``feature_norm_bound`` that was enforced is carried back under
        that name.

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

    columns = INT_COLS if features == "numeric" else INT_COLS + CAT_COLS
    df = pd.read_parquet(parquet_path, columns=columns + [LABEL_COL])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_test = int(round(len(df) * test_fraction))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    y_train = train_df[LABEL_COL].to_numpy(dtype=np.float32)
    y_test = test_df[LABEL_COL].to_numpy(dtype=np.float32)

    description = _DESCRIPTIONS[(features, preprocess)]
    if feature_norm_bound is not None:
        description += (
            f" Every row then rescaled to l2 norm at most "
            f"{feature_norm_bound}, one record at a time."
        )
    metadata: dict = {
        "features": features,
        "preprocess": preprocess,
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
        f"criteo:{features}:{preprocess}:{feature_norm_bound}",
        f"criteo: {description}",
    )

    return arrays_to_split(
        x_train, y_train, x_test, y_test, device=device, metadata=metadata
    )
