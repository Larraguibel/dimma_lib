"""The Criteo 1M sample: a click-prediction benchmark, eight ways.

Three independent choices, so eight modes. *Which columns* — the 13
integer features, or all 39 including the 26 categorical ones. *Whether
they are preprocessed* — the stored values as they come, or the fill,
clip, log1p and frequency encoding described in
``metadata["preprocessing"]``. *Whether they are standardized* — column
means and standard deviations fitted on the training split, or the
scales the columns already had. Stored is not raw; see below.

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

    description = (
        _DESCRIPTIONS[(features, preprocess)] + _STANDARDIZATION[standardize]
    )
    if feature_norm_bound is not None:
        description += (
            f" Every row then rescaled to l2 norm at most "
            f"{feature_norm_bound}, one record at a time."
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
