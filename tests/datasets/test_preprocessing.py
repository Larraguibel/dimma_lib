"""The per-record feature-norm cap.

The two properties the privacy argument rests on: every row inside the
ball afterwards, and nothing read across records to get there.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dimma.datasets.preprocessing import cap_feature_norms


def rows() -> np.ndarray:
    """Norms 5, 1, 0, 10 and 0.07: at R = 1, two over and one exactly on."""
    return np.array(
        [
            [3.0, 4.0],      # norm 5
            [0.6, 0.8],      # norm 1
            [0.0, 0.0],      # norm 0
            [-10.0, 0.0],    # norm 10, negative
            [0.05, 0.05],    # norm well under
        ],
        dtype=np.float32,
    )


def test_every_row_is_inside_the_ball_afterwards():
    capped, _ = cap_feature_norms(rows(), 1.0)
    assert np.all(np.linalg.norm(capped, axis=1) <= 1.0 + 1e-6)


def test_a_row_already_inside_is_returned_bit_for_bit():
    """The divisor is exactly 1.0 there, so the majority of records
    carry no distortion at all."""
    x = rows()
    capped, _ = cap_feature_norms(x, 1.0)
    inside = np.linalg.norm(x, axis=1) <= 1.0
    assert np.array_equal(capped[inside], x[inside])


def test_a_zero_row_survives_without_an_epsilon():
    """`x / max(1, ||x||/R)` divides by 1.0 here. The `min(1, R/||x||)`
    form would divide by zero, which is why stage 4 needs its 1e-12."""
    capped, _ = cap_feature_norms(rows(), 1.0)
    assert np.all(np.isfinite(capped))
    assert np.array_equal(capped[2], np.zeros(2, dtype=np.float32))


def test_the_cap_rescales_rather_than_rotates():
    x = rows()
    capped, _ = cap_feature_norms(x, 1.0)
    outside = np.linalg.norm(x, axis=1) > 1.0
    cosines = np.sum(capped[outside] * x[outside], axis=1) / (
        np.linalg.norm(capped[outside], axis=1)
        * np.linalg.norm(x[outside], axis=1)
    )
    assert np.allclose(cosines, 1.0, atol=1e-6)


def test_an_over_norm_row_lands_on_the_boundary():
    """Worked by hand: the row of norm 5 scales to 3/5, 4/5 at R = 1."""
    capped, _ = cap_feature_norms(rows(), 1.0)
    assert np.allclose(capped[0], [0.6, 0.8], atol=1e-6)


def test_capping_twice_is_capping_once():
    once, _ = cap_feature_norms(rows(), 1.0)
    twice, _ = cap_feature_norms(once, 1.0)
    assert np.allclose(once, twice, atol=1e-7)


def test_the_bound_comes_back_so_it_need_not_be_typed_again():
    _, enforced = cap_feature_norms(rows(), 0.5)
    assert enforced == 0.5


def test_nothing_about_the_data_comes_back_with_it():
    """A count of rescaled rows, or their norms, would be an
    unaccounted release through the return value."""
    assert len(cap_feature_norms(rows(), 1.0)) == 2


def test_float32_survives_the_cap():
    capped, _ = cap_feature_norms(rows(), 1.0)
    assert capped.dtype == np.float32


def with_a_hole() -> np.ndarray:
    """The same rows, with a NaN put where a fill would have gone."""
    x = rows()
    x[3, 0] = np.nan
    return x


def test_a_row_whose_norm_is_not_finite_is_announced():
    """The cap cannot make the bound true for such a row, and a bound
    believed but not enforced is the silent-false-epsilon failure."""
    with pytest.warns(UserWarning, match="finite"):
        cap_feature_norms(with_a_hole(), 1.0)


def test_the_warning_does_not_say_how_many_rows_it_could_not_bound():
    """A count of offending rows is a statistic of the data, and a
    warning is as much a release as a return value. One hole and three
    must therefore read identically."""
    one, three = with_a_hole(), with_a_hole()
    three[0, 1] = np.nan
    three[4, 0] = np.inf

    messages = []
    for x in (one, three):
        with pytest.warns(UserWarning) as caught:
            cap_feature_norms(x, 1.0)
        messages.append(str(caught[0].message))
    assert messages[0] == messages[1]


def test_the_announcement_is_the_only_thing_said():
    """numpy's own `invalid value encountered in divide` says less and
    would reach a caller who turned warnings into errors first."""
    with pytest.warns(UserWarning) as caught:
        cap_feature_norms(with_a_hole(), 1.0)
    assert len(caught) == 1


def test_clean_data_passes_without_a_word():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cap_feature_norms(rows(), 1.0)


def test_the_other_rows_are_capped_as_usual():
    """Warned, not refused: a caller preprocessing their own way keeps
    the rows the cap can speak for."""
    with pytest.warns(UserWarning):
        capped, _ = cap_feature_norms(with_a_hole(), 1.0)
    intact = np.isfinite(capped).all(axis=1)
    assert np.all(np.linalg.norm(capped[intact], axis=1) <= 1.0 + 1e-6)
    assert np.allclose(capped[0], [0.6, 0.8], atol=1e-6)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_a_bound_that_bounds_nothing_is_rejected(bad):
    """`R` is the whole privacy argument; a nonsense one must not pass
    silently into an accountant."""
    with pytest.raises(ValueError, match="bound"):
        cap_feature_norms(rows(), bad)
