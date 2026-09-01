"""Vectorised grouped trailing statistics.

``groupby(...).transform(lambda s: s.shift(1).rolling(w).mean())`` is the
obvious way to write "each team's form before this game", and it is also the
single slowest thing in the feature build -- pandas runs the lambda once per
group in Python.  These helpers compute the same values from prefix sums in
numpy, which is roughly two orders of magnitude faster and, more importantly,
keeps the cost flat as history grows.

All functions assume the input is already sorted by group and then by time,
which ``team_game_long`` guarantees.  Every window **excludes the current row**
-- that is the whole point: a game must never see its own result.
"""
from __future__ import annotations

import numpy as np


def _group_starts(codes: np.ndarray) -> np.ndarray:
    """First row index of each row's group, for contiguous sorted codes."""
    if codes.size == 0:
        return codes
    boundary = np.empty(codes.size, dtype=bool)
    boundary[0] = True
    np.not_equal(codes[1:], codes[:-1], out=boundary[1:])
    starts = np.flatnonzero(boundary)
    # Broadcast each group's start index across its rows.
    return np.repeat(starts, np.diff(np.append(starts, codes.size)))


def trailing_mean(values: np.ndarray, codes: np.ndarray,
                  window: int | None = None) -> np.ndarray:
    """Mean of the previous ``window`` values within each group.

    ``window=None`` means expanding (all prior rows).  NaN inputs are skipped,
    matching ``rolling(..., min_periods=1).mean()``.  Rows with no prior values
    come back NaN.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return values.copy()
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)

    # Prefix sums, offset by one so index i holds the total strictly before i.
    csum = np.concatenate(([0.0], np.cumsum(filled)))
    ccnt = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))

    starts = _group_starts(np.asarray(codes))
    idx = np.arange(n)
    if window is None:
        lo = starts
    else:
        lo = np.maximum(starts, idx - int(window))

    total = csum[idx] - csum[lo]
    count = ccnt[idx] - ccnt[lo]
    out = np.full(n, np.nan)
    np.divide(total, count, out=out, where=count > 0)
    return out


def trailing_count(codes: np.ndarray) -> np.ndarray:
    """Number of prior rows within each group -- i.e. games played so far."""
    codes = np.asarray(codes)
    return np.arange(codes.size) - _group_starts(codes)


def group_codes(*keys) -> np.ndarray:
    """Contiguous integer codes for already-sorted grouping columns."""
    if not keys:
        raise ValueError("need at least one key")
    n = len(keys[0])
    boundary = np.zeros(n, dtype=bool)
    if n:
        boundary[0] = True
    for k in keys:
        arr = np.asarray(k, dtype=object)
        if n > 1:
            boundary[1:] |= arr[1:] != arr[:-1]
    return np.cumsum(boundary) - 1
