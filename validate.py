"""
Walk-forward (expanding window) cross-validation for time series.

``sklearn.model_selection.KFold`` shuffles, which on a price series means
training on next month to predict last month. Even ``TimeSeriesSplit``
leaves the test fold immediately adjacent to the training fold, so an
``h``-day forward target leaks ``h`` days of information across the boundary.

``walk_forward_splits`` adds an explicit embargo gap of ``horizon`` rows
between train and test to close that leak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    index: int
    train: np.ndarray
    test: np.ndarray

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Fold({self.index}: train[0:{self.train[-1] + 1}] "
            f"test[{self.test[0]}:{self.test[-1] + 1}])"
        )


def walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    min_train: int = 250,
    horizon: int = 1,
) -> list[Fold]:
    """Produce expanding-window folds with an embargo gap.

    Each fold trains on everything from the start of the series up to a
    cutoff, skips ``horizon`` rows, then tests on the following block. Train
    always precedes test; no row is ever in both.
    """
    if n_samples <= min_train + horizon + n_splits:
        raise ValueError(
            f"not enough samples ({n_samples}) for {n_splits} folds with "
            f"min_train={min_train} and horizon={horizon}"
        )

    usable = n_samples - min_train - horizon
    test_size = usable // n_splits
    if test_size < 1:
        raise ValueError("test blocks would be empty; reduce n_splits or min_train")

    folds: list[Fold] = []
    for i in range(n_splits):
        train_end = min_train + i * test_size
        test_start = train_end + horizon
        test_end = test_start + test_size if i < n_splits - 1 else n_samples
        if test_start >= n_samples:
            break
        folds.append(
            Fold(
                index=i,
                train=np.arange(0, train_end),
                test=np.arange(test_start, min(test_end, n_samples)),
            )
        )
    return folds
