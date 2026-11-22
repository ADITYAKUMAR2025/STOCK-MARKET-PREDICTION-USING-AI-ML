"""Walk-forward splitting must never let the future train the past."""

import numpy as np
import pytest

from stockpred.validate import walk_forward_splits


def test_train_always_precedes_test_with_an_embargo_gap():
    for horizon in (1, 3, 10):
        folds = walk_forward_splits(
            2000, n_splits=5, min_train=400, horizon=horizon
        )
        for fold in folds:
            assert fold.train[-1] < fold.test[0]
            assert fold.test[0] - fold.train[-1] >= horizon


def test_train_and_test_never_overlap():
    folds = walk_forward_splits(1500, n_splits=6, min_train=300, horizon=1)
    for fold in folds:
        assert not set(fold.train) & set(fold.test)


def test_training_window_expands_monotonically():
    folds = walk_forward_splits(1500, n_splits=5, min_train=300, horizon=1)
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_all_indices_are_in_range():
    n = 900
    folds = walk_forward_splits(n, n_splits=4, min_train=300, horizon=2)
    for fold in folds:
        assert fold.train.min() >= 0
        assert fold.test.max() < n


def test_rejects_impossible_configurations():
    with pytest.raises(ValueError):
        walk_forward_splits(100, n_splits=5, min_train=250, horizon=1)
