"""
The walk-forward experiment loop that produces the results table.

For every model and every fold: fit on the expanding training window,
predict the held-out block, record directional accuracy, then concatenate
the out-of-sample predictions across folds and backtest them once. Nothing
is scored in-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score

from .backtest import backtest_long_flat, buy_and_hold
from .models import model_zoo
from .validate import walk_forward_splits


def _fit_predict(model, X_train, y_train, X_test):
    """Fit a fresh copy of ``model`` and return predictions plus probabilities."""
    try:
        estimator = clone(model)
    except TypeError:
        estimator = model  # our hand-written baselines are not sklearn clones
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    try:
        proba = estimator.predict_proba(X_test)[:, 1]
    except (AttributeError, IndexError):
        proba = None
    return np.asarray(preds).astype(int), proba


def run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    forward_returns: pd.Series,
    n_splits: int = 5,
    min_train: int = 500,
    horizon: int = 1,
    cost_bps: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every model in the zoo on walk-forward folds.

    Returns ``(per_fold, summary)``: accuracy per model per fold, and one
    row per model combining pooled out-of-sample accuracy with backtest
    statistics.
    """
    folds = walk_forward_splits(
        len(X), n_splits=n_splits, min_train=min_train, horizon=horizon
    )
    models = model_zoo()

    fold_rows: list[dict] = []
    pooled: dict[str, list] = {name: [] for name in models}
    pooled_index: dict[str, list] = {name: [] for name in models}

    for fold in folds:
        X_train, y_train = X.iloc[fold.train], y.iloc[fold.train]
        X_test, y_test = X.iloc[fold.test], y.iloc[fold.test]

        for name, model in models.items():
            preds, proba = _fit_predict(model, X_train, y_train, X_test)
            acc = accuracy_score(y_test, preds)
            try:
                auc = roc_auc_score(y_test, proba) if proba is not None else np.nan
            except ValueError:
                auc = np.nan

            fold_rows.append(
                {
                    "fold": fold.index,
                    "model": name,
                    "n_train": len(fold.train),
                    "n_test": len(fold.test),
                    "accuracy": acc,
                    "roc_auc": auc,
                }
            )
            pooled[name].extend(preds.tolist())
            pooled_index[name].extend(X_test.index.tolist())

    per_fold = pd.DataFrame(fold_rows)

    bh = buy_and_hold(forward_returns.loc[pooled_index[next(iter(models))]])
    summary_rows: list[dict] = []
    for name in models:
        idx = pooled_index[name]
        preds = pd.Series(pooled[name], index=idx)
        oos_truth = y.loc[idx]
        result = backtest_long_flat(
            preds.to_numpy(), forward_returns.loc[idx], cost_bps=cost_bps
        )
        summary_rows.append(
            {
                "model": name,
                "oos_accuracy": accuracy_score(oos_truth, preds),
                "accuracy_std_across_folds": per_fold.loc[
                    per_fold.model == name, "accuracy"
                ].std(),
                "sharpe": result.sharpe_ratio,
                "ann_return": result.annualised_return,
                "max_drawdown": result.max_drawdown,
                "exposure": result.exposure,
                "n_trades": result.n_trades,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        "oos_accuracy", ascending=False, ignore_index=True
    )
    summary.attrs["buy_and_hold_sharpe"] = bh.sharpe_ratio
    summary.attrs["buy_and_hold_ann_return"] = bh.annualised_return
    summary.attrs["base_rate_up"] = float(y.loc[pooled_index[name]].mean())
    return per_fold, summary
