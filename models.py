"""
Models, including the baselines that make the other numbers interpretable.

A directional accuracy of 52% means nothing on its own. It has to be read
against what the market gives away for free, which on a drifting equity
series is "always predict up" — often 52-54%. Both free strategies are
implemented here as first-class models so they appear in every results
table.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class AlwaysUp:
    """Predict "up" every day. The benchmark equity drift gives for free."""

    name = "always_up"

    def fit(self, X, y):  # noqa: D102
        return self

    def predict(self, X):  # noqa: D102
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):  # noqa: D102
        p = np.ones(len(X))
        return np.column_stack([1 - p, p])


class PersistenceBaseline:
    """Predict that today's direction repeats yesterday's realised return.

    Uses the ``ret_lag_1`` column, which is the most recent return known at
    prediction time.
    """

    name = "persistence"

    def __init__(self, column: str = "ret_lag_1") -> None:
        self.column = column

    def fit(self, X, y):  # noqa: D102
        if self.column not in X.columns:
            raise KeyError(f"{self.column!r} not in feature matrix")
        return self

    def predict(self, X):  # noqa: D102
        return (X[self.column].to_numpy() > 0).astype(int)

    def predict_proba(self, X):  # noqa: D102
        p = self.predict(X).astype(float)
        return np.column_stack([1 - p, p])


def logistic_model() -> Pipeline:
    """L2 logistic regression on standardised features."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, C=0.1, random_state=42),
            ),
        ]
    )


def gradient_boosting_model() -> GradientBoostingClassifier:
    """Shallow gradient boosting — deliberately capacity-limited.

    Depth 2 with 150 trees on ~20 noisy features. Anything deeper memorises
    the training window and the walk-forward folds show it immediately.
    """
    return GradientBoostingClassifier(
        n_estimators=150,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        random_state=42,
    )


def random_forest_model() -> RandomForestClassifier:
    """Random forest with a floor on leaf size to limit overfitting."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42,
    )


def model_zoo() -> dict:
    """The full set of models scored by ``scripts/run_experiment.py``."""
    return {
        "always_up": AlwaysUp(),
        "persistence": PersistenceBaseline(),
        "logistic": logistic_model(),
        "gradient_boosting": gradient_boosting_model(),
        "random_forest": random_forest_model(),
    }
