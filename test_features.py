"""
The look-ahead tests. These are the point of the repository.

If a feature on row t used information from row t+1 or later, the first test
here fails. That is the single most common defect in published
stock-prediction code, and it is cheap to test for: change the future, and
check that the past did not move.
"""

import numpy as np
import pandas as pd
import pytest

from stockpred.data import load_synthetic
from stockpred.features import build_features, make_dataset


@pytest.fixture
def prices():
    return load_synthetic(n_days=400, seed=11).frame


def test_features_do_not_depend_on_the_future(prices):
    """Perturbing the tail must leave every earlier feature row untouched."""
    cutoff = 300

    baseline = build_features(prices).iloc[:cutoff]

    tampered = prices.copy()
    tampered.iloc[cutoff:, :] *= 5.0  # a violent, unmistakable future shock
    after = build_features(tampered).iloc[:cutoff]

    pd.testing.assert_frame_equal(baseline, after)


def test_target_is_strictly_forward_looking(prices):
    """y on row t must describe the move from t to t+1, not t-1 to t."""
    X, y, forward = make_dataset(prices, horizon=1)

    log_close = np.log(prices["close"])
    expected = (log_close.shift(-1) - log_close).loc[X.index]

    pd.testing.assert_series_equal(
        forward, expected, check_names=False, rtol=1e-12
    )
    assert ((forward > 0).astype(int) == y).all()


def test_last_rows_are_dropped_because_their_future_is_unknown(prices):
    """With horizon h, the final h rows cannot have a label."""
    for horizon in (1, 3, 5):
        X, _, _ = make_dataset(prices, horizon=horizon)
        assert X.index[-1] <= prices.index[-1 - horizon]


def test_no_nans_survive_into_the_dataset(prices):
    X, y, forward = make_dataset(prices)
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert not forward.isna().any()
    assert len(X) == len(y) == len(forward)


def test_leakage_canary_would_be_caught():
    """A deliberately leaked feature must be learnable to near-perfection.

    This guards the guard: it shows that the walk-forward harness is capable
    of detecting leakage, so a clean result is meaningful rather than an
    artefact of a weak evaluation.
    """
    from sklearn.linear_model import LogisticRegression

    frame = load_synthetic(n_days=600, seed=3).frame
    X, y, _ = make_dataset(frame)

    leaked = X.copy()
    leaked["tomorrow_peek"] = y.to_numpy()  # the answer, handed over

    split = 400
    model = LogisticRegression(max_iter=1000).fit(
        leaked.iloc[:split], y.iloc[:split]
    )
    cheating_accuracy = model.score(leaked.iloc[split:], y.iloc[split:])

    assert cheating_accuracy > 0.95, (
        "the harness failed to reward an obviously leaked feature, so it "
        "cannot be trusted to detect subtler leakage"
    )


def test_honest_features_are_not_predictive_on_a_random_walk():
    """The complement of the canary: clean features must NOT be predictive.

    Synthetic prices are a random walk with volatility regimes. Direction is
    genuinely unpredictable, so a clean pipeline must land near the base
    rate. A high score here means look-ahead bias has crept back in.
    """
    from sklearn.linear_model import LogisticRegression

    frame = load_synthetic(n_days=1200, seed=5).frame
    X, y, _ = make_dataset(frame)

    split = 800
    model = LogisticRegression(max_iter=2000).fit(X.iloc[:split], y.iloc[:split])
    accuracy = model.score(X.iloc[split:], y.iloc[split:])

    assert accuracy < 0.60, (
        f"out-of-sample accuracy of {accuracy:.1%} on a random walk indicates "
        "look-ahead bias, not skill"
    )
