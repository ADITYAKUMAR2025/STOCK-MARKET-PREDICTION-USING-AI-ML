"""
Feature engineering, with look-ahead bias designed out.

The contract every function in this module obeys:

    A feature value on row ``t`` may use price and volume data from rows
    ``<= t`` only. The target on row ``t`` is the return realised between
    ``t`` and ``t + 1``.

That single convention is what separates a believable result from the
"99% accuracy" results that are common in this problem space. Those almost
always come from feeding the model the same day's close (or a centred
rolling window) and asking it to predict that day's price.

``make_dataset`` enforces the contract by construction, and
``tests/test_features.py`` asserts it by perturbing the future and checking
that no feature row changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 5, 10)
VOL_WINDOWS = (5, 10, 21)
MA_WINDOWS = (5, 10, 21, 50)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using a trailing simple average of gains/losses."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_histogram(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """MACD histogram. ``ewm`` is causal, so this uses no future information."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    return macd - macd.ewm(span=signal, adjust=False).mean()


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the trailing-only feature matrix from an OHLCV frame."""
    close = frame["close"]
    log_return = np.log(close).diff()

    features = pd.DataFrame(index=frame.index)

    # Lagged returns. Lag 1 is "yesterday's realised return", known at t.
    for lag in LAGS:
        features[f"ret_lag_{lag}"] = log_return.shift(lag - 1)

    # Trailing realised volatility.
    for window in VOL_WINDOWS:
        features[f"vol_{window}"] = log_return.rolling(
            window, min_periods=window
        ).std()

    # Distance from trailing moving averages, in volatility units.
    for window in MA_WINDOWS:
        ma = close.rolling(window, min_periods=window).mean()
        features[f"close_vs_ma_{window}"] = (close - ma) / close

    # Momentum over trailing windows.
    for window in (5, 21, 63):
        features[f"momentum_{window}"] = np.log(close / close.shift(window))

    # Oscillators.
    features["rsi_14"] = _rsi(close, 14)
    features["macd_hist"] = _macd_histogram(close)

    # Intraday range and volume pressure, both same-day-close safe.
    features["hl_range"] = (frame["high"] - frame["low"]) / close
    volume = frame["volume"].replace(0, np.nan)
    features["volume_ratio_21"] = volume / volume.rolling(21, min_periods=21).mean()

    # Day of week can matter for short-horizon effects.
    features["day_of_week"] = frame.index.dayofweek

    return features


def make_dataset(
    frame: pd.DataFrame, horizon: int = 1
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Assemble aligned ``(X, y_direction, forward_return)``.

    ``y_direction`` is 1 when the forward log return over ``horizon`` days is
    positive, else 0. ``forward_return`` is that same log return, kept
    separately so the backtest can size a position without re-deriving it.

    The final ``horizon`` rows are dropped: their future is not yet known.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    features = build_features(frame)
    log_close = np.log(frame["close"])
    forward_return = log_close.shift(-horizon) - log_close

    data = features.copy()
    data["__forward_return"] = forward_return
    data = data.dropna()

    forward = data.pop("__forward_return")
    direction = (forward > 0).astype(int)
    direction.name = "direction"
    forward.name = "forward_return"

    return data, direction, forward
