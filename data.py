"""
Price data loading.

Two sources are supported:

* ``yfinance`` — real daily OHLCV data, used when the package is installed
  and the network is reachable.
* ``synthetic`` — a seeded geometric-random-walk generator with volatility
  regimes, used as a fallback so the pipeline is runnable offline and in CI.

The synthetic series has **no predictable structure beyond volatility
clustering**. That is deliberate: a model that reports strong directional
accuracy on synthetic data has a bug, and this gives you a way to catch it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class PriceData:
    """Daily OHLCV frame plus provenance, so results are never ambiguous."""

    frame: pd.DataFrame
    ticker: str
    source: str

    def __post_init__(self) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"price frame missing columns: {sorted(missing)}")
        if not self.frame.index.is_monotonic_increasing:
            raise ValueError("price frame index must be sorted ascending by date")

    @property
    def is_synthetic(self) -> bool:
        return self.source == "synthetic"

    def __len__(self) -> int:
        return len(self.frame)


def load_synthetic(
    ticker: str = "SYNTH",
    n_days: int = 2000,
    seed: int = 7,
    start: str = "2016-01-01",
) -> PriceData:
    """Generate a seeded price series with volatility regimes.

    Daily log returns are drawn from a normal distribution whose standard
    deviation switches between a calm and a stressed regime via a two-state
    Markov chain. Drift is small and positive, as equity indices tend to be.
    """
    rng = np.random.default_rng(seed)

    calm_vol, stressed_vol = 0.008, 0.022
    p_stay_calm, p_stay_stressed = 0.985, 0.95
    daily_drift = 0.00025

    regimes = np.empty(n_days, dtype=bool)  # True == stressed
    regimes[0] = False
    for t in range(1, n_days):
        if regimes[t - 1]:
            regimes[t] = rng.random() < p_stay_stressed
        else:
            regimes[t] = rng.random() >= p_stay_calm

    vol = np.where(regimes, stressed_vol, calm_vol)
    log_returns = rng.normal(loc=daily_drift, scale=vol)

    close = 100.0 * np.exp(np.cumsum(log_returns))
    intraday = np.abs(rng.normal(0, 0.004, n_days)) * close
    open_ = close * (1 + rng.normal(0, 0.002, n_days))
    high = np.maximum(open_, close) + intraday
    low = np.minimum(open_, close) - intraday
    volume = rng.lognormal(mean=15.0, sigma=0.35, size=n_days) * (1 + 2 * regimes)

    index = pd.bdate_range(start=start, periods=n_days, name="date")
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume.round(),
        },
        index=index,
    )
    return PriceData(frame=frame, ticker=ticker, source="synthetic")


def load_yfinance(ticker: str, start: str, end: str | None = None) -> PriceData:
    """Download daily OHLCV from Yahoo Finance. Raises if unavailable."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "yfinance is not installed. Install it with `pip install yfinance`, "
            "or use source='synthetic'."
        ) from exc

    raw = yf.download(
        ticker, start=start, end=end, progress=False, auto_adjust=True
    )
    if raw is None or raw.empty:  # pragma: no cover - network dependent
        raise RuntimeError(f"yfinance returned no rows for {ticker!r}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    frame = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    frame.index.name = "date"
    return PriceData(frame=frame.sort_index(), ticker=ticker, source="yfinance")


def load_prices(
    ticker: str = "^NSEI",
    start: str = "2016-01-01",
    end: str | None = None,
    source: str = "auto",
    n_days: int = 2000,
    seed: int = 7,
) -> PriceData:
    """Load prices, falling back to synthetic data when real data is unreachable.

    Parameters
    ----------
    source
        ``"yfinance"`` to require real data, ``"synthetic"`` to force the
        generator, or ``"auto"`` (default) to try real data and fall back.
    """
    if source == "synthetic":
        return load_synthetic(ticker=ticker, n_days=n_days, seed=seed, start=start)

    if source == "yfinance":
        return load_yfinance(ticker, start=start, end=end)

    if source != "auto":
        raise ValueError(f"unknown source {source!r}")

    try:
        return load_yfinance(ticker, start=start, end=end)
    except Exception as exc:
        warnings.warn(
            f"Could not load real data for {ticker!r} ({exc}). "
            "Falling back to synthetic prices — results below describe a "
            "random walk, not a real market.",
            RuntimeWarning,
            stacklevel=2,
        )
        return load_synthetic(ticker=ticker, n_days=n_days, seed=seed, start=start)
