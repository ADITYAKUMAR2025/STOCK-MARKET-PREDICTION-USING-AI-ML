"""
A deliberately plain long/flat backtest.

Directional accuracy is not a P&L. A model can be right 55% of the time and
still lose money if it is wrong on the large moves, or if it trades often
enough that costs eat the edge. This module turns a prediction series into
an equity curve so that question gets answered rather than assumed.

Assumptions, stated plainly because they flatter the result:

* Long or flat only, full size, no leverage and no shorting.
* Entry and exit at the close that the signal is computed from, which is
  optimistic — a real implementation fills on the next open.
* Costs are a fixed ``cost_bps`` charged on each change of position.
* No slippage model, no market impact, no borrow cost, no taxes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    n_trades: float
    exposure: float
    equity_curve: pd.Series

    def summary(self) -> dict:
        out = asdict(self)
        out.pop("equity_curve")
        return out


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fractional decline of an equity curve."""
    running_peak = equity.cummax()
    return float((equity / running_peak - 1.0).min())


def backtest_long_flat(
    predictions: np.ndarray | pd.Series,
    forward_returns: pd.Series,
    cost_bps: float = 5.0,
) -> BacktestResult:
    """Run a long/flat backtest of ``predictions`` against ``forward_returns``.

    ``predictions`` are 1 (hold long over the next period) or 0 (stay flat).
    ``forward_returns`` are the log returns realised over that same period,
    as produced by :func:`stockpred.features.make_dataset`.
    """
    position = pd.Series(
        np.asarray(predictions).astype(float), index=forward_returns.index
    )

    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (cost_bps / 10_000.0)

    strategy_log_return = position * forward_returns - cost
    equity = np.exp(strategy_log_return.cumsum())

    n_periods = len(strategy_log_return)
    years = n_periods / TRADING_DAYS_PER_YEAR

    total_return = float(equity.iloc[-1] - 1.0) if n_periods else 0.0
    ann_return = float(equity.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0
    ann_vol = float(strategy_log_return.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0

    traded = position > 0
    hit_rate = (
        float((forward_returns[traded] > 0).mean()) if traded.any() else float("nan")
    )

    return BacktestResult(
        total_return=total_return,
        annualised_return=ann_return,
        annualised_volatility=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown(equity),
        hit_rate=hit_rate,
        n_trades=float(turnover.sum()),
        exposure=float(position.mean()),
        equity_curve=equity,
    )


def buy_and_hold(forward_returns: pd.Series) -> BacktestResult:
    """The strategy any model has to beat to be worth running."""
    return backtest_long_flat(
        np.ones(len(forward_returns)), forward_returns, cost_bps=0.0
    )
