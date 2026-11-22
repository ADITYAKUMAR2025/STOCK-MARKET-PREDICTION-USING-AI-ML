"""Backtest arithmetic, checked against cases with a known answer."""

import numpy as np
import pandas as pd
import pytest

from stockpred.backtest import backtest_long_flat, buy_and_hold, max_drawdown


@pytest.fixture
def returns():
    idx = pd.bdate_range("2020-01-01", periods=250, name="date")
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)


def test_staying_flat_earns_nothing(returns):
    result = backtest_long_flat(np.zeros(len(returns)), returns, cost_bps=5.0)
    assert result.total_return == pytest.approx(0.0, abs=1e-12)
    assert result.exposure == 0.0


def test_perfect_foresight_beats_buy_and_hold(returns):
    oracle = (returns > 0).astype(int).to_numpy()
    assert (
        backtest_long_flat(oracle, returns, cost_bps=0.0).total_return
        > buy_and_hold(returns).total_return
    )


def test_costs_only_ever_reduce_return(returns):
    signal = np.resize([1, 0], len(returns))  # flips every single day
    free = backtest_long_flat(signal, returns, cost_bps=0.0).total_return
    charged = backtest_long_flat(signal, returns, cost_bps=25.0).total_return
    assert charged < free


def test_turnover_counts_position_changes(returns):
    held = backtest_long_flat(np.ones(len(returns)), returns)
    flipped = backtest_long_flat(np.resize([1, 0], len(returns)), returns)
    assert held.n_trades == pytest.approx(1.0)
    assert flipped.n_trades > len(returns) / 2


def test_max_drawdown_of_a_known_curve():
    equity = pd.Series([1.0, 1.25, 1.0, 1.5])  # 20% fall from the 1.25 peak
    assert max_drawdown(equity) == pytest.approx(-0.2)


def test_max_drawdown_is_never_positive(returns):
    result = backtest_long_flat((returns > 0).astype(int).to_numpy(), returns)
    assert result.max_drawdown <= 0
