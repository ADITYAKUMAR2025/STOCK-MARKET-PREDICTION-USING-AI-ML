#!/usr/bin/env python3
"""
Run the full walk-forward experiment and print the results table.

Examples
--------
    python scripts/run_experiment.py                       # auto source
    python scripts/run_experiment.py --source synthetic    # offline, seeded
    python scripts/run_experiment.py --ticker RELIANCE.NS --source yfinance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from stockpred.data import load_prices  # noqa: E402
from stockpred.experiment import run_walk_forward  # noqa: E402
from stockpred.features import make_dataset  # noqa: E402

RESULTS_DIR = ROOT / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker", default="^NSEI", help="symbol (default: ^NSEI, Nifty 50)")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=None)
    p.add_argument(
        "--source",
        default="auto",
        choices=["auto", "yfinance", "synthetic"],
        help="auto tries yfinance then falls back to synthetic",
    )
    p.add_argument("--horizon", type=int, default=1, help="forecast horizon in days")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--min-train", type=int, default=500)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--save", action="store_true", help="write CSVs to results/")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 20)

    prices = load_prices(
        ticker=args.ticker, start=args.start, end=args.end, source=args.source
    )
    X, y, forward = make_dataset(prices.frame, horizon=args.horizon)

    print(f"Ticker          : {prices.ticker}")
    print(f"Data source     : {prices.source}")
    if prices.is_synthetic:
        print("                  (random walk - no real predictable structure)")
    print(f"Rows usable     : {len(X)}  ({X.index[0].date()} to {X.index[-1].date()})")
    print(f"Features        : {X.shape[1]}")
    print(f"Horizon         : {args.horizon} day(s)")
    print(f"Base rate 'up'  : {y.mean():.3%}")

    per_fold, summary = run_walk_forward(
        X,
        y,
        forward,
        n_splits=args.n_splits,
        min_train=args.min_train,
        horizon=args.horizon,
        cost_bps=args.cost_bps,
    )

    print("\n--- Out-of-sample results, pooled across walk-forward folds ---")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"\nBuy and hold: Sharpe {summary.attrs['buy_and_hold_sharpe']:.4f}, "
          f"annualised return {summary.attrs['buy_and_hold_ann_return']:.4%}")

    print("\n--- Accuracy by fold ---")
    pivot = per_fold.pivot(index="model", columns="fold", values="accuracy")
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    best = summary.iloc[0]
    baseline = summary.loc[summary.model.isin(["always_up", "persistence"]), "oos_accuracy"].max()
    print(
        f"\nBest model '{best.model}' at {best.oos_accuracy:.2%} vs best free "
        f"baseline at {baseline:.2%} "
        f"({'+' if best.oos_accuracy >= baseline else ''}"
        f"{(best.oos_accuracy - baseline) * 100:.2f} pp)."
    )

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        per_fold.to_csv(RESULTS_DIR / "per_fold.csv", index=False)
        summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
        print(f"\nSaved to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
