# Stock Market Prediction using AI/ML

A walk-forward framework for next-day equity direction prediction, built so
that the two defects which inflate most results in this problem space are
hard to commit: **look-ahead bias** and **the missing baseline**.

The headline finding is deliberately unglamorous, and it is the point:

> On a synthetic random walk, none of the four models beats "always predict
> up". Gradient boosting lands at 49.4% out-of-sample directional accuracy
> against a 51.6% free baseline. That is the correct answer — and a pipeline
> that reports 95% accuracy on this problem is reporting a bug.

## Why this repo is shaped this way

Search GitHub for stock prediction and you will find a lot of notebooks
reporting 97% accuracy and an R² of 0.99. Almost all of them share one of
two defects:

| Defect | What it looks like | What this repo does |
|---|---|---|
| Look-ahead bias | A feature on day *t* is built from a centred rolling window, or the model predicts *today's* close from *today's* open | Every feature uses data from *t* or earlier; the target is the *t → t+1* return. Asserted by test, not by comment |
| Predicting price, not return | Next-day price is ~99% explained by today's price, so R² looks spectacular and means nothing | The target is the sign of the forward log return, where the base rate is ~51% |
| Random K-fold on a time series | Training on 2024 to predict 2023 | Expanding-window walk-forward with an embargo gap of `horizon` rows |
| No baseline | "72% accuracy" quoted with nothing to compare against | `always_up` and `persistence` are scored on identical folds and appear in every results table |

## Results

Run on the seeded synthetic series (`--source synthetic`), 5 walk-forward
folds, 1,935 usable rows, 20 features, 5 bps cost per position change:

| Model | OOS accuracy | σ across folds | Sharpe | Ann. return | Max drawdown |
|---|---|---|---|---|---|
| `always_up` (baseline) | **51.60%** | 0.026 | 0.498 | +8.66% | −25.0% |
| `persistence` (baseline) | 50.91% | 0.028 | −0.082 | −0.97% | −26.9% |
| `gradient_boosting` | 49.37% | 0.041 | −0.043 | −0.54% | −30.3% |
| `random_forest` | 48.19% | 0.030 | 0.016 | +0.21% | −28.6% |
| `logistic` | 47.77% | 0.021 | −0.279 | −3.55% | −38.7% |

Buy and hold over the same window: Sharpe 0.498, annualised return 8.66%.

Two things worth reading off that table. First, **every learned model
underperforms doing nothing**, which is what an unpredictable series should
produce. Second, the per-fold standard deviation (~0.03) is larger than the
gaps between models, so the ranking among the three learned models is noise.

Fold-level detail is printed by the runner and saved with `--save`.

## Quick start

```bash
git clone https://github.com/ADITYAKUMAR2025/STOCK-MARKET-PREDICTION-USING-AI-ML.git
cd STOCK-MARKET-PREDICTION-USING-AI-ML

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Reproduce the table above - fully offline, seeded
python scripts/run_experiment.py --source synthetic

# Run it on real data
python scripts/run_experiment.py --ticker ^NSEI --source yfinance --save
python scripts/run_experiment.py --ticker RELIANCE.NS --start 2018-01-01

# Run the test suite
pytest
```

With `--source auto` (the default) the loader tries Yahoo Finance and falls
back to synthetic prices with a loud warning if the network is unavailable,
so the pipeline never silently reports results from data you did not expect.

## Repository layout

```
src/stockpred/
  data.py         yfinance loader + seeded synthetic generator with vol regimes
  features.py     20 trailing-only features; the no-look-ahead contract lives here
  validate.py     expanding-window walk-forward splits with an embargo gap
  models.py       AlwaysUp, Persistence, logistic, gradient boosting, random forest
  backtest.py     long/flat equity curve, Sharpe, drawdown, turnover, costs
  experiment.py   the fold loop that produces the results table
scripts/
  run_experiment.py   CLI entry point
tests/
  test_features.py    look-ahead assertions, including a leakage canary
  test_validate.py    fold ordering, overlap, embargo
  test_backtest.py    backtest arithmetic against known answers
```

## The tests are the argument

`pytest` runs 17 tests. Four of them carry the methodological weight:

- **`test_features_do_not_depend_on_the_future`** multiplies the last 100
  rows of the price frame by 5 and asserts that not one earlier feature value
  changes. If any feature peeks forward, this fails.
- **`test_target_is_strictly_forward_looking`** checks the label against an
  independently computed `log(close).shift(-1) - log(close)`.
- **`test_leakage_canary_would_be_caught`** hands the model the answer as a
  feature and asserts accuracy above 95%. This guards the guard: it proves
  the evaluation is *capable* of detecting leakage, so a clean result means
  something.
- **`test_honest_features_are_not_predictive_on_a_random_walk`** asserts
  out-of-sample accuracy stays below 60% on synthetic data. A passing clean
  pipeline must fail to predict the unpredictable.

## Honest limitations

- **Daily direction only.** No intraday data, no order book, no horizons
  beyond the `--horizon` flag.
- **Long or flat, full size.** No shorting, no position sizing, no leverage,
  no portfolio construction across symbols.
- **Optimistic fills.** The backtest trades at the close the signal is
  computed from. A real implementation fills at the next open, which costs
  more than the 5 bps modelled here.
- **Price and volume features only.** No fundamentals, no macro series, no
  news or sentiment — all of which is where any real edge is more likely to
  live.
- **No multiple-testing correction.** Five models on one series is already
  enough to produce a flattering result by chance; treat any single positive
  number with suspicion.
- **Not investment advice.** This is research code for learning
  methodology, not a trading system.

## Roadmap

- Sequence models (LSTM / temporal CNN) added to the same fold harness, so
  they are scored against the same baselines rather than in isolation
- Deflated Sharpe ratio and White's reality check to correct for the number
  of strategies tried
- Cross-sectional ranking across a universe instead of single-symbol timing
- Regime conditioning on trailing volatility
- Next-open fills and a slippage model

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch and review workflow.

## License

[MIT](LICENSE)
