"""
stockpred — an honest baseline framework for equity return prediction.

The design goal of this package is not to maximise a reported accuracy
number. It is to make the two mistakes that inflate most published
stock-prediction results structurally hard to commit:

1. **Look-ahead bias.** Every feature is built from information available
   strictly at or before time t, and the target is the return realised
   between t and t+1. See `features.make_dataset`.
2. **Missing baseline.** Every model is scored against a persistence
   baseline on the same walk-forward folds, so "72% accuracy" can be read
   next to what you would get for free. See `models.PersistenceBaseline`.
"""

__version__ = "0.1.0"

__all__ = ["data", "features", "models", "validate", "backtest"]
