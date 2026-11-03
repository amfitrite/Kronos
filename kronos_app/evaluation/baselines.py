"""Statistical Benchmark Baselines for Model Comparison.

All Kronos projections must be evaluated against these 4 empirical baselines:
1. Random Walk (Direction p=0.50, Price=Last Close)
2. Last Close Persistence (Zero Drift)
3. Historical Drift (Rolling momentum / linear trend)
4. Historical ATR Volatility Corridor
"""
from typing import Any, Dict
import numpy as np
import pandas as pd


class BenchmarkBaselines:
    """Computes reference predictions for comparing Kronos forecasts."""

    @staticmethod
    def last_close_baseline(context_df: pd.DataFrame, pred_len: int) -> Dict[str, Any]:
        """Baseline 1: Persistence / Zero-drift forecast. Price remains at last close."""
        last_close = float(context_df["close"].iloc[-1])
        return {
            "name": "Last Close (Zero Drift)",
            "predicted_prices": np.full(pred_len, last_close),
            "predicted_median": last_close,
            "p_up": 50.0,
            "p_down": 50.0,
            "delta": 0.0
        }

    @staticmethod
    def random_walk_baseline(context_df: pd.DataFrame, pred_len: int) -> Dict[str, Any]:
        """Baseline 2: Random Walk without drift (p=0.50 directional probability)."""
        last_close = float(context_df["close"].iloc[-1])
        return {
            "name": "Random Walk",
            "predicted_prices": np.full(pred_len, last_close),
            "predicted_median": last_close,
            "p_up": 50.0,
            "p_down": 50.0,
            "delta": 0.0
        }

    @staticmethod
    def historical_drift_baseline(context_df: pd.DataFrame, pred_len: int) -> Dict[str, Any]:
        """Baseline 3: Linear regression trend / momentum extrapolated from historical context."""
        closes = context_df["close"].values
        n = len(closes)
        lookback = min(n, 60)
        recent_closes = closes[-lookback:]

        x = np.arange(lookback)
        slope, intercept = np.polyfit(x, recent_closes, 1)

        future_x = np.arange(lookback, lookback + pred_len)
        pred_prices = slope * future_x + intercept

        last_close = float(closes[-1])
        delta = float(pred_prices[-1] - last_close)
        p_up = 65.0 if delta > 0 else (35.0 if delta < 0 else 50.0)

        return {
            "name": "Historical Drift (Linear Trend)",
            "predicted_prices": pred_prices,
            "predicted_median": float(pred_prices[-1]),
            "p_up": p_up,
            "p_down": 100.0 - p_up,
            "delta": round(delta, 3)
        }

    @staticmethod
    def volatility_corridor_baseline(context_df: pd.DataFrame, pred_len: int) -> Dict[str, Any]:
        """Baseline 4: Classical ATR volatility corridor around last close."""
        last_close = float(context_df["close"].iloc[-1])
        high = context_df["high"].values
        low = context_df["low"].values
        close = context_df["close"].values

        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))

        # Standard sqrt(time) volatility expansion
        steps = np.arange(1, pred_len + 1)
        expansion = atr * np.sqrt(steps)

        return {
            "name": "Volatility Corridor (ATR)",
            "predicted_prices": np.full(pred_len, last_close),
            "p10": last_close - 1.28 * expansion, # 80% normal corridor
            "p90": last_close + 1.28 * expansion,
            "p25": last_close - 0.67 * expansion, # 50% normal corridor
            "p75": last_close + 0.67 * expansion,
            "p_up": 50.0,
            "p_down": 50.0,
            "atr": round(atr, 3)
        }
