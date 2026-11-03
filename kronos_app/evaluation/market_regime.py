"""Market Regime and Volatility State Classifier for Spot Gold (XAUUSD).

Provides objective statistical classification of market conditions:
- Trend: TRENDING_BULLISH, TRENDING_BEARISH, or RANGING
- Volatility: HIGH_VOLATILITY, NORMAL_VOLATILITY, or LOW_VOLATILITY
Used solely for post-hoc segmentation and model diagnostics, NOT for trading signals.
"""
from typing import Any, Dict
import numpy as np
import pandas as pd


def detect_market_regime(df: pd.DataFrame, lookback: int = 40) -> Dict[str, Any]:
    """Analyzes recent candlestick structure to detect prevailing market state.
    
    Args:
        df: DataFrame with open, high, low, close columns.
        lookback: Number of recent candles to evaluate.
    """
    if len(df) < 20:
        return {
            "trend_regime": "INSUFFICIENT_DATA",
            "volatility_regime": "INSUFFICIENT_DATA",
            "adx": 0.0,
            "atr_ratio": 1.0
        }

    sub_df = df.iloc[-lookback:].copy()
    close = sub_df["close"].values
    high = sub_df["high"].values
    low = sub_df["low"].values

    # 1. Trend Direction & Slope via Linear Regression
    x = np.arange(len(close))
    slope, _ = np.polyfit(x, close, 1)
    # Normalized slope: basis points per bar
    norm_slope_bps = (slope / close[-1]) * 10000.0

    # 2. Volatility Analysis (ATR relative to baseline)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr_recent = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
    atr_baseline = np.mean(tr) if len(tr) > 0 else 1.0

    atr_ratio = float(atr_recent / max(1e-4, atr_baseline))

    if atr_ratio >= 1.30:
        vol_regime = "HIGH_VOLATILITY"
    elif atr_ratio <= 0.75:
        vol_regime = "LOW_VOLATILITY"
    else:
        vol_regime = "NORMAL_VOLATILITY"

    # 3. Trend vs Ranging (R-squared of linear fit)
    y_pred = slope * x + np.mean(close) - slope * np.mean(x)
    ss_tot = np.sum((close - np.mean(close)) ** 2)
    ss_res = np.sum((close - y_pred) ** 2)
    r_squared = float(1.0 - (ss_res / max(1e-5, ss_tot)))

    if r_squared >= 0.45 and abs(norm_slope_bps) >= 1.0:
        trend_regime = "TRENDING_BULLISH" if slope > 0 else "TRENDING_BEARISH"
    else:
        trend_regime = "RANGING"

    return {
        "trend_regime": trend_regime,
        "volatility_regime": vol_regime,
        "trend_r_squared": round(r_squared, 3),
        "normalized_slope_bps": round(norm_slope_bps, 2),
        "current_atr": round(float(atr_recent), 3),
        "atr_ratio_vs_baseline": round(atr_ratio, 2)
    }
