"""Quantitative Evaluation Metrics for Financial Price and Probability Forecasting.

Covers:
- Directional Metrics (Accuracy, Hit Rate Bull/Bear, Confusion Matrix)
- Price Point Metrics (MAE, RMSE, MedAE, MAPE)
- Probabilistic Calibration (Brier Score, Log Loss, Calibration Curve Bins, ECE)
- Prediction Interval Quality (PICP 80%, PICP 50%, PINAW)
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def calculate_directional_metrics(
    actual_returns: np.ndarray,
    predicted_returns: np.ndarray
) -> Dict[str, Any]:
    """Calculates directional accuracy and confusion matrix.
    
    Args:
        actual_returns: Array of realized returns (e.g. (actual_close - T0_price)).
        predicted_returns: Array of forecasted returns (e.g. (pred_median - T0_price)).
    """
    actual_dir = (actual_returns > 0).astype(int)
    pred_dir = (predicted_returns > 0).astype(int)

    n = len(actual_dir)
    if n == 0:
        return {"directional_accuracy": 0.0, "total_evaluations": 0}

    correct = np.sum(actual_dir == pred_dir)
    da = float(correct / n) * 100.0

    # Confusion Matrix:
    # TP: Pred Bull, Actual Bull
    # FP: Pred Bull, Actual Bear
    # TN: Pred Bear, Actual Bear
    # FN: Pred Bear, Actual Bull
    tp = int(np.sum((pred_dir == 1) & (actual_dir == 1)))
    fp = int(np.sum((pred_dir == 1) & (actual_dir == 0)))
    tn = int(np.sum((pred_dir == 0) & (actual_dir == 0)))
    fn = int(np.sum((pred_dir == 0) & (actual_dir == 1)))

    bull_total = tp + fn
    bear_total = tn + fp

    bull_hit_rate = float(tp / bull_total * 100.0) if bull_total > 0 else 0.0
    bear_hit_rate = float(tn / bear_total * 100.0) if bear_total > 0 else 0.0

    return {
        "directional_accuracy": round(da, 2),
        "total_evaluations": n,
        "correct_predictions": int(correct),
        "bullish_hit_rate": round(bull_hit_rate, 2),
        "bearish_hit_rate": round(bear_hit_rate, 2),
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        }
    }


def calculate_price_error_metrics(
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray
) -> Dict[str, float]:
    """Calculates standard regression price forecast accuracy metrics."""
    actuals = np.asarray(actual_prices, dtype=np.float64)
    preds = np.asarray(predicted_prices, dtype=np.float64)

    errors = preds - actuals
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    med_ae = float(np.median(abs_errors))
    mape = float(np.mean(abs_errors / np.maximum(actuals, 1e-4))) * 100.0

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "median_ae": round(med_ae, 3),
        "mape_pct": round(mape, 3)
    }


def calculate_probabilistic_calibration(
    actual_outcomes: np.ndarray,
    predicted_probabilities: np.ndarray,
    num_bins: int = 10
) -> Dict[str, Any]:
    """Calculates Brier Score, Log Loss, and empirical calibration curve.
    
    Args:
        actual_outcomes: Binary 1 (event occurred, e.g. Bullish) or 0 (did not occur).
        predicted_probabilities: Model forecast probability [0.0, 1.0].
        num_bins: Number of probability buckets for reliability curve.
    """
    y_true = np.asarray(actual_outcomes, dtype=np.float64)
    p_pred = np.clip(np.asarray(predicted_probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)

    n = len(y_true)
    if n == 0:
        return {"brier_score": 1.0, "log_loss": 1.0, "calibration_bins": []}

    # Brier Score = (1/N) * sum((p - y)^2)
    # Reference: 0.25 is uninformative coin toss (p=0.5). Lower is strictly better.
    brier = float(np.mean((p_pred - y_true) ** 2))

    # Log Loss / Cross Entropy
    log_loss = float(-np.mean(y_true * np.log(p_pred) + (1.0 - y_true) * np.log(1.0 - p_pred)))

    # Reliability Diagram Bins
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins_data = []
    total_ece = 0.0

    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (p_pred >= low) & (p_pred < high) if i < num_bins - 1 else (p_pred >= low) & (p_pred <= high)
        bin_count = int(np.sum(mask))

        if bin_count > 0:
            avg_pred = float(np.mean(p_pred[mask]))
            empirical_freq = float(np.mean(y_true[mask]))
            calibration_gap = abs(avg_pred - empirical_freq)
            total_ece += (bin_count / n) * calibration_gap
        else:
            avg_pred = round((low + high) / 2.0, 2)
            empirical_freq = 0.0
            calibration_gap = 0.0

        bins_data.append({
            "bin_range": f"{int(low*100)}-{int(high*100)}%",
            "count": bin_count,
            "mean_predicted_prob": round(avg_pred * 100.0, 1),
            "observed_frequency": round(empirical_freq * 100.0, 1),
            "calibration_gap": round(calibration_gap * 100.0, 1)
        })

    return {
        "brier_score": round(brier, 4),
        "brier_score_skill_vs_coin_toss_pct": round((0.25 - brier) / 0.25 * 100.0, 2),
        "log_loss": round(log_loss, 4),
        "expected_calibration_error_pct": round(total_ece * 100.0, 2),
        "calibration_bins": bins_data
    }


def calculate_interval_metrics(
    actual_prices: np.ndarray,
    p10_series: np.ndarray,
    p90_series: np.ndarray,
    p25_series: np.ndarray,
    p75_series: np.ndarray
) -> Dict[str, float]:
    """Calculates coverage and normalized width for 80% and 50% prediction intervals."""
    actuals = np.asarray(actual_prices, dtype=np.float64)
    p10 = np.asarray(p10_series, dtype=np.float64)
    p90 = np.asarray(p90_series, dtype=np.float64)
    p25 = np.asarray(p25_series, dtype=np.float64)
    p75 = np.asarray(p75_series, dtype=np.float64)

    n = len(actuals)
    if n == 0:
        return {}

    # Prediction Interval Coverage Probability (PICP)
    inside_80 = (actuals >= p10) & (actuals <= p90)
    inside_50 = (actuals >= p25) & (actuals <= p75)

    picp_80 = float(np.mean(inside_80)) * 100.0
    picp_50 = float(np.mean(inside_50)) * 100.0

    # Prediction Interval Normalized Average Width (PINAW)
    price_range = max(1.0, float(np.max(actuals) - np.min(actuals)))
    pinaw_80 = float(np.mean(p90 - p10) / price_range) * 100.0
    pinaw_50 = float(np.mean(p75 - p25) / price_range) * 100.0

    return {
        "picp_80_coverage_pct": round(picp_80, 2),
        "picp_80_nominal_target": 80.0,
        "picp_80_error": round(picp_80 - 80.0, 2),
        "pinaw_80_width_pct": round(pinaw_80, 2),
        "picp_50_coverage_pct": round(picp_50, 2),
        "picp_50_nominal_target": 50.0,
        "picp_50_error": round(picp_50 - 50.0, 2),
        "pinaw_50_width_pct": round(pinaw_50, 2),
    }
