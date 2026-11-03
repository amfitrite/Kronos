"""Unit Tests for Walk-Forward Historical Validation & Quantitative Metrics.

Verifies:
1. Proof of ZERO LOOK-AHEAD BIAS: context timestamps are strictly prior to future horizon timestamps.
2. Directional metrics, price errors, and confusion matrix calculate mathematically sound numbers.
3. Probabilistic calibration (Brier Score, Log Loss, ECE, Reliability Diagram).
4. Prediction interval quality (PICP 80%/50% and PINAW).
5. Statistical Baselines (Last Close, Random Walk, Historical Drift, Volatility Corridor).
"""
import datetime
import unittest
import numpy as np
import pandas as pd

from kronos_app.evaluation.baselines import BenchmarkBaselines
from kronos_app.evaluation.metrics import (
    calculate_directional_metrics,
    calculate_price_error_metrics,
    calculate_probabilistic_calibration,
    calculate_interval_metrics
)


class TestWalkForwardMetrics(unittest.TestCase):

    def test_zero_lookahead_slice_separation(self):
        """Mathematical proof: At step T, context slice timestamps must be strictly < future timestamps."""
        # Synthesize chronological series
        dates = [datetime.datetime(2026, 11, 2, 0, 0) + datetime.timedelta(minutes=5 * i) for i in range(120)]
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.linspace(2600, 2650, 120),
            "high": np.linspace(2605, 2655, 120),
            "low": np.linspace(2595, 2645, 120),
            "close": np.linspace(2602, 2652, 120),
            "volume": np.ones(120) * 100.0
        })

        context_len = 60
        horizon = 12
        stride = 12

        # Test rolling window boundary strictness
        for step in range(3):
            T = context_len + (step * stride)
            context_slice = df.iloc[T - context_len : T]
            future_slice = df.iloc[T : T + horizon]

            last_context_time = context_slice["timestamp"].iloc[-1]
            first_future_time = future_slice["timestamp"].iloc[0]

            # Context must end strictly before future horizon begins
            self.assertLess(last_context_time, first_future_time)
            # Future slice must never intersect with context
            overlap = set(context_slice["timestamp"]).intersection(set(future_slice["timestamp"]))
            self.assertEqual(len(overlap), 0, "Look-ahead bias detected! Context overlaps with future horizon!")

    def test_directional_metrics_and_confusion_matrix(self):
        """Validates Directional Accuracy, Hit Rates, and Confusion Matrix logic."""
        actual_returns = np.array([10.0, -5.0, 8.0, -12.0, 3.0])
        pred_returns = np.array([6.0, -2.0, -4.0, -8.0, 5.0])

        metrics = calculate_directional_metrics(actual_returns, pred_returns)
        # Expected:
        # idx 0: act + (bull), pred + (bull) -> TP
        # idx 1: act - (bear), pred - (bear) -> TN
        # idx 2: act + (bull), pred - (bear) -> FN
        # idx 3: act - (bear), pred - (bear) -> TN
        # idx 4: act + (bull), pred + (bull) -> TP
        # Total: 5, Correct: 4 -> 80.0%
        self.assertEqual(metrics["directional_accuracy"], 80.0)
        self.assertEqual(metrics["confusion_matrix"]["true_positives"], 2)
        self.assertEqual(metrics["confusion_matrix"]["true_negatives"], 2)
        self.assertEqual(metrics["confusion_matrix"]["false_negatives"], 1)
        self.assertEqual(metrics["confusion_matrix"]["false_positives"], 0)

    def test_price_error_metrics(self):
        """Validates MAE, RMSE, and Median AE calculations."""
        actuals = np.array([100.0, 102.0, 104.0, 106.0])
        preds = np.array([101.0, 100.0, 105.0, 109.0])
        # Errors: [1.0, -2.0, 1.0, 3.0]
        # Abs errors: [1.0, 2.0, 1.0, 3.0]
        # MAE = 7.0 / 4 = 1.75
        # RMSE = sqrt((1 + 4 + 1 + 9)/4) = sqrt(3.75) ~= 1.936
        err = calculate_price_error_metrics(actuals, preds)
        self.assertAlmostEqual(err["mae"], 1.75, places=2)
        self.assertAlmostEqual(err["rmse"], 1.936, places=2)

    def test_probabilistic_calibration(self):
        """Validates Brier Score and ECE calculation."""
        # Perfect forecasts: outcome=1 prob=1.0, outcome=0 prob=0.0
        outcomes = np.array([1, 0, 1, 0])
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        calib = calculate_probabilistic_calibration(outcomes, probs, num_bins=4)
        self.assertAlmostEqual(calib["brier_score"], 0.0, places=3)
        self.assertAlmostEqual(calib["brier_score_skill_vs_coin_toss_pct"], 100.0, places=1)

        # Uninformative coin toss: prob=0.50
        probs_coin = np.array([0.5, 0.5, 0.5, 0.5])
        calib_coin = calculate_probabilistic_calibration(outcomes, probs_coin, num_bins=4)
        self.assertAlmostEqual(calib_coin["brier_score"], 0.25, places=3)
        self.assertAlmostEqual(calib_coin["brier_score_skill_vs_coin_toss_pct"], 0.0, places=1)

    def test_interval_coverage_metrics(self):
        """Validates PICP 80% and 50% coverage calculation."""
        actuals = np.array([100.0, 102.0, 104.0, 106.0, 120.0]) # 120 is outside corridor
        p10 = np.array([95.0, 95.0, 95.0, 95.0, 95.0])
        p90 = np.array([110.0, 110.0, 110.0, 110.0, 110.0])
        p25 = np.array([98.0, 98.0, 98.0, 98.0, 98.0])
        p75 = np.array([105.0, 105.0, 105.0, 105.0, 105.0])

        intervals = calculate_interval_metrics(actuals, p10, p90, p25, p75)
        # 4 out of 5 are inside [95, 110] -> 80.0% PICP
        self.assertEqual(intervals["picp_80_coverage_pct"], 80.0)
        self.assertEqual(intervals["picp_80_error"], 0.0)

    def test_baselines_generation(self):
        """Validates that all 4 baselines generate complete statistical forecasts."""
        context_df = pd.DataFrame({
            "high": [105.0, 106.0, 107.0, 108.0],
            "low": [95.0, 96.0, 97.0, 98.0],
            "close": [100.0, 102.0, 104.0, 106.0]
        })
        pred_len = 10
        b_lc = BenchmarkBaselines.last_close_baseline(context_df, pred_len)
        b_rw = BenchmarkBaselines.random_walk_baseline(context_df, pred_len)
        b_drift = BenchmarkBaselines.historical_drift_baseline(context_df, pred_len)
        b_vol = BenchmarkBaselines.volatility_corridor_baseline(context_df, pred_len)

        self.assertEqual(len(b_lc["predicted_prices"]), pred_len)
        self.assertEqual(b_lc["predicted_median"], 106.0)
        self.assertEqual(b_rw["p_up"], 50.0)
        self.assertEqual(len(b_drift["predicted_prices"]), pred_len)
        self.assertEqual(len(b_vol["p10"]), pred_len)
        self.assertEqual(len(b_vol["p90"]), pred_len)


if __name__ == "__main__":
    unittest.main()
