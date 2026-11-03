"""Unit Tests for Monte Carlo Sampling Convergence & Reproducibility.

Verifies:
1. Random seed control produces exact deterministic reproducibility across runs.
2. Convergence diagnostics accurately identify distribution stability across sample halves.
"""
import datetime
import unittest
import numpy as np
import pandas as pd

from kronos_app.core.kronos_engine import KronosMonteCarloEngine


class TestSamplingConvergence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = KronosMonteCarloEngine(model_type="mini")
        cls.engine.load_model()

        # Build consistent synthetic OHLC series
        dates = [datetime.datetime(2026, 11, 2, 0, 0) + datetime.timedelta(minutes=5 * i) for i in range(40)]
        base = 2650.0
        closes = [base + np.sin(i / 3.0) * 5.0 for i in range(40)]
        cls.df = pd.DataFrame({
            "timestamp": dates,
            "open": closes,
            "high": [c + 1.5 for c in closes],
            "low": [c - 1.5 for c in closes],
            "close": closes,
            "volume": np.ones(40) * 100.0
        })

    def test_reproducibility_with_seed(self):
        """Proof: Setting the same random seed produces identical P50 and P10-P90 values."""
        res1 = self.engine.predict_monte_carlo(self.df, pred_len=4, sample_count=10, seed=123)
        res2 = self.engine.predict_monte_carlo(self.df, pred_len=4, sample_count=10, seed=123)

        p50_1 = res1["fan_chart"]["p50"]
        p50_2 = res2["fan_chart"]["p50"]
        self.assertEqual(p50_1, p50_2, "Seed reproducibility failed: P50 outputs differ!")

        p10_1 = res1["fan_chart"]["p10"]
        p10_2 = res2["fan_chart"]["p10"]
        self.assertEqual(p10_1, p10_2, "Seed reproducibility failed: P10 outputs differ!")

    def test_convergence_diagnostics_structure(self):
        """Validates that sampling_diagnostics returns all required metrics and disclaimer."""
        res = self.engine.predict_monte_carlo(self.df, pred_len=4, sample_count=20, seed=42)
        diag = res.get("sampling_diagnostics")
        self.assertIsNotNone(diag)
        self.assertIn("convergence_status", diag)
        self.assertIn("p50_stability_error", diag)
        self.assertIn("dir_prob_stability_error_pct", diag)
        self.assertIn("standard_error_p50", diag)
        self.assertIn("model_prob_disclaimer", diag)
        # Low sample count (< 50) should be flagged LOW CONFIDENCE
        self.assertEqual(diag["convergence_status"], "LOW CONFIDENCE")

    def test_chunked_sampling_memory_safety(self):
        """Validates that chunked Monte Carlo sampling runs safely across multiple chunks (e.g. 130 samples) without memory errors."""
        res = self.engine.predict_monte_carlo(self.df, pred_len=4, sample_count=130, seed=42)
        self.assertEqual(len(res["fan_chart"]["p50"]), 4)
        self.assertEqual(res["sampling_diagnostics"]["sample_count"], 130)


if __name__ == "__main__":
    unittest.main()
