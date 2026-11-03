"""Walk-Forward Historical Validation Engine for Kronos.

Strict Non-Overlapping / Rolling Backtesting with Zero Look-Ahead Bias:
1. Slices historical candle context up to point T (no future information).
2. Generates Kronos Monte Carlo probabilistic path distribution.
3. Generates 4 Benchmark Baselines (Random Walk, Last Close, Historical Drift, Volatility Corridor).
4. Records forecast vs actual outcomes after horizon completion.
5. Computes all statistical evaluation metrics:
   - Directional Accuracy, Bull/Bear Hit Rate, Confusion Matrix
   - MAE, RMSE, MedAE, MAPE
   - Brier Score, Log Loss, Calibration Reliability Diagram, ECE
   - Prediction Interval Coverage (PICP 80%/50%) and Width (PINAW)
"""
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from kronos_app.core.kronos_engine import KronosMonteCarloEngine
from kronos_app.evaluation.baselines import BenchmarkBaselines
from kronos_app.evaluation.market_regime import detect_market_regime
from kronos_app.evaluation.metrics import (
    calculate_directional_metrics,
    calculate_price_error_metrics,
    calculate_probabilistic_calibration,
    calculate_interval_metrics
)

logger = logging.getLogger("Kronos.WalkForward")


class WalkForwardValidator:
    """Rigorous walk-forward validation harness ensuring zero look-ahead bias."""

    def __init__(
        self,
        engine: Optional[KronosMonteCarloEngine] = None,
        results_dir: Optional[Path] = None
    ):
        self.engine = engine or KronosMonteCarloEngine(model_type="mini")
        self.results_dir = results_dir or Path("d:/Kronos/data")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_validation(
        self,
        df: pd.DataFrame,
        timeframe: str = "M5",
        horizon: int = 12,
        context_len: int = 60,
        num_windows: int = 10,
        sample_count: int = 50,
        seed: int = 42,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """Executes walk-forward backtest over chronological slices.
        
        Args:
            df: Historical DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
            timeframe: 'M1' or 'M5'.
            horizon: Number of bars ahead to predict (e.g. 12 bars on M5 = 1 hour).
            context_len: Number of historical bars in input window.
            num_windows: Number of chronological walk-forward evaluation points.
            sample_count: Number of Monte Carlo samples per evaluation.
            seed: Base random seed for reproducibility.
            progress_callback: Optional callback(current_step, total_steps, message).
        """
        required_rows = context_len + (num_windows * horizon)
        if len(df) < required_rows:
            # Adjust num_windows if data is limited
            available_for_windows = len(df) - context_len - horizon
            if available_for_windows <= 0:
                raise ValueError(
                    f"Insufficient data: need at least {context_len + horizon + 1} candles, got {len(df)}"
                )
            stride = max(horizon, available_for_windows // num_windows)
            num_windows = max(3, available_for_windows // stride)
        else:
            stride = horizon

        # Ensure model is ready
        self.engine.load_model()

        # Storage for evaluation records
        eval_records: List[Dict[str, Any]] = []

        # Start index for rolling windows: we work backwards from latest completed data
        total_span = context_len + (num_windows * stride)
        start_base_idx = len(df) - total_span
        if start_base_idx < 0:
            start_base_idx = 0

        logger.info(
            f"Starting Walk-Forward on {timeframe} | Windows: {num_windows} | "
            f"Horizon: {horizon} bars | Context: {context_len} bars | Stride: {stride} bars"
        )

        for step in range(num_windows):
            T = start_base_idx + context_len + (step * stride)
            if T + horizon > len(df):
                break

            # STRICT NO LOOK-AHEAD:
            # Context is strictly up to bar T (index T is NOT included in context)
            context_slice = df.iloc[T - context_len : T].copy().reset_index(drop=True)
            # Future horizon is strictly from T to T + horizon
            future_slice = df.iloc[T : T + horizon].copy().reset_index(drop=True)

            t0_ts = str(context_slice["timestamp"].iloc[-1])
            target_ts = str(future_slice["timestamp"].iloc[-1])
            t0_price = float(context_slice["close"].iloc[-1])
            actual_price = float(future_slice["close"].iloc[-1])
            actual_return = actual_price - t0_price
            actual_outcome_bull = 1 if actual_return > 0 else 0

            if progress_callback:
                progress_callback(
                    step + 1,
                    num_windows,
                    f"Evaluating Window {step + 1}/{num_windows} @ {t0_ts} -> {target_ts}"
                )

            # 1. Detect market regime at point T
            regime = detect_market_regime(context_slice)

            # 2. Run Kronos Monte Carlo Forecast
            window_seed = seed + step
            forecast = self.engine.predict_monte_carlo(
                df=context_slice,
                pred_len=horizon,
                sample_count=sample_count,
                timeframe=timeframe,
                seed=window_seed,
                temperature=1.0,
                top_p=0.9
            )

            fan = forecast["fan_chart"]
            p50_series = np.array(fan["p50"])
            p10_series = np.array(fan["p10"])
            p90_series = np.array(fan["p90"])
            p25_series = np.array(fan["p25"])
            p75_series = np.array(fan["p75"])

            pred_median = float(p50_series[-1])
            pred_return = pred_median - t0_price

            # Model probability of up move at horizon
            horizons_data = forecast.get("horizons", [])
            last_h = horizons_data[-1] if horizons_data else {}
            p_up = float(last_h.get("model_p_up", 50.0)) / 100.0

            # 3. Generate Benchmark Baselines at exact same point T
            b_last_close = BenchmarkBaselines.last_close_baseline(context_slice, horizon)
            b_random_walk = BenchmarkBaselines.random_walk_baseline(context_slice, horizon)
            b_drift = BenchmarkBaselines.historical_drift_baseline(context_slice, horizon)
            b_vol = BenchmarkBaselines.volatility_corridor_baseline(context_slice, horizon)

            record = {
                "window_idx": step + 1,
                "t0_timestamp": t0_ts,
                "target_timestamp": target_ts,
                "t0_price": t0_price,
                "actual_price": actual_price,
                "actual_return": round(actual_return, 3),
                "actual_direction": "BULLISH" if actual_outcome_bull == 1 else "BEARISH",
                "market_regime": regime["trend_regime"],
                "volatility_regime": regime["volatility_regime"],
                # Kronos predictions
                "kronos_pred_median": round(pred_median, 3),
                "kronos_pred_return": round(pred_return, 3),
                "kronos_pred_direction": "BULLISH" if pred_return > 0 else "BEARISH",
                "kronos_p_up": round(p_up, 3),
                "kronos_p10": round(float(p10_series[-1]), 3),
                "kronos_p90": round(float(p90_series[-1]), 3),
                "kronos_p25": round(float(p25_series[-1]), 3),
                "kronos_p75": round(float(p75_series[-1]), 3),
                "actual_series": [round(float(v), 3) for v in future_slice["close"].values],
                "p50_series": [round(float(v), 3) for v in p50_series],
                "p10_series": [round(float(v), 3) for v in p10_series],
                "p90_series": [round(float(v), 3) for v in p90_series],
                # Baseline predictions
                "baseline_last_close_price": round(float(b_last_close["predicted_median"]), 3),
                "baseline_drift_price": round(float(b_drift["predicted_median"]), 3),
                "baseline_drift_return": round(float(b_drift["delta"]), 3),
                "baseline_vol_p10": round(float(b_vol["p10"][-1]), 3),
                "baseline_vol_p90": round(float(b_vol["p90"][-1]), 3),
            }
            eval_records.append(record)

        # Aggregate and Compute Comparative Metrics
        summary = self._compute_summary_metrics(eval_records, timeframe, horizon)

        # Save to disk
        out_path = self.results_dir / f"walk_forward_{timeframe.lower()}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Walk-forward validation results saved to {out_path}")

        return summary

    def _compute_summary_metrics(
        self,
        records: List[Dict[str, Any]],
        timeframe: str,
        horizon: int
    ) -> Dict[str, Any]:
        """Calculates full comparative metric suite between Kronos and baselines."""
        n = len(records)
        if n == 0:
            return {"success": False, "error": "No validation windows completed."}

        actual_returns = np.array([r["actual_return"] for r in records])
        actual_prices = np.array([r["actual_price"] for r in records])
        actual_outcomes = np.array([1 if r > 0 else 0 for r in actual_returns])

        kronos_returns = np.array([r["kronos_pred_return"] for r in records])
        kronos_prices = np.array([r["kronos_pred_median"] for r in records])
        kronos_probs = np.array([r["kronos_p_up"] for r in records])

        # 1. Kronos Metrics
        kronos_dir = calculate_directional_metrics(actual_returns, kronos_returns)
        kronos_price_err = calculate_price_error_metrics(actual_prices, kronos_prices)
        kronos_calib = calculate_probabilistic_calibration(actual_outcomes, kronos_probs, num_bins=5)

        # All-point prediction interval metrics across entire future series
        all_actuals = []
        all_p10 = []
        all_p90 = []
        all_p25 = []
        all_p75 = []
        for r in records:
            all_actuals.extend(r["actual_series"])
            all_p10.extend(r["p10_series"])
            all_p90.extend(r["p90_series"])
            # approximate 50% from p10/p90 if series not stored per step
            all_p25.extend([r["kronos_p25"]] * len(r["actual_series"]))
            all_p75.extend([r["kronos_p75"]] * len(r["actual_series"]))

        kronos_intervals = calculate_interval_metrics(
            np.array(all_actuals),
            np.array(all_p10),
            np.array(all_p90),
            np.array(all_p25),
            np.array(all_p75)
        )

        # 2. Baseline 1: Last Close (Zero Drift)
        lc_prices = np.array([r["baseline_last_close_price"] for r in records])
        lc_returns = np.zeros(n)
        lc_dir = calculate_directional_metrics(actual_returns, lc_returns)
        lc_price_err = calculate_price_error_metrics(actual_prices, lc_prices)

        # 3. Baseline 2: Random Walk (p=0.50)
        rw_probs = np.full(n, 0.50)
        rw_calib = calculate_probabilistic_calibration(actual_outcomes, rw_probs, num_bins=5)

        # 4. Baseline 3: Historical Drift (Linear Regression)
        drift_prices = np.array([r["baseline_drift_price"] for r in records])
        drift_returns = np.array([r["baseline_drift_return"] for r in records])
        drift_dir = calculate_directional_metrics(actual_returns, drift_returns)
        drift_price_err = calculate_price_error_metrics(actual_prices, drift_prices)

        # 5. Baseline 4: Volatility Corridor (ATR)
        vol_p10 = np.array([r["baseline_vol_p10"] for r in records])
        vol_p90 = np.array([r["baseline_vol_p90"] for r in records])
        vol_inside = (actual_prices >= vol_p10) & (actual_prices <= vol_p90)
        vol_picp = float(np.mean(vol_inside)) * 100.0

        # Comparative Win/Loss vs Baselines
        beats_last_close_mae = kronos_price_err["mae"] < lc_price_err["mae"]
        beats_drift_mae = kronos_price_err["mae"] < drift_price_err["mae"]
        beats_drift_da = kronos_dir["directional_accuracy"] >= drift_dir["directional_accuracy"]
        beats_rw_brier = kronos_calib["brier_score"] <= rw_calib["brier_score"]

        # Market Regime Breakdown
        regime_breakdown = {}
        for r in records:
            reg = r["market_regime"]
            if reg not in regime_breakdown:
                regime_breakdown[reg] = {"count": 0, "correct_dir": 0, "errors": []}
            regime_breakdown[reg]["count"] += 1
            is_correct = (r["actual_direction"] == r["kronos_pred_direction"])
            if is_correct:
                regime_breakdown[reg]["correct_dir"] += 1
            regime_breakdown[reg]["errors"].append(abs(r["kronos_pred_median"] - r["actual_price"]))

        for reg, data in regime_breakdown.items():
            cnt = data["count"]
            data["directional_accuracy"] = round((data["correct_dir"] / cnt) * 100.0, 2) if cnt > 0 else 0.0
            data["mean_mae"] = round(float(np.mean(data["errors"])), 3) if cnt > 0 else 0.0
            del data["errors"]

        return {
            "success": True,
            "timeframe": timeframe.upper(),
            "evaluated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "windows_count": n,
            "horizon_bars": horizon,
            "model_tested": self.engine.model_type,
            "kronos_metrics": {
                "directional": kronos_dir,
                "price_error": kronos_price_err,
                "probabilistic": kronos_calib,
                "intervals": kronos_intervals
            },
            "baseline_comparisons": {
                "last_close_persistence": {
                    "mae": lc_price_err["mae"],
                    "rmse": lc_price_err["rmse"],
                    "directional_accuracy": lc_dir["directional_accuracy"]
                },
                "random_walk": {
                    "brier_score": rw_calib["brier_score"],
                    "log_loss": rw_calib["log_loss"]
                },
                "historical_drift": {
                    "mae": drift_price_err["mae"],
                    "rmse": drift_price_err["rmse"],
                    "directional_accuracy": drift_dir["directional_accuracy"]
                },
                "volatility_corridor_atr": {
                    "picp_80_coverage_pct": round(vol_picp, 2)
                }
            },
            "benchmark_verdict": {
                "beats_last_close_mae": beats_last_close_mae,
                "beats_drift_mae": beats_drift_mae,
                "beats_drift_directional_accuracy": beats_drift_da,
                "beats_random_walk_brier": beats_rw_brier,
                "summary": (
                    f"Kronos MAE: {kronos_price_err['mae']} vs Last Close: {lc_price_err['mae']} "
                    f"| DA: {kronos_dir['directional_accuracy']}% vs Drift: {drift_dir['directional_accuracy']}%"
                )
            },
            "market_regime_breakdown": regime_breakdown,
            "window_records": records
        }
