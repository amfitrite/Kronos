"""Stand-alone Walk-Forward Historical Validation Script for XAUUSD M1 and M5.

Executes rigorous walk-forward backtesting with ZERO look-ahead bias:
- XAUUSD M1: 15-bar horizon (15 minutes), 60-bar context
- XAUUSD M5: 12-bar horizon (1 hour), 60-bar context
- Evaluates Kronos against 4 benchmark baselines:
    1. Random Walk (p=0.50)
    2. Last Close (Zero Drift)
    3. Historical Drift (Linear Trend)
    4. Volatility Corridor (ATR)
- Outputs full statistical verification report and persists JSON artifacts to `data/`.
"""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kronos_app.data.dukascopy_feed import DukascopyFeed
from kronos_app.core.kronos_engine import KronosMonteCarloEngine
from kronos_app.evaluation.walk_forward import WalkForwardValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("WalkForwardRunner")


def format_table(headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    separator_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_lines = [
        " | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row))
        for row in rows
    ]
    return f"{header_line}\n{separator_line}\n" + "\n".join(row_lines)


def run_evaluation_for_timeframe(
    timeframe: str,
    horizon: int,
    num_windows: int = 5,
    sample_count: int = 50,
    seed: int = 42
):
    print("=" * 80)
    print(f"RUNNING WALK-FORWARD VALIDATION: XAUUSD {timeframe.upper()}")
    print(f"Settings: {num_windows} rolling windows | Horizon: {horizon} bars | MC Samples: {sample_count}")
    print("=" * 80)

    feed = DukascopyFeed()
    cache_path = feed.get_cache_path(timeframe)
    if not cache_path.exists():
        print(f"Cache not found for {timeframe}, downloading from Dukascopy...")
        df = feed.fetch_historical_data(timeframe=timeframe, force_refresh=True)
    else:
        df = pd.read_parquet(cache_path)

    print(f"Loaded {len(df)} candles for {timeframe}. Timestamp range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    engine = KronosMonteCarloEngine(model_type="mini")
    validator = WalkForwardValidator(engine=engine, results_dir=Path("d:/Kronos/data"))

    def progress(step, total, msg):
        print(f"  [{step}/{total}] {msg}")

    summary = validator.run_validation(
        df=df,
        timeframe=timeframe,
        horizon=horizon,
        context_len=60,
        num_windows=num_windows,
        sample_count=sample_count,
        seed=seed,
        progress_callback=progress
    )

    km = summary["kronos_metrics"]
    bc = summary["baseline_comparisons"]
    bv = summary["benchmark_verdict"]

    print("\n" + "-" * 80)
    print(f"EVALUATION RESULTS SUMMARY: XAUUSD {timeframe.upper()}")
    print("-" * 80)

    # 1. Directional Accuracy Table
    dir_headers = ["Metric / Model", "Kronos Mini", "Last Close", "Historical Drift", "Random Walk"]
    dir_rows = [
        [
            "Directional Accuracy (%)",
            f"{km['directional']['directional_accuracy']}%",
            f"{bc['last_close_persistence']['directional_accuracy']}%",
            f"{bc['historical_drift']['directional_accuracy']}%",
            "50.0% (Theoretical)"
        ],
        [
            "Bullish Hit Rate (%)",
            f"{km['directional']['bullish_hit_rate']}%",
            "N/A",
            "N/A",
            "50.0%"
        ],
        [
            "Bearish Hit Rate (%)",
            f"{km['directional']['bearish_hit_rate']}%",
            "N/A",
            "N/A",
            "50.0%"
        ]
    ]
    print("\n[DIRECTIONAL ACCURACY & HIT RATES]")
    print(format_table(dir_headers, dir_rows))

    # 2. Price Error Table
    err_headers = ["Price Error Metric", "Kronos Mini", "Last Close (Zero Drift)", "Historical Drift"]
    err_rows = [
        ["MAE ($)", f"${km['price_error']['mae']:.3f}", f"${bc['last_close_persistence']['mae']:.3f}", f"${bc['historical_drift']['mae']:.3f}"],
        ["RMSE ($)", f"${km['price_error']['rmse']:.3f}", f"${bc['last_close_persistence']['rmse']:.3f}", f"${bc['historical_drift']['rmse']:.3f}"],
        ["Median AE ($)", f"${km['price_error']['median_ae']:.3f}", "N/A", "N/A"],
        ["MAPE (%)", f"{km['price_error']['mape_pct']:.3f}%", "N/A", "N/A"]
    ]
    print("\n[POINT ESTIMATION PRICE ERRORS]")
    print(format_table(err_headers, err_rows))

    # 3. Probabilistic Calibration Table
    prob_headers = ["Calibration Metric", "Kronos Model Probability", "Random Walk (Coin Toss)"]
    prob_rows = [
        ["Brier Score (lower is better)", f"{km['probabilistic']['brier_score']:.4f}", f"{bc['random_walk']['brier_score']:.4f}"],
        ["Skill Score vs Coin Toss (%)", f"{km['probabilistic']['brier_score_skill_vs_coin_toss_pct']}%", "0.0%"],
        ["Log Loss", f"{km['probabilistic']['log_loss']:.4f}", f"{bc['random_walk']['log_loss']:.4f}"],
        ["Expected Calibration Error (ECE)", f"{km['probabilistic']['expected_calibration_error_pct']}%", "N/A"]
    ]
    print("\n[PROBABILISTIC CALIBRATION & BRIER SCORE]")
    print(format_table(prob_headers, prob_rows))

    # 4. Prediction Interval Table
    int_headers = ["Prediction Interval", "Nominal Target", "Actual Realized PICP", "Coverage Error", "PINAW Width"]
    int_rows = [
        [
            "80% Confidence Band (P10-P90)",
            "80.0%",
            f"{km['intervals']['picp_80_coverage_pct']}%",
            f"{km['intervals']['picp_80_error']:+.2f}%",
            f"{km['intervals']['pinaw_80_width_pct']:.2f}%"
        ],
        [
            "50% Confidence Band (P25-P75)",
            "50.0%",
            f"{km['intervals']['picp_50_coverage_pct']}%",
            f"{km['intervals']['picp_50_error']:+.2f}%",
            f"{km['intervals']['pinaw_50_width_pct']:.2f}%"
        ],
        [
            "Volatility Corridor (ATR Baseline)",
            "80.0%",
            f"{bc['volatility_corridor_atr']['picp_80_coverage_pct']}%",
            f"{bc['volatility_corridor_atr']['picp_80_coverage_pct'] - 80.0:+.2f}%",
            "N/A"
        ]
    ]
    print("\n[PREDICTION INTERVAL COVERAGE (PICP / PINAW)]")
    print(format_table(int_headers, int_rows))

    print("\n[BENCHMARK VERDICT SUMMARY]")
    print(f"  Summary: {bv['summary']}")
    print(f"  Beats Last Close MAE: {bv['beats_last_close_mae']}")
    print(f"  Beats Drift MAE:      {bv['beats_drift_mae']}")
    print(f"  Beats Drift DA:       {bv['beats_drift_directional_accuracy']}")
    print(f"  Beats Random Walk:    {bv['beats_random_walk_brier']}")
    print("=" * 80 + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Kronos Walk-Forward Validation")
    parser.add_argument("--windows", type=int, default=5, help="Number of rolling windows per timeframe")
    parser.add_argument("--samples", type=int, default=50, help="Monte Carlo sample count")
    parser.add_argument("--tf", type=str, default="both", choices=["m1", "m5", "both"], help="Timeframe to test")
    args = parser.parse_args()

    if args.tf in ("m5", "both"):
        # M5: 12 bars = 1 hour horizon
        run_evaluation_for_timeframe("M5", horizon=12, num_windows=args.windows, sample_count=args.samples)

    if args.tf in ("m1", "both"):
        # M1: 15 bars = 15 minutes horizon
        run_evaluation_for_timeframe("M1", horizon=15, num_windows=args.windows, sample_count=args.samples)


if __name__ == "__main__":
    main()
