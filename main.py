"""Kronos Foundation Core v1.0 - Main Launcher for XAUUSD.
Launches the Google Chrome Frameless GUI with Dukascopy Ingestion and Generative Forecaster.
"""
import argparse
import logging
import sys
from pathlib import Path

# Set up clean logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

from kronos_app.service import KronosService
from kronos_app.gui.app import launch_gui


def main():
    parser = argparse.ArgumentParser(description="Kronos Foundation Core - XAUUSD Generative Foundation Engine")
    parser.add_argument("--headless", action="store_true", help="Run forecast in terminal without opening GUI")
    parser.add_argument("--model", type=str, default="mini", choices=["mini", "base"], help="Model engine: mini (4.1M, ultra-fast) or base (102.3M)")
    parser.add_argument("--timeframe", type=str, default="H1", choices=["M1", "M5", "M15", "H1", "H4", "D1"], help="Timeframe: M1, M5, M15, H1, H4, D1 (default: H1)")
    parser.add_argument("--pred-len", type=int, default=12, help="Forecast horizon bars (default: 12)")
    parser.add_argument("--samples", type=int, default=5, help="Number of Monte Carlo paths (default: 5)")
    args = parser.parse_args()

    service = KronosService()

    if args.headless:
        print(f"\n=======================================================")
        print(f"  KRONOS FOUNDATION CORE | XAUUSD (Headless Mode)     ")
        print(f"=======================================================")
        print(f"Engine: {args.model.upper()} | Timeframe: {args.timeframe} | Horizon: +{args.pred_len} Bars | Samples: {args.samples}")
        print("Fetching Dukascopy data and generating autoregressive forecast...")

        res = service.execute_forecast(
            model_type=args.model,
            timeframe=args.timeframe,
            pred_len=args.pred_len,
            sample_count=args.samples
        )

        if not res.get("success"):
            print("Forecast failed:", res.get("error"))
            return

        data = res["data"]
        print(f"\nLast Real Price: {data['last_price']:.3f} | Current ATR(14): {data['current_atr']:.3f}")
        print("\n--- MULTI-HORIZON PROJECTION SUMMARY ---")
        for h in data["horizons"]:
            print(f"[{h['horizon']}] Target: {h['target_time']}")
            print(f"  Median: {h['median_price']:.2f} (Delta: {h['delta_points']:+.2f} pts)")
            print(f"  Corridor (P10-P90): {h['range_low']:.2f} - {h['range_high']:.2f}")
            print(f"  Probability: {h['p_up']:.1f}% Bullish vs {h['p_down']:.1f}% Bearish -> {h['dominant']}\n")

        print("--- GHOST CANDLESTICKS (First 5 Bars) ---")
        for g in data["ghost_candles"][:5]:
            print(f"  Bar {g['step']} ({g['timestamp']}): O={g['open']:.2f}, H={g['high']:.2f}, L={g['low']:.2f}, C={g['close']:.2f} ({'BULL' if g['is_bullish'] else 'BEAR'})")

    else:
        # Launch Google Chrome Frameless GUI
        launch_gui(service)


if __name__ == "__main__":
    main()
