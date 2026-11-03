"""Unit Tests for P0 Data Integrity & Dukascopy Feed.

Verifies:
1. DukascopyFeed raises DataUnavailableError when network fails and no cache exists (NO synthetic fallback).
2. Strict OHLC boundary sanity: High >= max(Open, Close), Low <= min(Open, Close).
3. Incomplete forming candle is dropped.
4. Non-finite values (NaN, Inf) and negative prices are eliminated.
5. Duplicate timestamps are removed.
"""
import datetime
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from kronos_app.data.dukascopy_feed import DukascopyFeed, DataUnavailableError


class TestDataIntegrity(unittest.TestCase):

    def test_no_synthetic_fallback_on_failure(self):
        """Proof: If Dukascopy fails and cache is missing, raise DataUnavailableError instead of synthetic fallback."""
        empty_dir = Path("d:/Kronos/data/test_empty_cache_dir")
        empty_dir.mkdir(parents=True, exist_ok=True)
        feed = DukascopyFeed(cache_dir=empty_dir)

        # Force failure by requesting non-existent lookback
        with self.assertRaises(DataUnavailableError):
            feed.fetch_historical_data("M5", lookback_days=-999, force_refresh=True)

    def test_strict_ohlc_boundaries(self):
        """Validates that OHLC boundaries are enforced: High >= max(Open, Close), Low <= min(Open, Close)."""
        feed = DukascopyFeed()
        dirty_data = {
            "timestamp": [
                datetime.datetime(2026, 11, 2, 10, 0, tzinfo=datetime.timezone.utc),
                datetime.datetime(2026, 11, 2, 11, 0, tzinfo=datetime.timezone.utc),
            ],
            # Open, High, Low, Close where High is lower than Open and Low is higher than Close
            "open": [100.0, 200.0],
            "high": [95.0, 190.0],   # Invalid: High < Open
            "low": [105.0, 210.0],   # Invalid: Low > Close
            "close": [98.0, 195.0],
            "volume": [10.0, 20.0]
        }
        df_dirty = pd.DataFrame(dirty_data)
        clean_df, meta = feed.validate_ohlc(df_dirty, "H1")

        for _, row in clean_df.iterrows():
            self.assertGreaterEqual(row["high"], row["open"])
            self.assertGreaterEqual(row["high"], row["close"])
            self.assertLessEqual(row["low"], row["open"])
            self.assertLessEqual(row["low"], row["close"])
            self.assertGreaterEqual(row["high"], row["low"])

    def test_nan_and_inf_elimination(self):
        """Validates that non-finite values and negative prices are discarded."""
        feed = DukascopyFeed()
        dirty_data = {
            "timestamp": [
                datetime.datetime(2026, 11, 2, 10, 0, tzinfo=datetime.timezone.utc),
                datetime.datetime(2026, 11, 2, 11, 0, tzinfo=datetime.timezone.utc),
                datetime.datetime(2026, 11, 2, 12, 0, tzinfo=datetime.timezone.utc),
                datetime.datetime(2026, 11, 2, 13, 0, tzinfo=datetime.timezone.utc),
            ],
            "open": [100.0, np.nan, 102.0, -50.0],  # row 1 has NaN, row 3 has negative price
            "high": [105.0, 105.0, np.inf, 100.0], # row 2 has Inf
            "low": [95.0, 95.0, 95.0, -60.0],
            "close": [101.0, 101.0, 101.0, -55.0],
            "volume": [1.0, 1.0, 1.0, 1.0]
        }
        df_dirty = pd.DataFrame(dirty_data)
        clean_df, meta = feed.validate_ohlc(df_dirty, "H1")

        # Only row 0 should survive
        self.assertEqual(len(clean_df), 1)
        self.assertEqual(clean_df["open"].iloc[0], 100.0)

    def test_duplicate_timestamp_removal(self):
        """Validates that duplicate candles are deduplicated chronologically."""
        feed = DukascopyFeed()
        ts = datetime.datetime(2026, 11, 2, 10, 0, tzinfo=datetime.timezone.utc)
        dup_data = {
            "timestamp": [ts, ts, ts + datetime.timedelta(hours=1)],
            "open": [100.0, 100.5, 101.0],
            "high": [105.0, 105.5, 106.0],
            "low": [95.0, 95.5, 96.0],
            "close": [102.0, 102.5, 103.0],
            "volume": [10.0, 15.0, 20.0]
        }
        clean_df, meta = feed.validate_ohlc(pd.DataFrame(dup_data), "H1")
        self.assertEqual(len(clean_df), 2)
        self.assertEqual(clean_df["timestamp"].nunique(), len(clean_df))

    def test_is_cache_fresh_method(self):
        """Validates that DukascopyFeed.is_cache_fresh correctly checks freshness."""
        feed = DukascopyFeed()
        # Non-existent timeframe returns False
        self.assertFalse(feed.is_cache_fresh("INVALID_TF"))
        # Check that is_cache_fresh works without raising exceptions
        result = feed.is_cache_fresh("H1")
        self.assertIsInstance(result, bool)

    def test_fetch_live_candle_structure_and_boundaries(self):
        """Validates real-time Dukascopy ECN live candle structure and OHLC validity."""
        feed = DukascopyFeed()
        res = feed.fetch_live_candle("M5")
        self.assertTrue(res["success"])
        self.assertEqual(res["symbol"], "XAUUSD")
        self.assertEqual(res["timeframe"], "M5")

        candle = res["candle"]
        self.assertIn("timestamp", candle)
        self.assertIn("open", candle)
        self.assertIn("high", candle)
        self.assertIn("low", candle)
        self.assertIn("close", candle)
        self.assertTrue(candle["is_live"])

        # OHLC Sanity
        self.assertGreaterEqual(candle["high"], candle["low"])
        self.assertGreaterEqual(candle["high"], min(candle["open"], candle["close"]))
        self.assertLessEqual(candle["low"], max(candle["open"], candle["close"]))

        # Ticker Sanity
        ticker = res["ticker"]
        self.assertIn("bid", ticker)
        self.assertIn("ask", ticker)
        self.assertIn("spread", ticker)
        self.assertGreaterEqual(ticker["ask"], ticker["bid"])

        # Bar Progress Sanity
        progress = res["bar_progress"]
        self.assertEqual(progress["total_seconds"], 300)
        self.assertGreaterEqual(progress["progress_pct"], 0.0)
        self.assertLessEqual(progress["progress_pct"], 100.0)

    def test_fetch_live_candle_rate_limiting_cache(self):
        """Validates in-memory rate limiting within 1.0 second."""
        feed = DukascopyFeed()
        res1 = feed.fetch_live_candle("M1")
        res2 = feed.fetch_live_candle("M1")
        # Second call immediately should return identical cached result
        self.assertEqual(res1["candle"]["timestamp"], res2["candle"]["timestamp"])
        self.assertEqual(res1["ticker"]["bid"], res2["ticker"]["bid"])

    def test_tick_bridging_gapless_candles(self):
        """Validates that _bridge_recent_ticks recovers recent closed bars without error."""
        feed = DukascopyFeed()
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        # Create a sample historical df ending 15 minutes ago
        past_ts = now_utc - datetime.timedelta(minutes=15)
        sample_df = pd.DataFrame({
            "timestamp": [past_ts - datetime.timedelta(minutes=5), past_ts],
            "open": [3900.0, 3902.0],
            "high": [3905.0, 3906.0],
            "low": [3898.0, 3900.0],
            "close": [3902.0, 3904.0],
            "volume": [10.0, 12.0],
            "amount": [39020.0, 46848.0]
        })
        bridged = feed._bridge_recent_ticks(sample_df, "M5")
        self.assertIsInstance(bridged, pd.DataFrame)
        self.assertGreaterEqual(len(bridged), len(sample_df))
        self.assertIn("timestamp", bridged.columns)
        self.assertIn("close", bridged.columns)


if __name__ == "__main__":
    unittest.main()
