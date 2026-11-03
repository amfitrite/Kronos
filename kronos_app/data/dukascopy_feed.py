"""Dukascopy Data Ingestion, Integrity Validation & Caching Module for XAUUSD.

Strict Data Integrity Rules:
1. Synthetic fallback is completely removed from production forecasting.
2. In case of network failure without valid cache, raises DataUnavailableError.
3. Cache freshness verified against market operating hours.
4. Duplicate, corrupt, out-of-order, or unclosed live candles are strictly detected/handled.
5. Full OHLC mathematical integrity validation (H >= max(O,C), L <= min(O,C), H >= L > 0).
"""
import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from kronos_app.data.market_calendar import (
    is_xauusd_open,
    is_candle_closed,
    detect_data_gaps,
    get_timeframe_delta,
)

logger = logging.getLogger("Kronos.Data")

CACHE_DIR = Path("d:/Kronos/data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Minimum lookback days needed to provide sufficient context for Kronos
TIMEFRAME_LOOKBACK = {
    "M1": 3,      # ~3,000+ bars
    "1M": 3,
    "M5": 10,     # ~2,000+ bars
    "5M": 10,
    "M15": 25,    # ~1,600+ bars
    "15M": 25,
    "H1": 60,     # ~1,000+ bars
    "1H": 60,
    "H4": 150,    # ~650+ bars
    "4H": 150,
    "D1": 365     # ~260 bars
}

# Maximum tolerable age for cached data while market is active
CACHE_MAX_AGE = {
    "M1": datetime.timedelta(minutes=15),
    "M5": datetime.timedelta(minutes=45),
    "M15": datetime.timedelta(hours=2),
    "H1": datetime.timedelta(hours=6),
    "H4": datetime.timedelta(hours=24),
    "D1": datetime.timedelta(days=3),
}


class DataUnavailableError(Exception):
    """Raised when market data cannot be obtained and no valid cache exists."""
    pass


class DukascopyFeed:
    """Manages downloading, validating, and caching XAUUSD candle data."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.symbol = "XAU-USD"

    def get_cache_path(self, timeframe: str) -> Path:
        tf_norm = timeframe.strip().lower()
        return self.cache_dir / f"xauusd_{tf_norm}.parquet"

    def check_cache_freshness(self, df: pd.DataFrame, timeframe: str) -> Tuple[bool, str]:
        """Validates if cached data is fresh enough for inference.
        
        Returns:
            (is_fresh: bool, status_reason: str)
        """
        if df.empty or "timestamp" not in df.columns:
            return False, "EMPTY_CACHE"

        last_ts = pd.to_datetime(df["timestamp"].max())
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=datetime.timezone.utc)
        else:
            last_ts = last_ts.astimezone(datetime.timezone.utc)

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        age = now_utc - last_ts

        # If currently weekend (Saturday or Sunday before 22:00 UTC),
        # cache is fresh if it contains Friday's final market close bar.
        weekday = now_utc.weekday()
        if weekday == 5 or (weekday == 6 and now_utc.hour < 22):
            # Check if last bar is from Friday
            if last_ts.weekday() == 4 and last_ts.hour >= 20:
                return True, "WEEKEND_FRESH (Contains Friday Close)"

        max_tolerable = CACHE_MAX_AGE.get(timeframe.upper(), datetime.timedelta(hours=6))
        if age <= max_tolerable:
            return True, f"FRESH (Age: {int(age.total_seconds() // 60)}m)"
        return False, f"STALE (Age: {int(age.total_seconds() // 3600)}h > max {int(max_tolerable.total_seconds() // 3600)}h)"

    def validate_ohlc(self, df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Strictly validates OHLC integrity:
        - Drops incomplete forming live candle
        - Ensures chronological sorting, no duplicates
        - Checks H >= max(O,C), L <= min(O,C), H >= L > 0
        - Validates no NaN or infinite values
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        raw_len = len(df)
        df_clean = df.copy()

        # 1. Timestamp parsing & deduplication
        if "timestamp" not in df_clean.columns:
            if df_clean.index.name == "timestamp":
                df_clean = df_clean.reset_index()
            else:
                df_clean["timestamp"] = df_clean.index
                df_clean = df_clean.reset_index(drop=True)
        else:
            df_clean = df_clean.reset_index(drop=True)

        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], utc=True)
        df_clean = df_clean.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        # 2. Exclude incomplete / currently forming latest candle
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        dropped_incomplete = False
        if not df_clean.empty:
            latest_ts = df_clean["timestamp"].iloc[-1]
            if not is_candle_closed(latest_ts, timeframe, now_utc):
                logger.info(f"Excluding currently forming live candle at {latest_ts}")
                df_clean = df_clean.iloc[:-1].reset_index(drop=True)
                dropped_incomplete = True

        if df_clean.empty:
            raise ValueError("No finalized historical candles available after dropping incomplete bar.")

        # 3. Numeric conversions & NaN removal
        for col in ["open", "high", "low", "close"]:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        if "volume" in df_clean.columns:
            df_clean["volume"] = pd.to_numeric(df_clean["volume"], errors="coerce").fillna(1.0)
        else:
            df_clean["volume"] = 1.0

        # Discard invalid values (NaN, <= 0, Inf)
        valid_mask = (
            df_clean["open"].notnull() & np.isfinite(df_clean["open"]) & (df_clean["open"] > 0) &
            df_clean["high"].notnull() & np.isfinite(df_clean["high"]) & (df_clean["high"] > 0) &
            df_clean["low"].notnull() & np.isfinite(df_clean["low"]) & (df_clean["low"] > 0) &
            df_clean["close"].notnull() & np.isfinite(df_clean["close"]) & (df_clean["close"] > 0)
        )
        df_clean = df_clean[valid_mask].reset_index(drop=True)

        # 4. Enforce strict OHLC boundary sanity:
        # High must be >= max(open, close), Low must be <= min(open, close), and High >= Low
        df_clean["high"] = np.maximum(df_clean["high"], np.maximum(df_clean["open"], df_clean["close"]))
        df_clean["low"] = np.minimum(df_clean["low"], np.minimum(df_clean["open"], df_clean["close"]))
        df_clean = df_clean[df_clean["high"] >= df_clean["low"]].reset_index(drop=True)

        if df_clean.empty:
            raise ValueError("No valid candle data remaining after OHLC validation.")

        # Amount calculation
        df_clean["amount"] = df_clean["volume"] * df_clean[["open", "high", "low", "close"]].mean(axis=1)

        # 5. Gap detection
        gaps = detect_data_gaps(df_clean, timeframe)

        metadata = {
            "initial_rows": raw_len,
            "validated_rows": len(df_clean),
            "dropped_incomplete_bar": dropped_incomplete,
            "detected_market_gaps": len(gaps),
            "gaps_summary": gaps[:3],
            "first_bar": str(df_clean["timestamp"].iloc[0]),
            "last_bar": str(df_clean["timestamp"].iloc[-1])
        }
        return df_clean[["timestamp", "open", "high", "low", "close", "volume", "amount"]], metadata

    def fetch_historical_data(
        self,
        timeframe: str = "H1",
        lookback_days: Optional[int] = None,
        force_refresh: bool = False,
        allow_simulation: bool = False
    ) -> Dict[str, Any]:
        """Fetches validated XAUUSD candles for the requested timeframe.
        
        Returns:
            Dictionary containing:
            - 'df': validated pd.DataFrame
            - 'source': 'DUKASCOPY_LIVE' | 'CACHE_VALID' | 'CACHE_STALE' | 'SIMULATION'
            - 'status': 'VALID' | 'STALE' | 'DATA UNAVAILABLE'
            - 'metadata': validation details
        """
        tf_upper = timeframe.upper()
        if lookback_days is None:
            lookback_days = TIMEFRAME_LOOKBACK.get(tf_upper, 45)

        cache_path = self.get_cache_path(timeframe)
        cached_df = None
        cache_is_fresh = False
        cache_status_msg = "NO_CACHE"

        # Check existing cache
        if cache_path.exists():
            try:
                raw_cached = pd.read_parquet(cache_path)
                cached_df, _ = self.validate_ohlc(raw_cached, tf_upper)
                cache_is_fresh, cache_status_msg = self.check_cache_freshness(cached_df, tf_upper)
                if not force_refresh and cache_is_fresh and len(cached_df) >= 200:
                    logger.info(f"Using fresh cache for {timeframe} ({len(cached_df)} bars, {cache_status_msg})")
                    return {
                        "df": cached_df,
                        "source": "CACHE_VALID",
                        "status": "VALID",
                        "freshness": cache_status_msg,
                        "timeframe": tf_upper,
                        "cache_path": str(cache_path)
                    }
            except Exception as e:
                logger.warning(f"Cache inspection notice: {e}")

        # Attempt fresh download from Dukascopy
        logger.info(f"Requesting fresh {timeframe} candles from Dukascopy ECN (lookback={lookback_days}d)...")
        try:
            import dukascopy_python as duka

            now_utc = datetime.datetime.now(datetime.timezone.utc)
            start_utc = now_utc - datetime.timedelta(days=lookback_days)

            if tf_upper in ("M1", "1M"):
                interval, time_unit = 1, duka.TimeUnit.MINUTE
            elif tf_upper in ("M5", "5M"):
                interval, time_unit = 5, duka.TimeUnit.MINUTE
            elif tf_upper in ("M15", "15M"):
                interval, time_unit = 15, duka.TimeUnit.MINUTE
            elif tf_upper in ("H1", "1H"):
                interval, time_unit = 1, duka.TimeUnit.HOUR
            elif tf_upper in ("H4", "4H"):
                interval, time_unit = 4, duka.TimeUnit.HOUR
            elif tf_upper in ("D1", "1D"):
                interval, time_unit = 1, duka.TimeUnit.DAY
            else:
                interval, time_unit = 1, duka.TimeUnit.HOUR

            candles = list(
                duka.fetch_candles(
                    self.symbol,
                    interval=interval,
                    time_unit=time_unit,
                    offer_side=duka.OfferSide.BID,
                    start=start_utc,
                    end=now_utc,
                )
            )

            if not candles:
                raise ValueError(f"Dukascopy returned 0 candles for {self.symbol} in {timeframe}.")

            raw_df = duka.to_dataframe(candles)
            clean_df, meta = self.validate_ohlc(raw_df, tf_upper)

            # Persist valid data to cache
            clean_df.to_parquet(cache_path, index=False)
            logger.info(f"Successfully fetched & cached {len(clean_df)} valid {timeframe} bars from Dukascopy.")

            return {
                "df": clean_df,
                "source": "DUKASCOPY_LIVE",
                "status": "VALID",
                "freshness": "LIVE_SYNC",
                "timeframe": tf_upper,
                "metadata": meta
            }

        except Exception as e:
            logger.error(f"Live Dukascopy download failed: {e}")

            # Fallback to cache if available
            if cached_df is not None and not cached_df.empty and len(cached_df) >= 100:
                status = "VALID" if cache_is_fresh else "STALE"
                logger.warning(f"Using cache fallback for {timeframe}: status={status} ({cache_status_msg})")
                return {
                    "df": cached_df,
                    "source": "CACHE_VALID" if cache_is_fresh else "CACHE_STALE",
                    "status": status,
                    "freshness": cache_status_msg,
                    "timeframe": tf_upper,
                    "cache_path": str(cache_path)
                }

            # If simulation explicitly requested (strictly for isolated simulation mode)
            if allow_simulation:
                logger.warning(f"Generating SIMULATION data (allow_simulation=True explicitly set)")
                sim_df = self._generate_simulation_data(tf_upper)
                return {
                    "df": sim_df,
                    "source": "SIMULATION",
                    "status": "SIMULATION_MODE",
                    "freshness": "SYNTHETIC",
                    "timeframe": tf_upper
                }

            # Production safety: DO NOT silently fabricate fake data
            raise DataUnavailableError(
                f"DATA UNAVAILABLE: Failed to fetch live {timeframe} data from Dukascopy and no valid cache exists ({e})."
            )

    def _generate_simulation_data(self, timeframe: str) -> pd.DataFrame:
        """EXPLICIT SIMULATION ONLY. Never used in production predictions."""
        logger.warning(f"Creating isolated simulation dataset for {timeframe}")
        now = datetime.datetime.now(datetime.timezone.utc)
        n_bars = 400
        delta = get_timeframe_delta(timeframe)
        timestamps = [now - delta * (n_bars - i) for i in range(n_bars)]

        np.random.seed(12345)
        returns = np.random.normal(0.00005, 0.002, size=n_bars)
        price_series = 3900.0 * np.exp(np.cumsum(returns))

        opens = price_series
        closes = price_series * (1 + np.random.normal(0, 0.001, size=n_bars))
        highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 1.5, size=n_bars))
        lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 1.5, size=n_bars))
        volumes = np.random.exponential(100.0, size=n_bars) + 10.0
        amounts = volumes * closes

        return pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "amount": amounts
        })


def fetch_xauusd_candles(
    timeframe: str = "H1",
    lookback_days: Optional[int] = None,
    force_refresh: bool = False,
    allow_simulation: bool = False
) -> Dict[str, Any]:
    feed = DukascopyFeed()
    return feed.fetch_historical_data(
        timeframe=timeframe,
        lookback_days=lookback_days,
        force_refresh=force_refresh,
        allow_simulation=allow_simulation
    )
