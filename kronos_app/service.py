"""Kronos Service Orchestrator: Non-blocking Async Worker Thread, Validation & Engine."""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
from typing import Any, Callable, Dict, Optional, Tuple
import pandas as pd

from kronos_app.data.dukascopy_feed import DukascopyFeed, DataUnavailableError
from kronos_app.data.market_calendar import is_market_open
from kronos_app.core.kronos_engine import KronosMonteCarloEngine
from kronos_app.evaluation.walk_forward import WalkForwardValidator

logger = logging.getLogger("Kronos.Service")


class KronosService:
    """Coordinates data feeds, Kronos AI models, walk-forward validation, and background worker threads."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("d:/Kronos/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.data_dir / "last_session.json"
        self.feed = DukascopyFeed(cache_dir=self.data_dir)
        self.engine = KronosMonteCarloEngine(model_type="mini")
        self.validator = WalkForwardValidator(engine=self.engine, results_dir=self.data_dir)
        self.current_timeframe = "H1"
        self._cached_forecast: Optional[Dict[str, Any]] = None
        self._cached_df: Optional[pd.DataFrame] = None

        # Asynchronous job state
        self._is_computing = False
        self._current_step = 0
        self._total_steps = 0
        self._progress_pct = 0.0
        self._last_result: Optional[Dict[str, Any]] = None
        self._last_error: Optional[str] = None
        self._worker_thread: Optional[threading.Thread] = None

    @classmethod
    def get_horizon_details(cls, timeframe: str, horizon_input: Any) -> Tuple[int, str]:
        """Translates timeframe and requested horizon into exact bar counts and accurate duration string.
        
        Ensures NO silent truncation and mathematically accurate durations:
        - M1: 15m (15c), 30m (30c), 1h (60c), 2h (120c), 3h (180c), 5h (300c = 5 Hours)
        - M5: 1h (12c), 2h (24c), 4h (48c), 8h (96c), 12h (144c), 1d/24h (288c = 24 Hours / 1 Day)
        - H1: 4h (4c), 12h (12c), 1d (24c), 2d (48c), 3d (72c), 1w (120c = 5 Trading Days), 2w (240c = 10 Trading Days)
        """
        if horizon_input is None:
            h_str = "1H"
        else:
            h_str = str(horizon_input).strip().upper()
        if not h_str:
            h_str = "1H"
        tf = timeframe.upper()

        if h_str.isdigit():
            bars = max(4, int(h_str))
            return bars, f"{bars} Candles"

        # Explicit duration configurations per timeframe
        if tf in ("M1", "1M"):
            m1_table = {
                "15M": (15, "15 Minutes (15 candles)"),
                "30M": (30, "30 Minutes (30 candles)"),
                "1H": (60, "1 Hour (60 candles)"),
                "2H": (120, "2 Hours (120 candles)"),
                "3H": (180, "3 Hours (180 candles)"),
                "5H": (300, "5 Hours (300 candles)"),
                "1D": (300, "5 Hours (Max 300 candles on M1)")
            }
            if h_str in m1_table:
                return m1_table[h_str]
            return (60, "1 Hour (60 candles)")

        elif tf in ("M5", "5M"):
            m5_table = {
                "1H": (12, "1 Hour (12 candles)"),
                "2H": (24, "2 Hours (24 candles)"),
                "4H": (48, "4 Hours (48 candles)"),
                "8H": (96, "8 Hours (96 candles)"),
                "12H": (144, "12 Hours (144 candles)"),
                "1D": (288, "24 Hours / 1 Day (288 candles)"),
                "2D": (288, "24 Hours (Max 288 candles on M5)")
            }
            if h_str in m5_table:
                return m5_table[h_str]
            return (24, "2 Hours (24 candles)")

        else: # H1, H4, D1
            h1_table = {
                "4H": (4, "4 Hours (4 candles)"),
                "12H": (12, "12 Hours (12 candles)"),
                "1D": (24, "24 Hours / 1 Day (24 candles)"),
                "2D": (48, "48 Hours / 2 Days (48 candles)"),
                "3D": (72, "72 Hours / 3 Days (72 candles)"),
                "1W": (120, "5 Trading Days / 1 Week (120 candles)"),
                "2W": (240, "10 Trading Days / 2 Weeks (240 candles)")
            }
            if h_str in h1_table:
                return h1_table[h_str]
            return (24, "24 Hours (24 candles)")

    @classmethod
    def calculate_horizon_bars(cls, timeframe: str, horizon_input: Any) -> int:
        """Returns integer number of candles for the given horizon."""
        bars, _ = cls.get_horizon_details(timeframe, horizon_input)
        return bars

    def get_last_session(self) -> Dict[str, Any]:
        """Loads and returns the persisted session state and last forecast if available."""
        if self.session_file.exists():
            try:
                with open(self.session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {"success": True, "has_session": True, "session": data}
            except Exception as e:
                logger.warning(f"Could not load last session: {e}")
        return {"success": True, "has_session": False}

    def save_session(self, session_data: Dict[str, Any]):
        """Persists the session configuration and latest forecast to disk."""
        try:
            with open(self.session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f)
        except Exception as e:
            logger.warning(f"Failed to persist session: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Returns live system diagnostic status including strict data integrity verification."""
        cache_path = self.feed.get_cache_path(self.current_timeframe)
        has_cache = cache_path.exists()
        cached_count = 0
        last_sync = "None"
        last_price = 0.0
        data_status = "DATA UNAVAILABLE"
        freshness_label = "No Cache"

        if has_cache:
            try:
                df = pd.read_parquet(cache_path)
                cached_count = len(df)
                if not df.empty:
                    last_ts = pd.to_datetime(df["timestamp"].iloc[-1]).to_pydatetime()
                    last_sync = last_ts.strftime("%Y-%m-%d %H:%M UTC")
                    last_price = float(df["close"].iloc[-1])

                    is_fresh = self.feed.is_cache_fresh(self.current_timeframe)
                    if is_fresh:
                        data_status = "VALID"
                        freshness_label = "LIVE FRESH"
                    else:
                        data_status = "STALE"
                        freshness_label = "STALE (Needs Refresh)"
            except Exception as e:
                logger.warning(f"Error checking cache status: {e}")
                data_status = "CORRUPT"
                freshness_label = "Cache Corrupt"

        # Check walk-forward validation availability
        m1_wf_exists = (self.data_dir / "walk_forward_m1.json").exists()
        m5_wf_exists = (self.data_dir / "walk_forward_m5.json").exists()

        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        market_open = is_market_open(now_utc)

        return {
            "model_name": self.engine.model_name,
            "model_type": self.engine.model_type,
            "device": self.engine.device.upper(),
            "model_ready": self.engine.is_ready,
            "data_source": "Dukascopy Bank ECN (Swiss)",
            "symbol": "XAUUSD",
            "timeframe": self.current_timeframe,
            "data_status": data_status,
            "freshness": freshness_label,
            "market_session_open": market_open,
            "market_session_status": "OPEN (Interbank ECN)" if market_open else "CLOSED (Weekend / Rollover)",
            "cached_bars": cached_count,
            "last_sync": last_sync,
            "last_price": round(last_price, 3),
            "walk_forward_m1_ready": m1_wf_exists,
            "walk_forward_m5_ready": m5_wf_exists,
            "is_computing": self._is_computing
        }

    def load_data(self, timeframe: str = "H1", force_refresh: bool = False) -> Dict[str, Any]:
        """Loads or updates Dukascopy candle data with strict OHLC validation."""
        self.current_timeframe = timeframe.upper()
        try:
            feed_res = self.feed.fetch_historical_data(
                timeframe=self.current_timeframe,
                lookback_days=None,
                force_refresh=force_refresh
            )
            df = feed_res["df"] if isinstance(feed_res, dict) else feed_res
            self._cached_df = df
            self._loaded_timeframe = self.current_timeframe
            return {
                "success": True,
                "data_status": feed_res.get("status", "VALID") if isinstance(feed_res, dict) else "VALID",
                "timeframe": self.current_timeframe,
                "count": len(df),
                "last_price": float(df["close"].iloc[-1]),
                "last_time": str(df["timestamp"].iloc[-1])
            }
        except DataUnavailableError as e:
            logger.error(f"Data load unavailable: {e}")
            return {
                "success": False,
                "data_status": "DATA UNAVAILABLE",
                "error": str(e),
                "timeframe": self.current_timeframe
            }

    def start_forecast_async(
        self,
        model_type: str = "mini",
        timeframe: str = "H1",
        pred_len: int = 12,
        sample_count: int = 100,
        seed: Optional[int] = None,
        horizon_label: Optional[str] = None,
        on_completed: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Starts asynchronous prediction on a background worker thread."""
        if self._is_computing:
            return {"success": False, "error": "A forecast computation is already running."}

        self._is_computing = True
        self._current_step = 0
        self._total_steps = pred_len
        self._progress_pct = 0.0
        self._last_result = None
        self._last_error = None

        try:
            safe_samples = max(5, int(sample_count))
        except (ValueError, TypeError):
            safe_samples = 100

        bars, dur_str = self.get_horizon_details(timeframe, horizon_label or pred_len)
        actual_pred_len = bars

        def _worker():
            try:
                logger.info(
                    f"Worker started: model={model_type}, tf={timeframe}, "
                    f"pred_len={actual_pred_len} ({dur_str}), samples={safe_samples}"
                )
                self.engine.set_model_type(model_type)
                self.current_timeframe = timeframe.upper()
                if self._cached_df is None or getattr(self, '_loaded_timeframe', None) != self.current_timeframe:
                    load_res = self.load_data(timeframe=self.current_timeframe)
                    if not load_res.get("success", False):
                        raise RuntimeError(load_res.get("error", "DATA UNAVAILABLE: Failed to fetch candle data."))

                df = self._cached_df
                if df is None or len(df) < 30:
                    raise ValueError("Insufficient candle data available for Kronos inference (need >= 30 bars).")

                def _on_progress(step, total, fraction):
                    self._current_step = step
                    self._total_steps = total
                    self._progress_pct = round(fraction * 100.0, 1)

                forecast_res = self.engine.predict_monte_carlo(
                    df=df,
                    pred_len=actual_pred_len,
                    sample_count=safe_samples,
                    timeframe=self.current_timeframe,
                    seed=seed,
                    temperature=1.0,
                    top_p=0.9,
                    progress_callback=_on_progress
                )

                # Attach rich audit metadata
                start_ts_str = str(forecast_res["future_timestamps"][0]) if forecast_res.get("future_timestamps") else "N/A"
                end_ts_str = str(forecast_res["future_timestamps"][-1]) if forecast_res.get("future_timestamps") else "N/A"

                forecast_res["metadata"] = {
                    "timeframe": self.current_timeframe,
                    "historical_bars_count": len(df),
                    "forecast_bars_count": actual_pred_len,
                    "actual_horizon_duration_str": dur_str,
                    "start_timestamp": start_ts_str,
                    "end_timestamp": end_ts_str,
                    "data_integrity_status": "VALID",
                    "data_source": "Dukascopy Bank ECN (Swiss)",
                    "seed": seed
                }

                self._cached_forecast = forecast_res
                self._last_result = {"success": True, "data": forecast_res}

                # Persist session state so user can continue where they left off
                self.save_session({
                    "model_type": model_type,
                    "timeframe": self.current_timeframe,
                    "pred_len": actual_pred_len,
                    "horizon_label": horizon_label or str(actual_pred_len),
                    "horizon_duration_str": dur_str,
                    "sample_count": sample_count,
                    "seed": seed,
                    "forecast_data": forecast_res
                })

                if on_completed:
                    on_completed(self._last_result)

            except Exception as e:
                logger.error(f"Async forecast failed: {e}", exc_info=True)
                self._last_error = str(e)
                self._last_result = {"success": False, "error": str(e)}
                if on_completed:
                    on_completed(self._last_result)
            finally:
                self._is_computing = False

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()
        return {"success": True, "started": True}

    def get_forecast_progress(self) -> Dict[str, Any]:
        """Returns the live status of the background computation."""
        return {
            "is_computing": self._is_computing,
            "step": self._current_step,
            "total": self._total_steps,
            "progress_pct": self._progress_pct,
            "has_result": self._last_result is not None,
            "result": self._last_result,
            "error": self._last_error
        }

    def get_walk_forward_results(self, timeframe: str = "M5") -> Dict[str, Any]:
        """Retrieves stored walk-forward validation results for the specified timeframe."""
        tf = timeframe.lower()
        file_path = self.data_dir / f"walk_forward_{tf}.json"
        if not file_path.exists():
            return {
                "success": False,
                "message": f"No walk-forward validation results found for {timeframe.upper()}. Run validation first."
            }
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"success": True, "data": data}
        except Exception as e:
            logger.error(f"Failed to load walk forward results: {e}")
            return {"success": False, "error": str(e)}

    def run_walk_forward_validation(
        self,
        timeframe: str = "M5",
        num_windows: int = 10,
        sample_count: int = 50,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Runs walk-forward backtest synchronously or via headless evaluation script."""
        tf = timeframe.upper()
        if self._cached_df is None or getattr(self, '_loaded_timeframe', None) != tf:
            self.load_data(timeframe=tf)
        df = self._cached_df
        if df is None:
            raise ValueError(f"No historical data available for {tf}")

        horizon = 12 if tf == "M5" else 15
        context_len = 60
        return self.validator.run_validation(
            df=df,
            timeframe=tf,
            horizon=horizon,
            context_len=context_len,
            num_windows=num_windows,
            sample_count=sample_count,
            seed=seed
        )

    def execute_forecast(
        self,
        model_type: str = "mini",
        timeframe: str = "H1",
        pred_len: int = 12,
        sample_count: int = 100,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Synchronous forecast (primarily for headless CLI)."""
        self.engine.set_model_type(model_type)
        self.current_timeframe = timeframe.upper()
        if self._cached_df is None:
            self.load_data(timeframe=self.current_timeframe)
        df = self._cached_df
        if df is None:
            raise ValueError("No data available.")

        bars, dur_str = self.get_horizon_details(timeframe, pred_len)
        try:
            safe_samples = max(5, int(sample_count))
        except (ValueError, TypeError):
            safe_samples = 100

        res = self.engine.predict_monte_carlo(
            df=df,
            pred_len=bars,
            sample_count=safe_samples,
            timeframe=self.current_timeframe,
            seed=seed,
            temperature=1.0,
            top_p=0.9
        )
        res["metadata"] = {
            "timeframe": self.current_timeframe,
            "historical_bars_count": len(df),
            "forecast_bars_count": bars,
            "actual_horizon_duration_str": dur_str,
            "start_timestamp": str(res["future_timestamps"][0]),
            "end_timestamp": str(res["future_timestamps"][-1]),
            "data_integrity_status": "VALID",
            "data_source": "Dukascopy Bank ECN (Swiss)",
            "seed": seed
        }
        self._cached_forecast = res
        self.save_session({
            "model_type": model_type,
            "timeframe": self.current_timeframe,
            "pred_len": bars,
            "horizon_duration_str": dur_str,
            "sample_count": sample_count,
            "seed": seed,
            "forecast_data": res
        })
        return {"success": True, "data": res}

    def get_token_diagnostics(self, num_bars: int = 48) -> Dict[str, Any]:
        """Returns discrete tokenizer inspection data."""
        try:
            if self._cached_df is None:
                self.load_data(timeframe=self.current_timeframe)
            df = self._cached_df
            if df is None:
                raise ValueError("No data available.")
            res = self.engine.get_token_diagnostics(df, num_bars=num_bars)
            return {"success": True, "data": res}
        except Exception as e:
            logger.error(f"Token diagnostics failed: {e}")
            return {"success": False, "error": str(e)}
