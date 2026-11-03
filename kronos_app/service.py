"""Kronos Service Orchestrator: Non-blocking Async Worker Thread & Multi-Model Engine."""
import json
import logging
from pathlib import Path
import threading
from typing import Any, Callable, Dict, Optional
import pandas as pd

from kronos_app.data.dukascopy_feed import DukascopyFeed
from kronos_app.core.kronos_engine import KronosMonteCarloEngine

logger = logging.getLogger("Kronos.Service")


class KronosService:
    """Coordinates data feeds, dual AI models (mini/base), and background worker threads."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("d:/Kronos/data")
        self.session_file = self.data_dir / "last_session.json"
        self.feed = DukascopyFeed(cache_dir=self.data_dir)
        self.engine = KronosMonteCarloEngine(model_type="mini")
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

    @staticmethod
    def calculate_horizon_bars(timeframe: str, horizon_input: Any) -> int:
        """Translates horizon strings (e.g. '15M', '1H', '4H', '1D', '2D', '3D', '1W', '2W') into bar counts."""
        h_str = str(horizon_input).strip().upper()
        tf = timeframe.upper()

        if h_str.isdigit():
            return max(4, int(h_str))

        # Hours mapping
        # 15M = 0.25h, 30M = 0.5h, 1H = 1h, 4H = 4h, 8H = 8h, 12H = 12h, 1D = 24h, 2D = 48h, 3D = 72h, 1W = 120h (5 trading days), 2W = 240h (10 trading days)
        hours_map = {
            "15M": 0.25,
            "30M": 0.5,
            "1H": 1.0,
            "4H": 4.0,
            "8H": 8.0,
            "12H": 12.0,
            "1D": 24.0,
            "2D": 48.0,
            "3D": 72.0,
            "1W": 120.0,
            "2W": 240.0
        }

        if h_str in hours_map:
            total_hours = hours_map[h_str]
            if tf in ("M1", "1M"):
                return max(4, min(300, int(total_hours * 60)))
            elif tf in ("M5", "5M"):
                return max(4, min(288, int(total_hours * 12)))
            elif tf in ("M15", "15M"):
                return max(4, min(300, int(total_hours * 4)))
            elif tf in ("H1", "1H"):
                return max(4, int(total_hours))
            elif tf in ("H4", "4H"):
                return max(2, int(total_hours / 4))
            elif tf in ("D1", "1D"):
                return max(1, int(total_hours / 24))

        return 24

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
        """Returns live system diagnostic status."""
        cache_path = self.feed.get_cache_path(self.current_timeframe)
        has_cache = cache_path.exists()
        cached_count = 0
        last_sync = "None"
        last_price = 0.0

        if has_cache:
            try:
                df = pd.read_parquet(cache_path)
                cached_count = len(df)
                if not df.empty:
                    last_sync = str(df["timestamp"].iloc[-1])
                    last_price = float(df["close"].iloc[-1])
            except Exception:
                pass

        return {
            "model_name": self.engine.model_name,
            "model_type": self.engine.model_type,
            "device": self.engine.device.upper(),
            "model_ready": self.engine.is_ready,
            "data_source": "Dukascopy Bank ECN (Swiss)",
            "symbol": "XAUUSD",
            "timeframe": self.current_timeframe,
            "cached_bars": cached_count,
            "last_sync": last_sync,
            "last_price": round(last_price, 3),
            "is_computing": self._is_computing
        }

    def load_data(self, timeframe: str = "H1", force_refresh: bool = False) -> Dict[str, Any]:
        """Loads or updates Dukascopy candle data."""
        self.current_timeframe = timeframe.upper()
        df = self.feed.fetch_historical_data(
            timeframe=self.current_timeframe,
            lookback_days=None,
            force_refresh=force_refresh
        )
        self._cached_df = df
        return {
            "success": True,
            "timeframe": self.current_timeframe,
            "count": len(df),
            "last_price": float(df["close"].iloc[-1]),
            "last_time": str(df["timestamp"].iloc[-1])
        }

    def start_forecast_async(
        self,
        model_type: str = "mini",
        timeframe: str = "H1",
        pred_len: int = 12,
        sample_count: int = 5,
        horizon_label: Optional[str] = None,
        on_completed: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Starts asynchronous prediction on a background worker thread."""
        if self._is_computing:
            return {"success": False, "error": "A forecast is already running."}

        self._is_computing = True
        self._current_step = 0
        self._total_steps = pred_len
        self._progress_pct = 0.0
        self._last_result = None
        self._last_error = None

        def _worker():
            try:
                logger.info(f"Worker started: model={model_type}, pred_len={pred_len}, samples={sample_count}")
                self.engine.set_model_type(model_type)
                self.current_timeframe = timeframe.upper()
                if self._cached_df is None or getattr(self, '_loaded_timeframe', None) != self.current_timeframe:
                    self.load_data(timeframe=self.current_timeframe)

                df = self._cached_df
                if df is None or len(df) < 30:
                    raise ValueError("Insufficient candle data available for Kronos inference.")

                def _on_progress(step, total, fraction):
                    self._current_step = step
                    self._total_steps = total
                    self._progress_pct = round(fraction * 100.0, 1)

                forecast_res = self.engine.predict_monte_carlo(
                    df=df,
                    pred_len=pred_len,
                    sample_count=sample_count,
                    temperature=1.0,
                    top_p=0.9,
                    progress_callback=_on_progress
                )
                self._cached_forecast = forecast_res
                self._last_result = {"success": True, "data": forecast_res}

                # Persist session state so user can continue where they left off
                self.save_session({
                    "model_type": model_type,
                    "timeframe": self.current_timeframe,
                    "pred_len": pred_len,
                    "horizon_label": horizon_label or str(pred_len),
                    "sample_count": sample_count,
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

    def execute_forecast(
        self,
        model_type: str = "mini",
        timeframe: str = "H1",
        pred_len: int = 12,
        sample_count: int = 5
    ) -> Dict[str, Any]:
        """Synchronous forecast (primarily for headless CLI)."""
        self.engine.set_model_type(model_type)
        self.current_timeframe = timeframe.upper()
        if self._cached_df is None:
            self.load_data(timeframe=self.current_timeframe)
        df = self._cached_df
        if df is None:
            raise ValueError("No data available.")

        res = self.engine.predict_monte_carlo(
            df=df,
            pred_len=pred_len,
            sample_count=sample_count,
            temperature=1.0,
            top_p=0.9
        )
        self._cached_forecast = res
        self.save_session({
            "model_type": model_type,
            "timeframe": self.current_timeframe,
            "pred_len": pred_len,
            "sample_count": sample_count,
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
