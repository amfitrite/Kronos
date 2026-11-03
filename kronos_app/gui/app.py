"""Google Chrome Frameless GUI Application for Kronos Foundation Core (Non-blocking Async)."""
import ctypes
import json
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Optional
import webview

from kronos_app.service import KronosService

logger = logging.getLogger("Kronos.GUI")

if sys.platform == "win32":
    try:
        myappid = "Kronos.Foundation.QuantitativeEngine.v1.0"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        logger.debug(f"Could not set AppUserModelID: {e}")


class ChromeWindowAPI:
    """JS API bridge exposed to the pywebview frontend."""

    def __init__(self, service: KronosService):
        self._service = service
        self._window: Optional[webview.Window] = None

    def set_window(self, window: webview.Window):
        self._window = window

    # --- Frameless Window Controls ---
    def minimize_window(self):
        if self._window:
            self._window.minimize()

    def toggle_maximize(self):
        if self._window:
            try:
                self._window.maximize()
            except Exception:
                pass

    def close_window(self):
        if self._window:
            self._window.destroy()

    # --- Asynchronous Non-blocking Forecasting Bridge ---
    def get_status(self) -> str:
        try:
            status = self._service.get_status()
            return json.dumps(status)
        except Exception as e:
            logger.error(f"get_status failed: {e}")
            return json.dumps({"error": str(e)})

    def get_last_session(self) -> str:
        """Returns the persisted session state and last forecast if available."""
        try:
            res = self._service.get_last_session()
            return json.dumps(res)
        except Exception as e:
            logger.error(f"get_last_session failed: {e}")
            return json.dumps({"success": False, "has_session": False, "error": str(e)})

    def start_forecast(
        self,
        model_type: str = "mini",
        timeframe: str = "H1",
        pred_len: object = "1D",
        sample_count: object = 100,
        seed: object = None
    ) -> str:
        """Starts asynchronous prediction on background worker thread with robust type guarding."""
        actual_seed = None
        if seed is not None and str(seed).strip() != "":
            try:
                actual_seed = int(seed)
            except (ValueError, TypeError):
                actual_seed = None

        actual_samples = 100
        if sample_count is not None and str(sample_count).strip() != "":
            try:
                parsed_samples = int(sample_count)
                if parsed_samples > 0:
                    actual_samples = parsed_samples
            except (ValueError, TypeError):
                actual_samples = 100

        bars = self._service.calculate_horizon_bars(timeframe, pred_len)

        def _on_completed(result):
            if self._window:
                try:
                    payload = json.dumps(result)
                    self._window.evaluate_js(f"window.onForecastCompleted({payload});")
                except Exception as e:
                    logger.debug(f"evaluate_js notice: {e}")

        res = self._service.start_forecast_async(
            model_type=model_type,
            timeframe=timeframe,
            pred_len=bars,
            sample_count=actual_samples,
            seed=actual_seed,
            horizon_label=str(pred_len),
            on_completed=_on_completed
        )
        return json.dumps(res)

    def get_forecast_status(self) -> str:
        """Polls current progress of background forecast computation."""
        progress = self._service.get_forecast_progress()
        return json.dumps(progress)

    def get_walk_forward_results(self, timeframe: str = "M5") -> str:
        """Retrieves stored walk-forward validation backtest results."""
        try:
            res = self._service.get_walk_forward_results(timeframe=timeframe)
            return json.dumps(res)
        except Exception as e:
            logger.error(f"get_walk_forward_results failed: {e}")
            return json.dumps({"success": False, "error": str(e)})

    def run_walk_forward(
        self,
        timeframe: str = "M5",
        num_windows: object = 10,
        sample_count: object = 50,
        seed: object = 42
    ) -> str:
        """Runs walk-forward backtest asynchronously with robust parameter parsing."""
        try:
            n_win = max(2, int(num_windows)) if num_windows is not None and str(num_windows).strip().isdigit() else 5
        except (ValueError, TypeError):
            n_win = 5
        try:
            s_cnt = max(10, int(sample_count)) if sample_count is not None and str(sample_count).strip().isdigit() else 50
        except (ValueError, TypeError):
            s_cnt = 50
        try:
            s_val = int(seed) if seed is not None and str(seed).strip().isdigit() else 42
        except (ValueError, TypeError):
            s_val = 42

        def _wf_worker():
            try:
                res = self._service.run_walk_forward_validation(
                    timeframe=timeframe,
                    num_windows=n_win,
                    sample_count=s_cnt,
                    seed=s_val
                )
                if self._window:
                    try:
                        payload = json.dumps({"success": True, "data": res})
                        self._window.evaluate_js(f"window.onWalkForwardCompleted({payload});")
                    except Exception as err:
                        logger.debug(f"WF evaluate_js err: {err}")
            except Exception as e:
                logger.error(f"Walk forward run failed: {e}", exc_info=True)
                if self._window:
                    try:
                        payload = json.dumps({"success": False, "error": str(e)})
                        self._window.evaluate_js(f"window.onWalkForwardCompleted({payload});")
                    except Exception:
                        pass

        wf_thread = threading.Thread(target=_wf_worker, daemon=True)
        wf_thread.start()
        return json.dumps({"success": True, "started": True, "timeframe": timeframe})

    def get_token_diagnostics(self) -> str:
        try:
            res = self._service.get_token_diagnostics(num_bars=48)
            return json.dumps(res)
        except Exception as e:
            logger.error(f"get_token_diagnostics failed: {e}")
            return json.dumps({"success": False, "error": str(e)})

    # --- Real-Time Dukascopy Live Feed & Ticker ---
    def set_timeframe(self, timeframe: str) -> str:
        """Updates the active timeframe in the service for the live ticker and data feed."""
        try:
            self._service.set_timeframe(timeframe)
            return json.dumps({"success": True, "timeframe": self._service.current_timeframe})
        except Exception as e:
            logger.error(f"set_timeframe failed: {e}")
            return json.dumps({"success": False, "error": str(e)})

    def get_live_candle(self, timeframe: Optional[str] = None) -> str:
        """Fetches the latest real-time tick and forming live bar from Dukascopy."""
        try:
            res = self._service.get_live_candle(timeframe)
            return json.dumps(res)
        except Exception as e:
            logger.error(f"get_live_candle failed: {e}")
            return json.dumps({"success": False, "error": str(e)})

    def toggle_live_feed(self, enabled: bool) -> str:
        """Toggles background real-time streaming of Dukascopy live candle updates."""
        try:
            if enabled:
                def _on_live_update(data):
                    if self._window:
                        try:
                            payload = json.dumps(data)
                            self._window.evaluate_js(f"window.onLiveCandleUpdate({payload});")
                        except Exception as err:
                            logger.debug(f"Live dispatch notice: {err}")

                self._service.start_live_stream(callback=_on_live_update, poll_interval=2.5)
            else:
                self._service.stop_live_stream()
            return json.dumps({"success": True, "live_active": enabled})
        except Exception as e:
            logger.error(f"toggle_live_feed failed: {e}")
            return json.dumps({"success": False, "error": str(e)})


def _apply_taskbar_icon():
    """Applies custom icon to Win32 window handle for crisp taskbar thumbnail."""
    if sys.platform != "win32":
        return
    try:
        time.sleep(0.5)
        hwnd = ctypes.windll.user32.FindWindowW(None, "Kronos Foundation Core | XAUUSD")
        if hwnd:
            ico_path = str(Path(__file__).parent.parent.parent / "assets" / "icon.ico")
            if Path(ico_path).exists():
                IMAGE_ICON = 1
                LR_LOADFROMFILE = 0x00000010
                hicon_big = ctypes.windll.user32.LoadImageW(None, ico_path, IMAGE_ICON, 48, 48, LR_LOADFROMFILE)
                hicon_small = ctypes.windll.user32.LoadImageW(None, ico_path, IMAGE_ICON, 16, 16, LR_LOADFROMFILE)
                WM_SETICON = 0x0080
                if hicon_big:
                    ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, 1, hicon_big)
                if hicon_small:
                    ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, 0, hicon_small)
    except Exception as e:
        logger.debug(f"Taskbar icon notice: {e}")


def launch_gui(service: Optional[KronosService] = None):
    """Launches Google Chrome Frameless GUI without console logs."""
    app_service = service or KronosService()
    api = ChromeWindowAPI(app_service)

    html_file = Path(__file__).parent / "templates" / "chrome_ui.html"
    if not html_file.exists():
        raise FileNotFoundError(f"HTML UI template not found at {html_file}")

    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    window = webview.create_window(
        title="Kronos Foundation Core | XAUUSD",
        html=html_content,
        js_api=api,
        width=1340,
        height=860,
        frameless=True,       # Frameless Chrome styling!
        easy_drag=True,
        min_size=(1050, 700),
        background_color="#1e1f22"
    )
    api.set_window(window)

    webview.start(_apply_taskbar_icon, debug=False)
