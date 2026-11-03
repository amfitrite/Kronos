"""Market Calendar and Session Logic for Spot Gold (XAUUSD).

XAUUSD (Swiss ECN / Interbank Spot Gold) Trading Schedule (UTC):
- Weekly Open: Sunday 23:00 UTC (18:00 EST)
- Weekly Close: Friday 21:00 UTC (17:00 EST)
- Daily Maintenance Rollover Break: Mon-Thu 21:00 to 22:00 UTC (1 hour trading pause)
- Weekend Break: Friday 21:00 UTC to Sunday 23:00 UTC (50 hours closed)
"""
import datetime
from typing import Any, Dict, List, Optional
import pandas as pd


TIMEFRAME_DELTAS = {
    "M1": datetime.timedelta(minutes=1),
    "1M": datetime.timedelta(minutes=1),
    "M5": datetime.timedelta(minutes=5),
    "5M": datetime.timedelta(minutes=5),
    "M15": datetime.timedelta(minutes=15),
    "15M": datetime.timedelta(minutes=15),
    "H1": datetime.timedelta(hours=1),
    "1H": datetime.timedelta(hours=1),
    "H4": datetime.timedelta(hours=4),
    "4H": datetime.timedelta(hours=4),
    "D1": datetime.timedelta(days=1),
    "1D": datetime.timedelta(days=1),
}


def get_timeframe_delta(timeframe: str) -> datetime.timedelta:
    """Returns the timedelta corresponding to the specified timeframe."""
    tf_norm = timeframe.strip().upper()
    if tf_norm not in TIMEFRAME_DELTAS:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(TIMEFRAME_DELTAS.keys())}")
    return TIMEFRAME_DELTAS[tf_norm]


def is_xauusd_open(dt: datetime.datetime) -> bool:
    """Checks whether the XAUUSD market is open at the given UTC datetime.
    
    Dukascopy Swiss ECN Spot Gold Schedule (UTC):
    - Saturday (weekday 5): Closed all day.
    - Sunday (weekday 6): Closed before 22:00 UTC; Open 22:00 - 24:00 UTC.
    - Friday (weekday 4): Open 00:00 - 21:00 UTC; Closed from 21:00 UTC onwards (last bar is 20:00-20:59).
    - Mon-Thu (weekday 0-3): Open except 21:00 - 22:00 UTC daily rollover maintenance break (Tue-Thu).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    else:
        dt = dt.astimezone(datetime.timezone.utc)

    weekday = dt.weekday()
    hour = dt.hour

    # Saturday: completely closed
    if weekday == 5:
        return False

    # Sunday: closed until 22:00 UTC
    if weekday == 6:
        return hour >= 22

    # Friday: closed from 21:00 UTC onwards
    if weekday == 4:
        return hour < 21

    # Tuesday, Wednesday, Thursday: hour 21 is rollover break
    if weekday in (1, 2, 3) and hour == 21:
        return False

    return True


is_market_open = is_xauusd_open


def next_market_bar(current_dt: datetime.datetime, timeframe: str) -> datetime.datetime:
    """Advances current_dt by the timeframe duration, skipping non-trading sessions."""
    delta = get_timeframe_delta(timeframe)
    if current_dt.tzinfo is None:
        current_dt = current_dt.replace(tzinfo=datetime.timezone.utc)
    else:
        current_dt = current_dt.astimezone(datetime.timezone.utc)

    candidate = current_dt + delta

    # If candidate lands during closed hours, fast-forward to next market open
    while not is_xauusd_open(candidate):
        weekday = candidate.weekday()
        hour = candidate.hour

        # Friday 21:00 or later -> skip forward to Sunday 22:00 UTC
        if weekday == 4 and hour >= 21:
            candidate = candidate.replace(hour=22, minute=0, second=0, microsecond=0) + datetime.timedelta(days=2)
        elif weekday == 5: # Saturday -> skip to Sunday 22:00 UTC
            candidate = candidate.replace(hour=22, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
        elif weekday == 6 and hour < 22: # Sunday before 22:00 -> advance to 22:00 UTC
            candidate = candidate.replace(hour=22, minute=0, second=0, microsecond=0)
        elif weekday in (1, 2, 3) and hour == 21: # Tue-Thu rollover break
            candidate = candidate.replace(hour=22, minute=0, second=0, microsecond=0)
        else:
            candidate += delta

    return candidate


def generate_market_timestamps(
    start_dt: datetime.datetime,
    count: Optional[int] = None,
    timeframe: str = "H1",
    pred_len: Optional[int] = None
) -> List[datetime.datetime]:
    """Generates an exact sequence of consecutive future trading bar timestamps."""
    n_bars = count if count is not None else (pred_len if pred_len is not None else 24)
    timestamps = []
    curr = start_dt
    for _ in range(n_bars):
        curr = next_market_bar(curr, timeframe)
        timestamps.append(curr)
    return timestamps


def is_candle_closed(
    candle_timestamp: datetime.datetime,
    timeframe: str,
    now_dt: Optional[datetime.datetime] = None
) -> bool:
    """Checks whether the candle opened at candle_timestamp has completely closed.
    
    A candle that opened at 13:00:00 H1 closes at 14:00:00.
    If now is 13:45:00, the candle is still forming (incomplete).
    """
    if now_dt is None:
        now_dt = datetime.datetime.now(datetime.timezone.utc)
    elif now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=datetime.timezone.utc)

    if candle_timestamp.tzinfo is None:
        candle_timestamp = candle_timestamp.replace(tzinfo=datetime.timezone.utc)

    delta = get_timeframe_delta(timeframe)
    candle_close_time = candle_timestamp + delta
    return now_dt >= candle_close_time


def detect_data_gaps(
    df: pd.DataFrame,
    timeframe: str,
    tolerance_missing_bars: int = 3
) -> List[Dict[str, Any]]:
    """Detects missing candle gaps during active market trading hours."""
    if len(df) < 2 or "timestamp" not in df.columns:
        return []

    delta = get_timeframe_delta(timeframe)
    delta_seconds = delta.total_seconds()
    timestamps = pd.to_datetime(df["timestamp"]).dt.tz_localize(datetime.timezone.utc, ambiguous='NaT', nonexistent='NaT') if df["timestamp"].dt.tz is None else df["timestamp"]

    gaps = []
    for i in range(len(timestamps) - 1):
        t1 = timestamps.iloc[i]
        t2 = timestamps.iloc[i + 1]

        diff_sec = (t2 - t1).total_seconds()
        # If difference exceeds expected step by more than tolerance
        if diff_sec > delta_seconds * tolerance_missing_bars:
            # Check if this entire period was simply weekend/rollover closure
            # Sample midpoint
            mid = t1 + datetime.timedelta(seconds=diff_sec / 2)
            if is_xauusd_open(mid):
                gaps.append({
                    "start": str(t1),
                    "end": str(t2),
                    "duration_seconds": diff_sec,
                    "missing_approx_bars": int(diff_sec // delta_seconds) - 1
                })

    return gaps
