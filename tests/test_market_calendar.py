"""Unit Tests for Spot Gold (XAUUSD) Market Calendar.

Verifies:
1. Saturday is closed all day.
2. Sunday before 22:00 UTC is closed, 22:00-24:00 UTC is open.
3. Friday after 21:00 UTC is closed (last bar is 20:00 UTC).
4. Daily maintenance rollover break (Mon-Thu 21:00-22:00 UTC) is respected.
5. next_market_bar correctly skips closed sessions into open sessions.
6. generate_market_timestamps creates only timestamps during valid trading hours.
"""
import datetime
import unittest

from kronos_app.data.market_calendar import (
    is_xauusd_open,
    is_market_open,
    next_market_bar,
    generate_market_timestamps
)


class TestMarketCalendar(unittest.TestCase):

    def test_weekend_closed(self):
        """Saturday must be closed at all hours."""
        for hour in range(24):
            sat_dt = datetime.datetime(2026, 11, 7, hour, 0, tzinfo=datetime.timezone.utc)
            self.assertFalse(is_xauusd_open(sat_dt), f"Saturday at {hour}:00 should be closed")

    def test_sunday_open_after_22_utc(self):
        """Sunday before 22:00 UTC is closed; 22:00 UTC onwards is open."""
        sun_closed = datetime.datetime(2026, 11, 8, 20, 0, tzinfo=datetime.timezone.utc)
        self.assertFalse(is_xauusd_open(sun_closed))

        sun_open = datetime.datetime(2026, 11, 8, 22, 0, tzinfo=datetime.timezone.utc)
        self.assertTrue(is_xauusd_open(sun_open))

    def test_friday_close_after_21_utc(self):
        """Friday is open until 21:00 UTC, closed from 21:00 UTC onwards."""
        fri_open = datetime.datetime(2026, 11, 6, 20, 0, tzinfo=datetime.timezone.utc)
        self.assertTrue(is_xauusd_open(fri_open))

        fri_closed = datetime.datetime(2026, 11, 6, 21, 0, tzinfo=datetime.timezone.utc)
        self.assertFalse(is_xauusd_open(fri_closed))

    def test_daily_rollover_break(self):
        """Tuesday 21:00-22:00 UTC is closed for maintenance rollover."""
        tue_break = datetime.datetime(2026, 11, 3, 21, 30, tzinfo=datetime.timezone.utc)
        self.assertFalse(is_xauusd_open(tue_break))

        tue_open_again = datetime.datetime(2026, 11, 3, 22, 15, tzinfo=datetime.timezone.utc)
        self.assertTrue(is_xauusd_open(tue_open_again))

    def test_next_market_bar_friday_to_sunday(self):
        """Bar advancing from Friday close skips Saturday and lands on Sunday open (22:00 UTC)."""
        last_friday_bar = datetime.datetime(2026, 11, 6, 20, 0, tzinfo=datetime.timezone.utc) # H1
        next_bar = next_market_bar(last_friday_bar, "H1")
        self.assertEqual(next_bar.weekday(), 6) # Sunday
        self.assertEqual(next_bar.hour, 22)

    def test_generate_market_timestamps(self):
        """Future timestamps must never land on closed hours."""
        start_ts = datetime.datetime(2026, 11, 6, 18, 0, tzinfo=datetime.timezone.utc)
        future_ts = generate_market_timestamps(start_ts, pred_len=8, timeframe="H1")
        self.assertEqual(len(future_ts), 8)
        for ts in future_ts:
            self.assertTrue(is_market_open(ts), f"Generated timestamp {ts} is in closed market session!")


if __name__ == "__main__":
    unittest.main()
