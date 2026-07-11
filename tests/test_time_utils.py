from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from mental_state_bot.time_utils import journal_date, parse_hhmm, zoneinfo


def test_parse_hhmm() -> None:
    parsed = parse_hhmm("09:30")

    assert parsed.hour == 9
    assert parsed.minute == 30


def test_zoneinfo_loads_configured_kyiv_timezone() -> None:
    assert zoneinfo("Europe/Kyiv").key == "Europe/Kyiv"


def test_journal_date_uses_active_start_as_day_boundary() -> None:
    now = datetime(2026, 7, 2, 1, 44, tzinfo=ZoneInfo("Europe/Kyiv"))

    assert journal_date("Europe/Kyiv", active_start="09:00", now=now) == date(2026, 7, 1)
