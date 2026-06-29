from __future__ import annotations

from mental_state_bot.time_utils import parse_hhmm, zoneinfo


def test_parse_hhmm() -> None:
    parsed = parse_hhmm("09:30")

    assert parsed.hour == 9
    assert parsed.minute == 30


def test_zoneinfo_loads_configured_kyiv_timezone() -> None:
    assert zoneinfo("Europe/Kyiv").key == "Europe/Kyiv"
