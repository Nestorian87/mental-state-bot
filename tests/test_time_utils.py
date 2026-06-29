from __future__ import annotations

from mental_state_bot.time_utils import parse_hhmm


def test_parse_hhmm() -> None:
    parsed = parse_hhmm("09:30")

    assert parsed.hour == 9
    assert parsed.minute == 30
