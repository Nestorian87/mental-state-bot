from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from mental_state_bot.services.wake_time import (
    parse_wake_time_text,
    should_offer_wake_time_question,
)


def test_parse_wake_time_text_handles_clock_time() -> None:
    record = parse_wake_time_text(
        "приблизно о 9:30",
        timezone="Europe/Kyiv",
        now=datetime(2026, 7, 9, 8, 0, tzinfo=UTC),
    )

    assert record["estimated_woke_at"] == "2026-07-09T06:30:00+00:00"


def test_parse_wake_time_text_handles_relative_half_hour() -> None:
    record = parse_wake_time_text(
        "пів години тому",
        timezone="Europe/Kyiv",
        now=datetime(2026, 7, 9, 8, 0, tzinfo=UTC),
    )

    assert record["estimated_woke_at"] == "2026-07-09T07:30:00+00:00"


def test_wake_time_offer_for_first_meaningful_entry_even_with_time_in_text() -> None:
    settings = SimpleNamespace(settings_json={})
    entry = SimpleNamespace(source="manual", raw_text="Прокинувся приблизно о 9")

    assert should_offer_wake_time_question(
        entries=[entry],
        current_text=entry.raw_text,
        user_settings=settings,
        local_date="2026-07-09",
    )


def test_wake_time_offer_for_first_meaningful_entry_without_wake_time() -> None:
    settings = SimpleNamespace(settings_json={})
    entry = SimpleNamespace(source="manual", raw_text="Тільки прокинувся, настрій нормальний")

    assert should_offer_wake_time_question(
        entries=[entry],
        current_text=entry.raw_text,
        user_settings=settings,
        local_date="2026-07-09",
    )
