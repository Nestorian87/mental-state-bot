from __future__ import annotations

from datetime import UTC, datetime

from mental_state_bot.services.planned_events import detect_planned_event_candidate


def test_detect_planned_event_candidate_for_future_event() -> None:
    candidate = detect_planned_event_candidate(
        "Зустріч ще не починалася. Я ще вдома.",
        timezone="Europe/Kyiv",
        now=datetime(2026, 7, 9, 10, 0, tzinfo=UTC),
    )

    assert candidate is not None
    assert candidate["title"] == "зустріч"
    assert candidate["status"] == "pending"


def test_detect_planned_event_candidate_parses_relative_time() -> None:
    candidate = detect_planned_event_candidate(
        "Через годину зустріч, я вже збираюся.",
        timezone="Europe/Kyiv",
        now=datetime(2026, 7, 9, 10, 0, tzinfo=UTC),
    )

    assert candidate is not None
    assert candidate["starts_at"] == "2026-07-09T11:00:00+00:00"


def test_detect_planned_event_candidate_ignores_completed_event() -> None:
    assert (
        detect_planned_event_candidate(
            "Зустріч пройшла.",
            timezone="Europe/Kyiv",
            now=datetime(2026, 7, 9, 10, 0, tzinfo=UTC),
        )
        is None
    )
