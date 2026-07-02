from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace
from uuid import uuid4
from zoneinfo import ZoneInfo

import mental_state_bot.services.summaries as summaries_module
from mental_state_bot.services.review import format_period_summary
from mental_state_bot.services.summaries import (
    SummaryService,
    _date_period_bounds,
    _entries_query_text,
    _semantic_records_context,
    auto_morning_boundary_end,
    current_month_dates,
    current_week_dates,
    previous_month_dates,
    previous_week_dates,
    sleep_marker_target_date,
)


def test_current_week_dates_start_monday_end_sunday() -> None:
    start, end = current_week_dates(date(2026, 6, 29))

    assert start == date(2026, 6, 29)
    assert end == date(2026, 7, 5)


def test_previous_week_dates() -> None:
    start, end = previous_week_dates(date(2026, 6, 29))

    assert start == date(2026, 6, 22)
    assert end == date(2026, 6, 28)


def test_current_month_dates() -> None:
    start, end = current_month_dates(date(2026, 2, 15))

    assert start == date(2026, 2, 1)
    assert end == date(2026, 2, 28)


def test_previous_month_dates_cross_year() -> None:
    start, end = previous_month_dates(date(2026, 1, 10))

    assert start == date(2025, 12, 1)
    assert end == date(2025, 12, 31)


def test_auto_morning_boundary_end_uses_next_local_midnight() -> None:
    ended_at = auto_morning_boundary_end(date(2026, 6, 28), "Europe/Kyiv")

    assert ended_at == datetime(2026, 6, 28, 21, 0, tzinfo=UTC)


def test_sleep_marker_before_active_start_targets_previous_day() -> None:
    sleep_time = datetime(2026, 7, 1, 1, 31, tzinfo=ZoneInfo("Europe/Kyiv"))

    assert sleep_marker_target_date(sleep_time, active_start="09:00") == date(2026, 6, 30)


def test_sleep_marker_after_active_start_targets_current_day() -> None:
    sleep_time = datetime(2026, 7, 1, 23, 31, tzinfo=ZoneInfo("Europe/Kyiv"))

    assert sleep_marker_target_date(sleep_time, active_start="09:00") == date(2026, 7, 1)


def test_period_bounds_follow_journal_active_start() -> None:
    start, end = _date_period_bounds(
        date(2026, 7, 1),
        date(2026, 7, 7),
        "Europe/Kyiv",
        active_start="09:00",
    )

    assert start == datetime(2026, 7, 1, 9, 0, tzinfo=ZoneInfo("Europe/Kyiv"))
    assert end == datetime(2026, 7, 8, 8, 59, 59, 999999, tzinfo=ZoneInfo("Europe/Kyiv"))


async def test_sleep_summary_after_midnight_closes_previous_journal_day(monkeypatch) -> None:
    class FakeSummaryService(SummaryService):
        async def generate_day_summary(self, session, *, user, day, close_day=False):
            assert day.local_date == date(2026, 6, 30)
            assert close_day is True
            return SimpleNamespace(short_text="Закрив вчора.")

    user = SimpleNamespace(id=uuid4(), timezone="Europe/Kyiv")
    day = SimpleNamespace(id=uuid4(), local_date=date(2026, 6, 30))
    sleep_time = datetime(2026, 7, 1, 1, 31, tzinfo=ZoneInfo("Europe/Kyiv"))
    calls = {"day_dates": [], "entries": []}

    async def get_user_settings(session, user_id):
        assert user_id == user.id
        return SimpleNamespace(active_start="09:00")

    async def get_or_create_day(session, *, user_id, local_date_value, started_at):
        assert user_id == user.id
        calls["day_dates"].append(local_date_value)
        return day

    async def get_day_by_date(session, *, user_id, local_date_value):
        assert user_id == user.id
        assert local_date_value == date(2026, 6, 30)
        return SimpleNamespace(id=day.id, local_date=date(2026, 6, 30), ended_at=None)

    async def add_entry(session, **kwargs):
        calls["entries"].append(kwargs)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(summaries_module.repo, "get_user_settings", get_user_settings)
    monkeypatch.setattr(summaries_module.repo, "get_or_create_day", get_or_create_day)
    monkeypatch.setattr(summaries_module.repo, "get_day_by_date", get_day_by_date)
    monkeypatch.setattr(summaries_module.repo, "add_entry", add_entry)
    monkeypatch.setattr(summaries_module, "local_now", lambda timezone: sleep_time)
    monkeypatch.setattr(summaries_module, "utc_now", lambda: datetime(2026, 6, 30, 22, 31, tzinfo=UTC))

    service = FakeSummaryService(
        SimpleNamespace(embeddings_enabled=False, embedding_api_key=None),
        ai_service=None,
    )

    summary = await service.close_today_with_summary(object(), user=user)

    assert summary.short_text == "Закрив вчора."
    assert calls["day_dates"] == [date(2026, 6, 30)]
    assert calls["entries"][0]["day_id"] == day.id
    assert calls["entries"][0]["local_timestamp"] == sleep_time
    assert calls["entries"][0]["meta"]["sleep_marker_target_date"] == "2026-06-30"


async def test_yesterday_summary_auto_closes_uncertain_day(monkeypatch) -> None:
    class FakeSummaryService(SummaryService):
        async def generate_day_summary(self, session, *, user, day, close_day=False):
            assert day.boundary_kind == "auto_morning"
            assert day.data_quality == "day_boundary_uncertain"
            assert close_day is False
            return SimpleNamespace(short_text="Підсумок готовий.")

    day = SimpleNamespace(
        id=uuid4(),
        local_date=date(2026, 6, 28),
        ended_at=None,
        boundary_kind="calendar",
        data_quality=None,
    )
    user = SimpleNamespace(id=uuid4(), timezone="Europe/Kyiv")
    close_calls = []

    async def get_day_by_date(session, *, user_id, local_date_value):
        assert user_id == user.id
        assert local_date_value == date(2026, 6, 28)
        return day

    async def list_day_entries(session, *, day_id):
        assert day_id == day.id
        return [SimpleNamespace(id=uuid4(), raw_text="щось було")]

    async def close_day(session, *, day_id, ended_at, boundary_kind, data_quality=None):
        close_calls.append(
            {
                "day_id": day_id,
                "ended_at": ended_at,
                "boundary_kind": boundary_kind,
                "data_quality": data_quality,
            }
        )

    async def summary_exists(session, *, user_id, day_id, period_type):
        return False

    async def get_user_settings(session, user_id):
        assert user_id == user.id
        return SimpleNamespace(active_start="09:00")

    async def current_journal_date(session, *, user, user_settings):
        return date(2026, 6, 29)

    monkeypatch.setattr(summaries_module.repo, "get_user_settings", get_user_settings)
    monkeypatch.setattr(summaries_module, "current_journal_date", current_journal_date)
    monkeypatch.setattr(summaries_module.repo, "get_day_by_date", get_day_by_date)
    monkeypatch.setattr(summaries_module.repo, "list_day_entries", list_day_entries)
    monkeypatch.setattr(summaries_module.repo, "close_day", close_day)
    monkeypatch.setattr(summaries_module.repo, "summary_exists", summary_exists)

    service = FakeSummaryService(
        SimpleNamespace(embeddings_enabled=False, embedding_api_key=None),
        ai_service=None,
    )

    summary = await service.generate_yesterday_summary_if_needed(object(), user=user)

    assert summary.short_text == "Підсумок готовий."
    assert close_calls == [
        {
            "day_id": day.id,
            "ended_at": datetime(2026, 6, 28, 21, 0, tzinfo=UTC),
            "boundary_kind": "auto_morning",
            "data_quality": "day_boundary_uncertain",
        }
    ]
    assert day.ended_at == datetime(2026, 6, 28, 21, 0, tzinfo=UTC)


def test_entries_query_text_compacts_recent_entries() -> None:
    entries = [
        SimpleNamespace(
            raw_text="лежу   і\nне можу почати",
            source="manual",
            local_timestamp=datetime(2026, 6, 29, 10, 30),
            created_at=None,
        )
    ]

    text = _entries_query_text(entries, label="daily summary 2026-06-29")

    assert text.startswith("daily summary 2026-06-29")
    assert "2026-06-29T10:30:00 [manual] лежу і не можу почати" in text


def test_semantic_records_context_truncates_source_text() -> None:
    records = [
        SimpleNamespace(
            target_type="entry",
            target_id="abc",
            created_at=datetime(2026, 6, 1, 12, 0),
            source_hash="hash",
            source_text="x" * 900,
        )
    ]

    context = _semantic_records_context(records)

    assert context[0]["target_id"] == "abc"
    assert context[0]["created_at"] == "2026-06-01T12:00:00"
    assert len(context[0]["source_text"]) == 700
    assert context[0]["source_text"].endswith("…")


def test_format_period_summary_weekly() -> None:
    summary = SimpleNamespace(
        period_type="weekly",
        period_start=date(2026, 6, 22),
        period_end=date(2026, 6, 28),
        short_text="Тиждень був нерівний.",
        details={
            "period_story": "Були провали й кілька ясніших вечорів.",
            "repeated_patterns": ["важкий старт після довгих пауз"],
            "changes_vs_previous_period": ["більше записів"],
            "activity_state_patterns": ["лежання + низька енергія"],
            "what_helped": ["вихід надвір"],
            "what_worsened": ["довгі прогалини"],
            "notable_days": ["середа"],
            "data_gaps": ["понеділок після обіду"],
            "cautious_observations": ["даних ще небагато"],
            "data_quality": "partial",
        },
    )

    text = format_period_summary(summary)

    assert "Тижневий підсумок" in text
    assert "Тиждень був нерівний." in text
    assert "- важкий старт після довгих пауз" in text
    assert "Якість даних: частково" in text
