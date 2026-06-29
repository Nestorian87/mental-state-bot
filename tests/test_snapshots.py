from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from mental_state_bot.services.snapshots import _is_active_time, snapshot_question_context


def test_active_time_same_day_window() -> None:
    settings = SimpleNamespace(active_start="09:00", active_end="23:00")
    now = datetime(2026, 6, 29, 12, 0, tzinfo=ZoneInfo("UTC"))

    assert _is_active_time(now, "Europe/Kiev", settings)


def test_active_time_overnight_window() -> None:
    settings = SimpleNamespace(active_start="22:00", active_end="03:00")
    now = datetime(2026, 6, 29, 23, 30, tzinfo=ZoneInfo("UTC"))

    assert _is_active_time(now, "Europe/Kiev", settings)


def test_active_time_false_when_snapshots_paused() -> None:
    settings = SimpleNamespace(
        active_start="09:00",
        active_end="23:00",
        settings_json={"snapshots_paused": True},
    )
    now = datetime(2026, 6, 29, 12, 0, tzinfo=ZoneInfo("UTC"))

    assert not _is_active_time(now, "Europe/Kiev", settings)


def test_snapshot_question_context_includes_photo_and_body_preferences() -> None:
    entry = SimpleNamespace(
        created_at=datetime(2026, 6, 29, 10, 0),
        source="manual",
        raw_text="лежу і не можу почати",
    )
    settings = SimpleNamespace(
        tone="calm",
        humanity_level="balanced",
        ask_body_signals=True,
        photo_prompts_enabled=True,
        settings_json={"custom_interaction_style": "не звучати як терапевт"},
    )

    context = snapshot_question_context(
        recent_entries=[entry],
        user_settings=settings,
        trigger="scheduled",
        photo_prompt_opportunity=True,
    )

    assert context["recent_entries"][0]["raw_text"] == "лежу і не можу почати"
    assert context["style"]["custom_interaction_style"] == "не звучати як терапевт"
    assert context["question_preferences"]["ask_body_signals"] is True
    assert context["question_preferences"]["photo_prompts_enabled"] is True
    assert context["question_preferences"]["photo_prompt_opportunity"] is True
    assert "можна необов'язково запропонувати фото" in context["question_preferences"]["photo_prompt_style"]


def test_snapshot_question_context_disables_photo_prompt_style() -> None:
    settings = SimpleNamespace(
        tone="calm",
        humanity_level="balanced",
        ask_body_signals=False,
        photo_prompts_enabled=False,
    )

    context = snapshot_question_context(
        recent_entries=[],
        user_settings=settings,
        trigger="manual",
    )

    assert context["question_preferences"]["photo_prompt_style"] == "Не проси фото."
    assert context["question_preferences"]["body_signal_style"] == (
        "Не став окремих питань про тілесні сигнали."
    )
