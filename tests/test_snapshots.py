from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import mental_state_bot.services.snapshots as snapshots_module
from mental_state_bot.services.snapshots import _is_active_time, snapshot_question_context


def test_active_time_same_day_window() -> None:
    settings = SimpleNamespace(active_start="09:00", active_end="23:00")
    now = datetime(2026, 6, 29, 12, 0, tzinfo=ZoneInfo("UTC"))

    assert _is_active_time(now, "Europe/Kyiv", settings)


def test_active_time_overnight_window() -> None:
    settings = SimpleNamespace(active_start="22:00", active_end="03:00")
    now = datetime(2026, 6, 29, 23, 30, tzinfo=ZoneInfo("UTC"))

    assert _is_active_time(now, "Europe/Kyiv", settings)


def test_active_time_false_when_snapshots_paused() -> None:
    settings = SimpleNamespace(
        active_start="09:00",
        active_end="23:00",
        settings_json={"snapshots_paused": True},
    )
    now = datetime(2026, 6, 29, 12, 0, tzinfo=ZoneInfo("UTC"))

    assert not _is_active_time(now, "Europe/Kyiv", settings)


async def test_scheduled_snapshot_does_not_prompt_after_day_is_closed(monkeypatch) -> None:
    user = SimpleNamespace(id="user-id", timezone="Europe/Kyiv")
    user_settings = SimpleNamespace(
        active_start="09:00",
        active_end="23:30",
        min_interval_minutes=30,
        max_interval_minutes=70,
        reminder_delay_minutes=25,
        settings_json={},
    )
    closed_day = SimpleNamespace(ended_at=datetime(2026, 6, 29, 19, 45, tzinfo=ZoneInfo("UTC")))
    calls = {"sent": 0, "closed": 0}

    async def get_user_settings(session, user_id):
        return user_settings

    async def get_day_by_date(session, *, user_id, local_date_value):
        return closed_day

    async def get_open_snapshot(session, *, user_id):
        return None

    async def send_snapshot_prompt(*args, **kwargs):
        calls["sent"] += 1
        return True

    monkeypatch.setattr(snapshots_module.repo, "get_user_settings", get_user_settings)
    monkeypatch.setattr(snapshots_module.repo, "get_day_by_date", get_day_by_date)
    monkeypatch.setattr(snapshots_module.repo, "get_open_snapshot", get_open_snapshot)
    monkeypatch.setattr(snapshots_module, "send_snapshot_prompt", send_snapshot_prompt)
    monkeypatch.setattr(
        snapshots_module,
        "utc_now",
        lambda: datetime(2026, 6, 29, 20, 30, tzinfo=ZoneInfo("UTC")),
    )

    sent = await snapshots_module.maybe_send_scheduled_snapshot(
        object(),
        bot=object(),
        settings=SimpleNamespace(photo_prompt_chance=0.18),
        ai_service=object(),
        user=user,
    )

    assert sent is False
    assert calls["sent"] == 0


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
        day_entries=[entry],
        user_settings=settings,
        trigger="scheduled",
        photo_prompt_opportunity=True,
    )

    assert context["recent_entries"][0]["raw_text"] == "лежу і не можу почати"
    assert context["day_context"]["entries"][0]["raw_text"] == "лежу і не можу почати"
    assert context["day_context"]["entry_count"] == 1
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
