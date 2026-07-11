from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import mental_state_bot.services.snapshots as snapshots_module
from mental_state_bot.services.snapshots import (
    _ensure_photo_prompt_if_requested,
    _is_active_time,
    _next_snapshot_interval,
    _question_query_text,
    _snapshot_reminder_text,
    _verified_semantic_memory_insight,
    daily_rhythm_context,
    maybe_send_scheduled_snapshot,
    snapshot_question_context,
)


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


def test_active_time_false_during_quiet_until() -> None:
    settings = SimpleNamespace(
        active_start="09:00",
        active_end="23:00",
        settings_json={"quiet_until": "2026-06-29T13:00:00+00:00"},
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
        settings_json={"adaptive_observation_enabled": False},
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


async def test_scheduled_snapshot_waits_after_late_answer(monkeypatch) -> None:
    user = SimpleNamespace(id="user-id", timezone="Europe/Kyiv")
    user_settings = SimpleNamespace(
        active_start="09:00",
        active_end="23:30",
        min_interval_minutes=20,
        max_interval_minutes=20,
        reminder_delay_minutes=25,
        settings_json={"adaptive_observation_enabled": False},
    )
    now = datetime(2026, 6, 29, 12, 0, tzinfo=ZoneInfo("UTC"))
    last_snapshot = SimpleNamespace(
        prompted_at=datetime(2026, 6, 29, 11, 0, tzinfo=ZoneInfo("UTC")),
        closed_at=datetime(2026, 6, 29, 11, 55, tzinfo=ZoneInfo("UTC")),
    )
    calls = {"sent": 0}

    async def get_user_settings(session, user_id):
        return user_settings

    async def get_day_by_date(session, *, user_id, local_date_value):
        return None

    async def get_open_snapshot(session, *, user_id):
        return None

    async def get_last_snapshot(session, *, user_id):
        return last_snapshot

    async def send_snapshot_prompt(*args, **kwargs):
        calls["sent"] += 1
        return True

    monkeypatch.setattr(snapshots_module.repo, "get_user_settings", get_user_settings)
    monkeypatch.setattr(snapshots_module.repo, "get_day_by_date", get_day_by_date)
    monkeypatch.setattr(snapshots_module.repo, "get_open_snapshot", get_open_snapshot)
    monkeypatch.setattr(snapshots_module.repo, "get_last_snapshot", get_last_snapshot)
    monkeypatch.setattr(snapshots_module, "send_snapshot_prompt", send_snapshot_prompt)
    monkeypatch.setattr(snapshots_module, "utc_now", lambda: now)

    sent = await snapshots_module.maybe_send_scheduled_snapshot(
        object(),
        bot=object(),
        settings=SimpleNamespace(photo_prompt_chance=0.18),
        ai_service=object(),
        user=user,
    )

    assert sent is False
    assert calls["sent"] == 0


async def test_scheduled_snapshot_waits_after_recent_manual_entry(monkeypatch) -> None:
    user = SimpleNamespace(id="user-id", timezone="Europe/Kyiv")
    user_settings = SimpleNamespace(
        active_start="09:00",
        active_end="23:30",
        min_interval_minutes=20,
        max_interval_minutes=20,
        reminder_delay_minutes=25,
        settings_json={},
    )
    now = datetime(2026, 6, 29, 12, 0, tzinfo=ZoneInfo("UTC"))
    recent_entry = SimpleNamespace(created_at=datetime(2026, 6, 29, 11, 55, tzinfo=ZoneInfo("UTC")))
    calls = {"sent": 0}

    async def get_user_settings(session, user_id):
        return user_settings

    async def get_day_by_date(session, *, user_id, local_date_value):
        return None

    async def get_open_snapshot(session, *, user_id):
        return None

    async def get_last_snapshot(session, *, user_id):
        return None

    async def get_latest_observation_entry(session, *, user_id):
        return recent_entry

    async def send_snapshot_prompt(*args, **kwargs):
        calls["sent"] += 1
        return True

    monkeypatch.setattr(snapshots_module.repo, "get_user_settings", get_user_settings)
    monkeypatch.setattr(snapshots_module.repo, "get_day_by_date", get_day_by_date)
    monkeypatch.setattr(snapshots_module.repo, "get_open_snapshot", get_open_snapshot)
    monkeypatch.setattr(snapshots_module.repo, "get_last_snapshot", get_last_snapshot)
    monkeypatch.setattr(snapshots_module.repo, "get_latest_observation_entry", get_latest_observation_entry)
    monkeypatch.setattr(snapshots_module, "send_snapshot_prompt", send_snapshot_prompt)
    monkeypatch.setattr(snapshots_module, "utc_now", lambda: now)

    sent = await snapshots_module.maybe_send_scheduled_snapshot(
        object(),
        bot=object(),
        settings=SimpleNamespace(photo_prompt_chance=0.18),
        ai_service=object(),
        user=user,
    )

    assert sent is False
    assert calls["sent"] == 0


async def test_adaptive_interval_uses_a_stable_bounded_ai_window(monkeypatch) -> None:
    entry = SimpleNamespace(id="entry-1")
    settings = SimpleNamespace(
        min_interval_minutes=30,
        max_interval_minutes=70,
        settings_json={},
    )

    async def list_analyses_for_targets(session, *, target_type, target_ids):
        assert target_type == "entry"
        assert target_ids == ["entry-1"]
        return [
            SimpleNamespace(
                task_name="extract_entry_features",
                result={
                    "observation_cadence": {
                        "next_checkin_min_minutes": 20,
                        "next_checkin_max_minutes": 45,
                        "confidence": 0.8,
                        "volatility": "moving",
                        "change_likelihood": "high",
                        "eventfulness": "medium",
                        "reason": "подія ще розвивається",
                    }
                },
            )
        ]

    monkeypatch.setattr(snapshots_module.repo, "list_analyses_for_targets", list_analyses_for_targets)

    first_minutes, first_context = await _next_snapshot_interval(object(), user_settings=settings, entry=entry)
    second_minutes, second_context = await _next_snapshot_interval(object(), user_settings=settings, entry=entry)

    assert 30 <= first_minutes <= 45
    assert first_minutes == second_minutes
    assert first_context["mode"] == "adaptive"
    assert first_context == second_context


async def test_adaptive_interval_respects_disabled_preference() -> None:
    entry = SimpleNamespace(id="entry-1")
    settings = SimpleNamespace(
        min_interval_minutes=40,
        max_interval_minutes=40,
        settings_json={"adaptive_observation_enabled": False},
    )

    minutes, context = await _next_snapshot_interval(object(), user_settings=settings, entry=entry)

    assert minutes == 40
    assert context["mode"] == "settings_random"


def test_semantic_memory_insight_keeps_only_retrieved_evidence() -> None:
    insight = _verified_semantic_memory_insight(
        {
            "used": True,
            "hypothesis": "схожа ситуація може бути важливою",
            "evidence_entry_ids": ["entry-a", "invented"],
            "confidence": 0.8,
        },
        [{"target_id": "entry-a"}, {"target_id": "entry-b"}],
    )

    assert insight is not None
    assert insight["evidence_entry_ids"] == ["entry-a"]


def test_semantic_memory_insight_requires_real_evidence() -> None:
    insight = _verified_semantic_memory_insight(
        {
            "used": True,
            "hypothesis": "схожа ситуація може бути важливою",
            "evidence_entry_ids": ["invented"],
            "confidence": 0.8,
        },
        [{"target_id": "entry-a"}],
    )

    assert insight is None


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
    assert "явно додай необов'язкову можливість" in context["question_preferences"]["photo_prompt_style"]
    assert context["current_day_part"] in {"morning", "daytime", "evening", "night"}
    assert context["daily_rhythm"]["source"] == "insufficient_history"


def test_snapshot_question_context_accepts_daily_rhythm() -> None:
    settings = SimpleNamespace(
        tone="calm",
        humanity_level="balanced",
        ask_body_signals=True,
        photo_prompts_enabled=True,
        settings_json={},
        timezone="Europe/Kyiv",
    )

    context = snapshot_question_context(
        recent_entries=[],
        day_entries=[],
        user_settings=settings,
        trigger="scheduled",
        daily_rhythm={
            "average_first_entry_time_last_7_days": "10:20",
            "average_first_entry_label": "орієнтовний час старту дня, не точний час пробудження",
            "days_sampled": 5,
            "current_day_part": "morning",
            "minutes_since_average_first_entry": 35,
            "source": "first_meaningful_entry_after_journal_day_boundary",
        },
    )

    assert context["daily_rhythm"]["average_first_entry_time_last_7_days"] == "10:20"
    assert context["daily_rhythm"]["days_sampled"] == 5


async def test_daily_rhythm_context_uses_first_meaningful_entries(monkeypatch) -> None:
    user = SimpleNamespace(id="user-id", timezone="Europe/Kyiv")
    current_day = SimpleNamespace(local_date=date(2026, 7, 9))
    day_1 = SimpleNamespace(id="day-1")
    day_2 = SimpleNamespace(id="day-2")

    async def list_days_between(session, *, user_id, start_date, end_date):
        assert user_id == "user-id"
        assert start_date == date(2026, 7, 2)
        assert end_date == date(2026, 7, 8)
        return [day_1, day_2]

    async def list_day_entries(session, *, day_id):
        if day_id == "day-1":
            return [
                SimpleNamespace(
                    source="sleep_marker",
                    raw_text="лягаю спати",
                    local_timestamp=datetime(2026, 7, 2, 1, 0, tzinfo=ZoneInfo("Europe/Kyiv")),
                ),
                SimpleNamespace(
                    source="manual",
                    raw_text="прокинувся",
                    local_timestamp=datetime(2026, 7, 2, 10, 0, tzinfo=ZoneInfo("Europe/Kyiv")),
                ),
            ]
        return [
            SimpleNamespace(
                source="correction",
                raw_text="виправлення",
                local_timestamp=datetime(2026, 7, 3, 9, 0, tzinfo=ZoneInfo("Europe/Kyiv")),
            ),
            SimpleNamespace(
                source="manual",
                raw_text="ранок",
                local_timestamp=datetime(2026, 7, 3, 11, 0, tzinfo=ZoneInfo("Europe/Kyiv")),
            ),
        ]

    monkeypatch.setattr(snapshots_module.repo, "list_days_between", list_days_between)
    monkeypatch.setattr(snapshots_module.repo, "list_day_entries", list_day_entries)
    monkeypatch.setattr(
        snapshots_module,
        "utc_now",
        lambda: datetime(2026, 7, 9, 8, 30, tzinfo=ZoneInfo("UTC")),
    )

    context = await daily_rhythm_context(object(), user=user, current_day=current_day)

    assert context["average_first_entry_time_last_7_days"] == "10:30"
    assert context["days_sampled"] == 2
    assert context["source"] == "first_meaningful_entry_after_journal_day_boundary"


def test_snapshot_question_context_marks_empty_scheduled_day_as_morning_start() -> None:
    previous_entry = SimpleNamespace(
        created_at=datetime(2026, 6, 29, 22, 0),
        local_timestamp=None,
        source="manual",
        raw_text="Вчора ввечері працював над проєктом",
    )
    settings = SimpleNamespace(
        tone="calm",
        humanity_level="balanced",
        ask_body_signals=True,
        photo_prompts_enabled=True,
    )

    context = snapshot_question_context(
        recent_entries=[previous_entry],
        day_entries=[],
        user_settings=settings,
        trigger="scheduled",
    )

    assert context["day_phase"] == "morning_start"
    assert "сон" in _question_query_text(context)
    assert "проєктом" not in _question_query_text(context)


def test_snapshot_question_context_keeps_manual_empty_day_regular() -> None:
    settings = SimpleNamespace(
        tone="calm",
        humanity_level="balanced",
        ask_body_signals=True,
        photo_prompts_enabled=True,
    )

    context = snapshot_question_context(
        recent_entries=[],
        day_entries=[],
        user_settings=settings,
        trigger="manual",
    )

    assert context["day_phase"] == "regular"


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


def test_ensure_photo_prompt_adds_fallback_when_ai_omits_photo() -> None:
    text = _ensure_photo_prompt_if_requested(
        "Що зараз відбувається?",
        {
            "question_preferences": {
                "photo_prompts_enabled": True,
                "photo_prompt_opportunity": True,
            }
        },
    )

    assert "Що зараз відбувається?" in text
    assert "фото" in text


def test_ensure_photo_prompt_does_not_duplicate_existing_photo_mention() -> None:
    original = "Що зараз відбувається? Можеш показати фото цього моменту."

    text = _ensure_photo_prompt_if_requested(
        original,
        {
            "question_preferences": {
                "photo_prompts_enabled": True,
                "photo_prompt_opportunity": True,
            }
        },
    )

    assert text == original


def test_ensure_photo_prompt_ignores_regular_questions() -> None:
    original = "Що зараз відбувається?"

    text = _ensure_photo_prompt_if_requested(
        original,
        {
            "question_preferences": {
                "photo_prompts_enabled": True,
                "photo_prompt_opportunity": False,
            }
        },
    )

    assert text == original


def test_snapshot_reminder_text_does_not_repeat_old_template_or_question(monkeypatch) -> None:
    prompt = SimpleNamespace(
        prompt_kind="initial",
        text="Як зараз просувається робота над проєктом?",
    )
    monkeypatch.setattr(snapshots_module.random, "choice", lambda values: values[0])

    text = _snapshot_reminder_text([prompt])

    assert "М’яко повертаюся" not in text
    assert prompt.text not in text
    assert "зріз" in text


def test_snapshot_reminder_text_mentions_clarification_when_open(monkeypatch) -> None:
    prompts = [
        SimpleNamespace(prompt_kind="initial", text="Що зараз відбувається?"),
        SimpleNamespace(prompt_kind="clarification", text="А що саме стало важчим?"),
    ]
    monkeypatch.setattr(snapshots_module.random, "choice", lambda values: values[0])

    text = _snapshot_reminder_text(prompts)

    assert "уточнення" in text
    assert prompts[-1].text not in text


async def test_scheduled_snapshot_waits_for_active_post_entry_followup(monkeypatch) -> None:
    user = SimpleNamespace(id="user-id", timezone="Europe/Kyiv")
    settings_view = SimpleNamespace(
        settings_json={"pending_post_entry_followup": {"entry_id": "entry-id", "kind": "metric"}}
    )

    async def get_user_settings(session, user_id):
        assert user_id == user.id
        return settings_view

    monkeypatch.setattr(snapshots_module.repo, "get_user_settings", get_user_settings)

    assert not await maybe_send_scheduled_snapshot(
        object(),
        bot=object(),
        settings=SimpleNamespace(),
        ai_service=object(),
        user=user,
    )
