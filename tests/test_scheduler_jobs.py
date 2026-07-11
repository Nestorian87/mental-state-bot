from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4
from zoneinfo import ZoneInfo

import mental_state_bot.scheduler.jobs as jobs_module
from mental_state_bot.scheduler.jobs import _maybe_send_queued_clarification, morning_summary_tick


class FakeSession:
    def begin(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


class FakeSessionMaker:
    def __call__(self):
        return FakeSession()


async def test_queued_clarification_waits_for_active_post_entry_followup() -> None:
    user = SimpleNamespace(id=uuid4(), timezone="Europe/Kyiv")
    settings = SimpleNamespace(
        settings_json={"pending_post_entry_followup": {"entry_id": "entry-id", "kind": "emotion"}},
        active_start="09:00",
        active_end="23:30",
    )

    assert not await _maybe_send_queued_clarification(
        FakeSession(),
        bot=object(),
        user=user,
        user_settings=settings,
    )


async def test_morning_summary_tick_clears_stale_sleep_reflection_even_without_new_summary(
    monkeypatch,
) -> None:
    user = SimpleNamespace(
        id=uuid4(),
        telegram_user_id=123,
        chat_id=456,
        timezone="Europe/Kyiv",
    )
    settings_json = {"pending_input": "sleep_reflection"}
    updates = []

    async def list_active_users(session):
        return [user]

    async def get_user_settings(session, user_id):
        assert user_id == user.id
        return SimpleNamespace(settings_json=settings_json)

    async def update_user_settings(session, *, user_id, values):
        updates.append({"user_id": user_id, "values": values})

    class FakeSummaryService:
        async def generate_yesterday_summary_if_needed(self, session, *, user):
            return None

    class FakeBot:
        async def send_message(self, **kwargs):
            raise AssertionError("no summary should be sent in this test")

    monkeypatch.setattr(jobs_module, "local_now", lambda timezone: datetime(2026, 7, 9, 9, 30, tzinfo=ZoneInfo(timezone)))
    monkeypatch.setattr(jobs_module.repo, "list_active_users", list_active_users)
    monkeypatch.setattr(jobs_module.repo, "get_user_settings", get_user_settings)
    monkeypatch.setattr(jobs_module.repo, "update_user_settings", update_user_settings)

    await morning_summary_tick(
        bot=FakeBot(),
        settings=SimpleNamespace(telegram_allowed_user_ids=[]),
        sessionmaker=FakeSessionMaker(),
        summary_service=FakeSummaryService(),
    )

    assert updates == [{"user_id": user.id, "values": {"settings_json": {}}}]


async def test_queued_clarification_is_sent_once_and_marked_active(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4(), chat_id=456, timezone="Europe/Kyiv")
    item = {"id": "q1", "entry_id": str(uuid4()), "question": "Що саме було найважчим?", "status": "queued"}
    settings = SimpleNamespace(
        settings_json={"clarification_queue": [item]},
        max_clarifications=2,
        active_start="09:00",
        active_end="23:30",
    )
    updates = []
    sent = []

    async def get_open_snapshot(session, *, user_id):
        return None

    async def update_user_settings(session, *, user_id, values):
        updates.append(values["settings_json"])
        settings.settings_json = values["settings_json"]
        return settings

    class FakeBot:
        async def send_message(self, **kwargs):
            sent.append(kwargs)

    monkeypatch.setattr(jobs_module, "local_now", lambda timezone: datetime(2026, 7, 9, 19, 0, tzinfo=ZoneInfo(timezone)))
    monkeypatch.setattr(jobs_module.repo, "get_open_snapshot", get_open_snapshot)
    monkeypatch.setattr(jobs_module.repo, "update_user_settings", update_user_settings)

    assert await _maybe_send_queued_clarification(
        FakeSession(), bot=FakeBot(), user=user, user_settings=settings
    )
    assert len(sent) == 1
    assert sent[0]["text"] == item["question"]
    assert settings.settings_json["clarification_queue"][0]["status"] == "active"
    assert settings.settings_json["pending_clarification"]["id"] == "q1"
    assert len(updates) == 2


async def test_queued_clarification_can_arrive_during_the_day(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4(), chat_id=456, timezone="Europe/Kyiv")
    item = {"id": "q1", "entry_id": str(uuid4()), "question": "Що з силами в цей момент?", "status": "queued"}
    settings = SimpleNamespace(
        settings_json={"clarification_queue": [item]},
        active_start="09:00",
        active_end="23:30",
    )
    sent = []

    async def get_open_snapshot(session, *, user_id):
        return None

    async def update_user_settings(session, *, user_id, values):
        settings.settings_json = values["settings_json"]
        return settings

    class FakeBot:
        async def send_message(self, **kwargs):
            sent.append(kwargs)

    monkeypatch.setattr(jobs_module, "local_now", lambda timezone: datetime(2026, 7, 9, 11, 0, tzinfo=ZoneInfo(timezone)))
    monkeypatch.setattr(jobs_module, "_is_active_time", lambda *args: True)
    monkeypatch.setattr(jobs_module.repo, "get_open_snapshot", get_open_snapshot)
    monkeypatch.setattr(jobs_module.repo, "update_user_settings", update_user_settings)

    assert await _maybe_send_queued_clarification(
        FakeSession(), bot=FakeBot(), user=user, user_settings=settings
    )
    assert sent[0]["text"] == item["question"]


async def test_queued_clarification_prefers_reason_not_used_today(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4(), chat_id=456, timezone="Europe/Kyiv")
    used = {
        "id": "used",
        "entry_id": str(uuid4()),
        "question": "Як був настрій?",
        "reason": "missing_mood",
        "status": "answered",
        "answered_at": "2026-07-09T18:20:00+03:00",
    }
    same_reason = {
        "id": "same",
        "entry_id": str(uuid4()),
        "question": "Що з настроєм?",
        "reason": "missing_mood",
        "status": "queued",
        "created_at": "2026-07-09T10:00:00+03:00",
    }
    other_reason = {
        "id": "other",
        "entry_id": str(uuid4()),
        "question": "А з силами як?",
        "reason": "missing_energy",
        "status": "queued",
        "created_at": "2026-07-09T11:00:00+03:00",
    }
    settings = SimpleNamespace(
        settings_json={"clarification_queue": [used, same_reason, other_reason]},
        max_clarifications=2,
        active_start="09:00",
        active_end="23:30",
    )
    sent = []

    async def get_open_snapshot(session, *, user_id):
        return None

    async def update_user_settings(session, *, user_id, values):
        settings.settings_json = values["settings_json"]
        return settings

    class FakeBot:
        async def send_message(self, **kwargs):
            sent.append(kwargs)

    monkeypatch.setattr(jobs_module, "local_now", lambda timezone: datetime(2026, 7, 9, 19, 0, tzinfo=ZoneInfo(timezone)))
    monkeypatch.setattr(jobs_module.repo, "get_open_snapshot", get_open_snapshot)
    monkeypatch.setattr(jobs_module.repo, "update_user_settings", update_user_settings)

    assert await _maybe_send_queued_clarification(FakeSession(), bot=FakeBot(), user=user, user_settings=settings)

    assert settings.settings_json["pending_clarification"]["id"] == "other"
    assert sent[0]["text"].startswith("Є 2 відкладені уточнення")


async def test_queued_clarification_ai_can_group_similar_items(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4(), chat_id=456, timezone="Europe/Kyiv")
    first_entry_id = uuid4()
    second_entry_id = uuid4()
    first = {
        "id": "first",
        "entry_id": str(first_entry_id),
        "question": "Після прогулянки це була втома чи нормальні сили?",
        "reason": "missing_energy",
        "status": "queued",
        "created_at": "2026-07-09T10:00:00+03:00",
    }
    second = {
        "id": "second",
        "entry_id": str(second_entry_id),
        "question": "Увечері сили були нормальні чи вже низькі?",
        "reason": "missing_energy_evening",
        "status": "queued",
        "created_at": "2026-07-09T11:00:00+03:00",
    }
    settings = SimpleNamespace(
        settings_json={"clarification_queue": [first, second]},
        max_clarifications=2,
        active_start="09:00",
        active_end="23:30",
    )
    sent = []

    async def get_open_snapshot(session, *, user_id):
        return None

    async def list_entries_by_ids(session, *, entry_ids):
        return [
            SimpleNamespace(id=first_entry_id, raw_text="Гуляв, але про сили не написав."),
            SimpleNamespace(id=second_entry_id, raw_text="Увечері щось робив, сили неясні."),
        ]

    async def update_user_settings(session, *, user_id, values):
        settings.settings_json = values["settings_json"]
        return settings

    class FakeAI:
        async def review_clarification_queue(self, session, *, user_id, context):
            assert [item["id"] for item in context["queued_items"]] == ["first", "second"]
            return (
                SimpleNamespace(
                    should_ask=True,
                    item_ids=["first", "second"],
                    question="За цими двома моментами сили радше тримались нормально чи вже просідали?",
                    options=["Трималися нормально", "Поступово просідали", "Було по-різному"],
                    reason="grouped_energy_uncertainty",
                    confidence=0.82,
                ),
                None,
            )

    class FakeBot:
        async def send_message(self, **kwargs):
            sent.append(kwargs)

    monkeypatch.setattr(jobs_module, "local_now", lambda timezone: datetime(2026, 7, 9, 19, 0, tzinfo=ZoneInfo(timezone)))
    monkeypatch.setattr(jobs_module.repo, "get_open_snapshot", get_open_snapshot)
    monkeypatch.setattr(jobs_module.repo, "list_entries_by_ids", list_entries_by_ids)
    monkeypatch.setattr(jobs_module.repo, "update_user_settings", update_user_settings)

    assert await _maybe_send_queued_clarification(
        FakeSession(), bot=FakeBot(), ai_service=FakeAI(), user=user, user_settings=settings
    )

    pending = settings.settings_json["pending_clarification"]
    assert pending["id"] == "first"
    assert pending["grouped_item_ids"] == ["first", "second"]
    assert pending["options"] == ["Трималися нормально", "Поступово просідали", "Було по-різному"]
    assert "сили радше тримались" in sent[0]["text"]
    assert [item["status"] for item in settings.settings_json["clarification_queue"]] == ["active", "active"]
