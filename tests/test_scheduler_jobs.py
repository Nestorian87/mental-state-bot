from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4
from zoneinfo import ZoneInfo

import mental_state_bot.scheduler.jobs as jobs_module
from mental_state_bot.scheduler.jobs import build_scheduler, morning_summary_tick


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


def test_scheduler_does_not_automatically_send_deferred_clarifications() -> None:
    scheduler = build_scheduler(
        bot=object(),
        settings=SimpleNamespace(app_timezone="Europe/Kyiv"),
        sessionmaker=object(),
        ai_service=object(),
        summary_service=object(),
    )

    assert "clarification_queue_tick" not in {job.id for job in scheduler.get_jobs()}


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

    monkeypatch.setattr(
        jobs_module,
        "local_now",
        lambda timezone: datetime(2026, 7, 9, 9, 30, tzinfo=ZoneInfo(timezone)),
    )
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
