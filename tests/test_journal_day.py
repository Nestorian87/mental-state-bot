from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from mental_state_bot.services import journal_day as journal_day_module
from mental_state_bot.services.journal_day import current_journal_date


@pytest.mark.asyncio
async def test_current_journal_date_keeps_before_active_start_in_open_previous_day(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4(), timezone="Europe/Kyiv")
    settings = SimpleNamespace(active_start="09:00")
    previous_day = SimpleNamespace(local_date=date(2026, 7, 1), ended_at=None)

    async def get_day_by_date(session, *, user_id, local_date_value):
        assert local_date_value == date(2026, 7, 1)
        return previous_day

    monkeypatch.setattr(journal_day_module.repo, "get_day_by_date", get_day_by_date)

    result = await current_journal_date(
        object(),
        user=user,
        user_settings=settings,
        now=datetime(2026, 7, 1, 22, 44, tzinfo=UTC),
    )

    assert result == date(2026, 7, 1)


@pytest.mark.asyncio
async def test_current_journal_date_moves_to_new_day_if_previous_day_closed(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4(), timezone="Europe/Kyiv")
    settings = SimpleNamespace(active_start="09:00")
    previous_day = SimpleNamespace(
        local_date=date(2026, 7, 1),
        ended_at=datetime(2026, 7, 1, 22, 31, tzinfo=UTC),
    )

    async def get_day_by_date(session, *, user_id, local_date_value):
        assert local_date_value == date(2026, 7, 1)
        return previous_day

    monkeypatch.setattr(journal_day_module.repo, "get_day_by_date", get_day_by_date)

    result = await current_journal_date(
        object(),
        user=user,
        user_settings=settings,
        now=datetime(2026, 7, 1, 22, 44, tzinfo=UTC),
    )

    assert result == date(2026, 7, 2)
