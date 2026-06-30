from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import mental_state_bot.services.analysis_backfill as analysis_backfill_module
from mental_state_bot.services.analysis_backfill import (
    backfill_entry_features,
    entry_feature_context,
)


def test_entry_feature_context_includes_raw_time_metadata_and_extra_context() -> None:
    entry = SimpleNamespace(
        raw_text="лежу і не можу почати",
        source="snapshot_response",
        created_at=datetime(2026, 6, 29, 9, 0),
        local_timestamp=datetime(2026, 6, 29, 12, 0),
        meta={"button_action": None},
    )

    context = entry_feature_context(
        entry,
        extra_context={"snapshot_context": {"recent_pattern": "short replies"}, "backfill": False},
    )

    assert context == {
        "raw_text": "лежу і не можу почати",
        "source": "snapshot_response",
        "created_at": "2026-06-29T09:00:00",
        "local_timestamp": "2026-06-29T12:00:00",
        "metadata": {"button_action": None},
        "backfill": False,
        "snapshot_context": {"recent_pattern": "short replies"},
    }


async def test_force_backfill_selects_existing_entries_for_reanalysis(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    entry = SimpleNamespace(id=uuid4())
    calls = {"all_entries": 0, "missing_entries": 0, "analyzed": 0}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def begin(self):
            return self

    class FakeSessionMaker:
        def __call__(self):
            return FakeSession()

    async def get_user_by_telegram_id(session, telegram_user_id):
        return user

    async def list_user_entries(session, *, user_id, limit):
        calls["all_entries"] += 1
        return [entry]

    async def list_entries_without_analysis(session, **kwargs):
        calls["missing_entries"] += 1
        return []

    async def get_entry(session, entry_id):
        return entry

    async def analyze_entry_features(session, **kwargs):
        calls["analyzed"] += 1

    monkeypatch.setattr(analysis_backfill_module.repo, "get_user_by_telegram_id", get_user_by_telegram_id)
    monkeypatch.setattr(analysis_backfill_module.repo, "list_user_entries", list_user_entries)
    monkeypatch.setattr(
        analysis_backfill_module.repo,
        "list_entries_without_analysis",
        list_entries_without_analysis,
    )
    monkeypatch.setattr(analysis_backfill_module.repo, "get_entry", get_entry)
    monkeypatch.setattr(analysis_backfill_module, "analyze_entry_features", analyze_entry_features)

    result = await backfill_entry_features(
        settings=SimpleNamespace(),
        ai_service=object(),
        sessionmaker=FakeSessionMaker(),
        telegram_user_id=123,
        limit=10,
        force=True,
    )

    assert result.selected == 1
    assert result.processed == 1
    assert calls == {"all_entries": 1, "missing_entries": 0, "analyzed": 1}
