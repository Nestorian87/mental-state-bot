from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from mental_state_bot.services.analysis_backfill import entry_feature_context


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
