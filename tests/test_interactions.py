from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import mental_state_bot.services.interactions as interactions_module
from mental_state_bot.db.models import Snapshot
from mental_state_bot.services.interactions import InteractionService


async def test_record_missed_reason_resolves_prompt_and_closes_snapshot(monkeypatch) -> None:
    user_id = uuid4()
    snapshot_id = uuid4()
    day_id = uuid4()
    missed_id = uuid4()
    entry_id = uuid4()
    user = SimpleNamespace(id=user_id, timezone="Europe/Kyiv")
    missed = SimpleNamespace(id=missed_id, snapshot_id=snapshot_id)
    snapshot = SimpleNamespace(id=snapshot_id, day_id=day_id)
    calls = {"resolved": [], "closed": [], "entries": [], "analyses": []}

    class FakeSession:
        async def get(self, model, item_id):
            assert model is Snapshot
            assert item_id == snapshot_id
            return snapshot

    async def get_latest_open_missed_prompt(session, *, user_id):
        assert user_id == user.id
        return missed

    async def resolve_missed_prompt(session, *, missed_prompt_id, reason_text, status="explained"):
        calls["resolved"].append(
            {
                "missed_prompt_id": missed_prompt_id,
                "reason_text": reason_text,
                "status": status,
            }
        )
        return missed

    async def close_snapshot(session, *, snapshot_id, status="closed"):
        calls["closed"].append({"snapshot_id": snapshot_id, "status": status})

    async def add_entry(session, **kwargs):
        calls["entries"].append(kwargs)
        return SimpleNamespace(id=entry_id)

    async def analyze_entry_features(session, **kwargs):
        calls["analyses"].append(kwargs)

    monkeypatch.setattr(
        interactions_module.repo,
        "get_latest_open_missed_prompt",
        get_latest_open_missed_prompt,
    )
    monkeypatch.setattr(interactions_module.repo, "resolve_missed_prompt", resolve_missed_prompt)
    monkeypatch.setattr(interactions_module.repo, "close_snapshot", close_snapshot)
    monkeypatch.setattr(interactions_module.repo, "add_entry", add_entry)
    monkeypatch.setattr(interactions_module, "analyze_entry_features", analyze_entry_features)

    service = InteractionService(SimpleNamespace(ai_provider="deepseek", ai_live_model="flash"), None)
    result = await service.record_missed_reason(
        FakeSession(),
        user=user,
        reason_text="не було ресурсу відповідати",
        reason_code="no_capacity",
    )

    assert result.entry_id == entry_id
    assert result.should_embed_entry is True
    assert calls["resolved"] == [
        {
            "missed_prompt_id": missed_id,
            "reason_text": "не було ресурсу відповідати",
            "status": "explained",
        }
    ]
    assert calls["closed"] == [{"snapshot_id": snapshot_id, "status": "missed_explained"}]
    assert calls["entries"][0]["source"] == "missed_reason"
    assert calls["entries"][0]["day_id"] == day_id
    assert calls["entries"][0]["snapshot_id"] == snapshot_id
    assert calls["entries"][0]["raw_text"] == "причина пропуску: не було ресурсу відповідати"
    assert calls["entries"][0]["meta"] == {
        "missed_prompt_id": str(missed_id),
        "reason_code": "no_capacity",
    }
    assert calls["analyses"][0]["entry"].id == entry_id


async def test_record_missed_reason_without_open_prompt_is_only_a_reply(monkeypatch) -> None:
    async def get_latest_open_missed_prompt(session, *, user_id):
        return None

    monkeypatch.setattr(
        interactions_module.repo,
        "get_latest_open_missed_prompt",
        get_latest_open_missed_prompt,
    )

    service = InteractionService(SimpleNamespace(), None)
    result = await service.record_missed_reason(
        object(),
        user=SimpleNamespace(id=uuid4(), timezone="Europe/Kyiv"),
        reason_text="зайнятий",
        reason_code="busy",
    )

    assert result.entry_id is None
    assert result.should_embed_entry is False
    assert "Не бачу відкритого" in result.replies[0].text
