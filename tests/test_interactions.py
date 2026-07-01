from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import mental_state_bot.services.interactions as interactions_module
from mental_state_bot.db.models import Snapshot
from mental_state_bot.services.interactions import (
    InteractionService,
    _day_context,
    _latest_content_entry,
    _missing_core_metrics,
    _snapshot_conversation_context,
)


async def test_snapshot_conversation_context_includes_latest_prompt_and_entries(monkeypatch) -> None:
    snapshot_id = uuid4()
    snapshot = SimpleNamespace(id=snapshot_id, context_json={"intent": "state_and_activity"})
    prompts = [
        SimpleNamespace(
            prompt_kind="initial",
            text="Як просувається робота над альбомом?",
            sent_at=None,
        ),
        SimpleNamespace(
            prompt_kind="clarification",
            text="Який трек мастериш?",
            sent_at=None,
        ),
    ]
    entries = [
        SimpleNamespace(
            created_at=None,
            local_timestamp=None,
            source="snapshot_response",
            raw_text='"Море" прямо зараз',
        )
    ]

    async def get_snapshot_prompts(session, *, snapshot_id):
        return prompts

    async def list_snapshot_entries(session, *, snapshot_id):
        return entries

    monkeypatch.setattr(interactions_module.repo, "get_snapshot_prompts", get_snapshot_prompts)
    monkeypatch.setattr(interactions_module.repo, "list_snapshot_entries", list_snapshot_entries)

    context = await _snapshot_conversation_context(object(), snapshot=snapshot)

    assert context["latest_prompt"] == "Який трек мастериш?"
    assert context["entries"][0]["raw_text"] == '"Море" прямо зараз'


async def test_day_context_includes_all_day_entries(monkeypatch) -> None:
    day = SimpleNamespace(id=uuid4())
    entries = [
        SimpleNamespace(created_at=None, local_timestamp=None, source="manual", raw_text="ранкова кава"),
        SimpleNamespace(created_at=None, local_timestamp=None, source="snapshot_response", raw_text="мастеринг"),
    ]

    async def list_day_entries(session, *, day_id):
        assert day_id == day.id
        return entries

    monkeypatch.setattr(interactions_module.repo, "list_day_entries", list_day_entries)

    assert await _day_context(object(), day=day) == {
        "entry_count": 2,
        "omitted_entry_count": 0,
        "entries": [
            {
                "created_at": None,
                "local_timestamp": None,
                "source": "manual",
                "raw_text": "ранкова кава",
            },
            {
                "created_at": None,
                "local_timestamp": None,
                "source": "snapshot_response",
                "raw_text": "мастеринг",
            },
        ],
    }


async def test_day_context_reports_omitted_entries(monkeypatch) -> None:
    day = SimpleNamespace(id=uuid4())
    entries = [
        SimpleNamespace(created_at=None, local_timestamp=None, source="manual", raw_text=f"запис {index}")
        for index in range(3)
    ]

    async def list_day_entries(session, *, day_id):
        assert day_id == day.id
        return entries

    monkeypatch.setattr(interactions_module.repo, "list_day_entries", list_day_entries)

    assert await _day_context(object(), day=day, limit=2) == {
        "entry_count": 3,
        "omitted_entry_count": 1,
        "entries": [
        {
            "created_at": None,
            "local_timestamp": None,
            "source": "manual",
            "raw_text": "запис 1",
        },
        {
            "created_at": None,
            "local_timestamp": None,
            "source": "manual",
            "raw_text": "запис 2",
        },
        ],
    }


def test_missing_core_metrics_detects_unclear_mood_and_energy() -> None:
    features = interactions_module.EntryFeatures.model_validate(
        {
            "mood": {"value": "unclear", "confidence": 0.0},
            "energy": {"value": "unclear", "confidence": 0.0},
            "data_quality": "partial",
            "confidence": 0.7,
        }
    )

    assert _missing_core_metrics(features) == ["mood", "energy"]


async def test_clarification_need_triggers_for_missing_core_metrics(monkeypatch) -> None:
    entry_id = uuid4()
    feature_result = {
        "mood": {"value": "unclear", "confidence": 0.0},
        "energy": {"value": "somewhat_high", "confidence": 0.45},
        "data_quality": "partial",
        "confidence": 0.7,
    }

    async def list_analyses_for_targets(session, *, target_type, target_ids):
        assert target_type == "entry"
        assert target_ids == [entry_id]
        return [SimpleNamespace(task_name="extract_entry_features", result=feature_result)]

    monkeypatch.setattr(interactions_module.repo, "list_analyses_for_targets", list_analyses_for_targets)

    service = InteractionService(SimpleNamespace(max_clarifications_per_snapshot=2), None)
    should_clarify, need = await service._clarification_need(
        object(),
        snapshot=SimpleNamespace(clarification_count=0),
        text="Мастерю трек і ніби нормально просувається",
        entry=SimpleNamespace(id=entry_id),
    )

    assert should_clarify is True
    assert need["reason"] == "missing_mood"
    assert need["missing_metrics"] == ["mood"]


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


def test_latest_content_entry_ignores_button_entries() -> None:
    content = SimpleNamespace(source="snapshot_response")
    button = SimpleNamespace(source="button_later")

    assert _latest_content_entry([content, button]) is content


async def test_record_button_action_does_not_create_entry_for_later(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4(), timezone="Europe/Kyiv")
    snapshot = SimpleNamespace(id=uuid4())
    calls = {"closed": []}

    async def get_open_snapshot(session, *, user_id):
        assert user_id == user.id
        return snapshot

    async def list_snapshot_entries(session, *, snapshot_id):
        assert snapshot_id == snapshot.id
        return []

    async def close_snapshot(session, *, snapshot_id, status="closed"):
        calls["closed"].append({"snapshot_id": snapshot_id, "status": status})

    async def add_entry(session, **kwargs):
        raise AssertionError("button controls must not create diary entries")

    monkeypatch.setattr(interactions_module.repo, "get_open_snapshot", get_open_snapshot)
    monkeypatch.setattr(interactions_module.repo, "list_snapshot_entries", list_snapshot_entries)
    monkeypatch.setattr(interactions_module.repo, "close_snapshot", close_snapshot)
    monkeypatch.setattr(interactions_module.repo, "add_entry", add_entry)

    service = InteractionService(SimpleNamespace(), None)
    result = await service.record_button_action(object(), user=user, action="later")

    assert result.entry_id is None
    assert result.should_embed_entry is False
    assert calls["closed"] == [{"snapshot_id": snapshot.id, "status": "postponed"}]


async def test_record_button_action_embeds_existing_snapshot_answer_without_new_entry(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4(), timezone="Europe/Kyiv")
    snapshot = SimpleNamespace(id=uuid4())
    entry = SimpleNamespace(id=uuid4(), source="snapshot_response")

    async def get_open_snapshot(session, *, user_id):
        assert user_id == user.id
        return snapshot

    async def list_snapshot_entries(session, *, snapshot_id):
        assert snapshot_id == snapshot.id
        return [entry]

    async def close_snapshot(session, *, snapshot_id, status="closed"):
        assert snapshot_id == snapshot.id
        assert status == "closed_by_user"

    async def add_entry(session, **kwargs):
        raise AssertionError("button controls must not create diary entries")

    monkeypatch.setattr(interactions_module.repo, "get_open_snapshot", get_open_snapshot)
    monkeypatch.setattr(interactions_module.repo, "list_snapshot_entries", list_snapshot_entries)
    monkeypatch.setattr(interactions_module.repo, "close_snapshot", close_snapshot)
    monkeypatch.setattr(interactions_module.repo, "add_entry", add_entry)

    service = InteractionService(SimpleNamespace(), None)
    result = await service.record_button_action(object(), user=user, action="as_is")

    assert result.entry_id == entry.id
    assert result.should_embed_entry is True
    assert result.replies[0].keyboard == "correction"


async def test_record_correction_returns_revised_summary_with_correction_keyboard(monkeypatch) -> None:
    user_id = uuid4()
    day_id = uuid4()
    snapshot_id = uuid4()
    target_id = uuid4()
    target = SimpleNamespace(
        id=target_id,
        day_id=day_id,
        snapshot_id=snapshot_id,
        source="snapshot_response",
        raw_text='"Море" прямо зараз',
        created_at=None,
        local_timestamp=None,
    )
    snapshot = SimpleNamespace(id=snapshot_id, context_json={"intent": "state_and_activity"})
    user = SimpleNamespace(id=user_id, timezone="Europe/Kyiv")
    calls = {"feature_context": None, "micro_context": None, "analyses": [], "stale": []}

    class FakeSession:
        async def get(self, model, item_id):
            if model is Snapshot:
                assert item_id == snapshot_id
                return snapshot
            return SimpleNamespace(id=item_id)

    class FakeAI:
        async def generate_micro_summary(self, session, *, user_id, context):
            calls["micro_context"] = context
            return SimpleNamespace(text="Я почув, що “Море” — це назва треку, який ти мастериш."), None

    async def get_recent_entries(session, *, user_id, limit):
        return [target]

    async def get_user_settings(session, user_id):
        return SimpleNamespace(
            tone="calm",
            humanity_level="balanced",
            settings_json={},
        )

    async def list_day_entries(session, *, day_id):
        return [target]

    async def get_snapshot_prompts(session, *, snapshot_id):
        return [
            SimpleNamespace(
                prompt_kind="clarification",
                text="Який трек мастериш?",
                sent_at=None,
            )
        ]

    async def list_snapshot_entries(session, *, snapshot_id):
        return [target]

    async def add_ai_analysis(session, **kwargs):
        calls["analyses"].append(kwargs)

    async def analyze_entry_features(session, **kwargs):
        calls["feature_context"] = kwargs

    async def mark_day_summaries_stale(session, **kwargs):
        calls["stale"].append(kwargs)
        return 1

    monkeypatch.setattr(interactions_module.repo, "get_recent_entries", get_recent_entries)
    monkeypatch.setattr(interactions_module.repo, "get_user_settings", get_user_settings)
    monkeypatch.setattr(interactions_module.repo, "list_day_entries", list_day_entries)
    monkeypatch.setattr(interactions_module.repo, "get_snapshot_prompts", get_snapshot_prompts)
    monkeypatch.setattr(interactions_module.repo, "list_snapshot_entries", list_snapshot_entries)
    monkeypatch.setattr(interactions_module.repo, "add_ai_analysis", add_ai_analysis)
    monkeypatch.setattr(interactions_module.repo, "mark_day_summaries_stale", mark_day_summaries_stale)
    monkeypatch.setattr(interactions_module, "analyze_entry_features", analyze_entry_features)

    service = InteractionService(SimpleNamespace(ai_provider="deepseek", ai_live_model="flash"), FakeAI())
    result = await service.record_correction(
        FakeSession(),
        user=user,
        correction_text='Ні, "Море" це назва треку',
        telegram_message_id=10,
        reply_to_message_id=None,
    )

    assert result.replies[-1].text == "Я почув, що “Море” — це назва треку, який ти мастериш."
    assert result.replies[-1].keyboard == f"correction:{target_id}"
    assert result.entry_id == target_id
    assert result.should_embed_entry is True
    assert calls["micro_context"]["latest_prompt"] == "Який трек мастериш?"
    assert calls["micro_context"]["original_entry"]["raw_text"] == '"Море" прямо зараз'
    assert calls["feature_context"]["entry"] is target
    assert calls["feature_context"]["extra_context"]["latest_prompt"] == "Який трек мастериш?"
    assert calls["feature_context"]["extra_context"]["correction_text"] == 'Ні, "Море" це назва треку'
    assert [item["task_name"] for item in calls["analyses"]] == [
        "apply_correction",
        "generate_micro_summary",
    ]
    assert {item["target_id"] for item in calls["analyses"]} == {target_id}
    assert calls["stale"] == [{"user_id": user_id, "day_id": day_id, "reason": "entry_corrected"}]
