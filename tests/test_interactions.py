from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import mental_state_bot.services.interactions as interactions_module
from mental_state_bot.db.models import Snapshot
from mental_state_bot.services.interactions import (
    InteractionService,
    _clarification_bot_reply,
    _day_context,
    _latest_content_entry,
    _missing_core_metrics,
    _previous_affective_context,
    _recent_similar_clarification_exists,
    _snapshot_conversation_context,
    apply_emotion_calibration,
    apply_metric_calibration,
    interpretation_summary_reply,
    post_entry_reply,
)


async def test_snapshot_conversation_context_includes_latest_prompt_and_entries(monkeypatch) -> None:
    snapshot_id = uuid4()
    snapshot = SimpleNamespace(id=snapshot_id, context_json={"intent": "state_and_activity"})
    prompts = [
        SimpleNamespace(
            prompt_kind="initial",
            text="Як просувається робота над проєктом?",
            sent_at=None,
        ),
        SimpleNamespace(
            prompt_kind="clarification",
            text="Який трек опрацьовуєш?",
            sent_at=None,
        ),
    ]
    entries = [
        SimpleNamespace(
            created_at=None,
            local_timestamp=None,
            source="snapshot_response",
            raw_text='"Назва" прямо зараз',
        )
    ]

    async def get_snapshot_prompts(session, *, snapshot_id):
        return prompts

    async def list_snapshot_entries(session, *, snapshot_id):
        return entries

    monkeypatch.setattr(interactions_module.repo, "get_snapshot_prompts", get_snapshot_prompts)
    monkeypatch.setattr(interactions_module.repo, "list_snapshot_entries", list_snapshot_entries)

    context = await _snapshot_conversation_context(object(), snapshot=snapshot)

    assert context["latest_prompt"] == "Який трек опрацьовуєш?"
    assert context["entries"][0]["raw_text"] == '"Назва" прямо зараз'


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


def test_recent_similar_clarification_exists_blocks_semantic_duplicate() -> None:
    queue = [
        {
            "id": "q1",
            "question": "Як зараз із силами?",
            "reason": "missing_energy",
            "status": "queued",
        }
    ]

    assert _recent_similar_clarification_exists(
        queue,
        question="Як зараз із силами?",
    )


async def test_clarification_need_triggers_for_meaningful_missing_core_metric(monkeypatch) -> None:
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

    service = InteractionService(SimpleNamespace(), None)
    should_clarify, need = await service._clarification_need(
        object(),
        snapshot=SimpleNamespace(clarification_count=0),
        text="Працюю над трек і ніби нормально просувається",
        entry=SimpleNamespace(id=entry_id),
    )

    assert should_clarify is True
    assert need["reason"] == "missing_mood"
    assert need["missing_metrics"] == ["mood"]


async def test_clarification_need_triggers_when_analysis_requests_it(monkeypatch) -> None:
    entry_id = uuid4()
    feature_result = {
        "mood": {"value": "unclear", "confidence": 0.0},
        "energy": {"value": "somewhat_high", "confidence": 0.45},
        "needs_clarification": True,
        "clarification_question": "Це радше нормальний стан чи просто опис процесу?",
        "data_quality": "partial",
        "confidence": 0.7,
    }

    async def list_analyses_for_targets(session, *, target_type, target_ids):
        assert target_type == "entry"
        assert target_ids == [entry_id]
        return [SimpleNamespace(task_name="extract_entry_features", result=feature_result)]

    monkeypatch.setattr(interactions_module.repo, "list_analyses_for_targets", list_analyses_for_targets)

    service = InteractionService(SimpleNamespace(), None)
    should_clarify, need = await service._clarification_need(
        object(),
        snapshot=SimpleNamespace(clarification_count=0),
        text="Працюю над трек і ніби нормально просувається",
        entry=SimpleNamespace(id=entry_id),
    )

    assert should_clarify is True
    assert need["reason"] == "missing_mood"
    assert need["missing_metrics"] == ["mood"]
    assert need["suggested_question"] == "Це радше нормальний стан чи просто опис процесу?"


async def test_clarification_need_prioritizes_unclear_emotion_transition(monkeypatch) -> None:
    entry_id = uuid4()
    feature_result = {
        "emotion_needs_clarification": True,
        "emotion_transition": "unclear",
        "clarification_question": "Після попереднього моменту це відчуття ще трималося чи змінилося?",
        "data_quality": "partial",
        "confidence": 0.7,
    }

    async def list_analyses_for_targets(session, *, target_type, target_ids):
        return [SimpleNamespace(task_name="extract_entry_features", result=feature_result)]

    monkeypatch.setattr(interactions_module.repo, "list_analyses_for_targets", list_analyses_for_targets)

    should_clarify, need = await InteractionService(SimpleNamespace(), None)._clarification_need(
        object(),
        snapshot=SimpleNamespace(),
        text="Зараз уже вдома, роблю вечерю і слухаю музику",
        entry=SimpleNamespace(id=entry_id),
    )

    assert should_clarify is True
    assert need["reason"] == "emotion_transition_unclear"
    assert need["suggested_question"] == feature_result["clarification_question"]


async def test_previous_affective_context_keeps_only_current_evidenced_emotions(monkeypatch) -> None:
    first_id = uuid4()
    second_id = uuid4()
    day = SimpleNamespace(id=uuid4())
    entries = [
        SimpleNamespace(id=first_id, source="manual", local_timestamp=None),
        SimpleNamespace(id=second_id, source="manual", local_timestamp=None),
    ]
    analyses = [
        SimpleNamespace(
            task_name="extract_entry_features",
            target_id=first_id,
            result={
                "emotions": [
                    {
                        "label": "сум",
                        "intensity_level": "strong",
                        "confidence": 0.8,
                        "evidence": "дуже засмутився",
                        "time_scope": "current",
                    },
                    {
                        "label": "страх",
                        "evidence": "говорив про страх",
                        "time_scope": "mentioned_not_felt",
                    },
                ]
            },
        )
    ]

    async def list_day_entries(session, *, day_id):
        return entries

    async def list_analyses_for_targets(session, *, target_type, target_ids):
        return analyses

    monkeypatch.setattr(interactions_module.repo, "list_day_entries", list_day_entries)
    monkeypatch.setattr(interactions_module.repo, "list_analyses_for_targets", list_analyses_for_targets)

    context = await _previous_affective_context(object(), day=day, exclude_entry_id=second_id)

    assert context == [
        {
            "local_timestamp": None,
            "emotions": [
                {
                    "label": "сум",
                    "intensity_level": "strong",
                    "confidence": 0.8,
                    "evidence": "дуже засмутився",
                }
            ],
        }
    ]


async def test_apply_metric_calibration_writes_user_feature_analysis(monkeypatch) -> None:
    user_id = uuid4()
    entry_id = uuid4()
    day_id = uuid4()
    calls = {"analyses": [], "stale": []}
    feature_result = {
        "mood": {"value": "unclear", "confidence": 0.0},
        "energy": {"value": "neutral", "confidence": 0.8},
        "state_labels": [],
        "data_quality": "partial",
        "confidence": 0.5,
    }

    async def list_analyses_for_targets(session, *, target_type, target_ids):
        return [SimpleNamespace(task_name="extract_entry_features", result=feature_result)]

    async def add_ai_analysis(session, **kwargs):
        calls["analyses"].append(kwargs)
        return SimpleNamespace(id=uuid4())

    async def mark_day_summaries_stale(session, **kwargs):
        calls["stale"].append(kwargs)

    monkeypatch.setattr(interactions_module.repo, "list_analyses_for_targets", list_analyses_for_targets)
    monkeypatch.setattr(interactions_module.repo, "add_ai_analysis", add_ai_analysis)
    monkeypatch.setattr(interactions_module.repo, "mark_day_summaries_stale", mark_day_summaries_stale)

    replies = await apply_metric_calibration(
        object(),
        settings=SimpleNamespace(ai_provider="deepseek", ai_live_model="live"),
        user=SimpleNamespace(id=user_id),
        entry=SimpleNamespace(id=entry_id, day_id=day_id),
        metric="mood",
        score=6,
    )

    assert calls["analyses"][0]["provider"] == "user"
    assert calls["analyses"][0]["task_name"] == "extract_entry_features"
    assert calls["analyses"][0]["result"]["mood"] == {
        "value": "6",
        "confidence": 1.0,
        "source": "user_calibration",
    }
    assert calls["stale"][0]["reason"] == "metric_calibrated"
    assert replies[0].text.startswith("Записав")


async def test_interpretation_summary_reply_formats_current_features(monkeypatch) -> None:
    entry_id = uuid4()
    feature_result = {
        "mood": {"value": "high", "confidence": 0.8},
        "energy": {"value": "6", "confidence": 1.0},
        "emotion_labels": ["інтерес", "спокій"],
        "state_labels": [],
        "data_quality": "enough",
        "confidence": 0.82,
    }

    async def list_analyses_for_targets(session, *, target_type, target_ids):
        assert target_type == "entry"
        assert target_ids == [entry_id]
        return [SimpleNamespace(task_name="extract_entry_features", result=feature_result)]

    monkeypatch.setattr(interactions_module.repo, "list_analyses_for_targets", list_analyses_for_targets)

    reply = await interpretation_summary_reply(object(), entry=SimpleNamespace(id=entry_id))

    assert reply is not None
    assert "настрій високо" in reply.text
    assert "енергія 6/10" in reply.text
    assert "емоції: інтерес, спокій" in reply.text
    assert reply.keyboard == f"interpretation:{entry_id}"


async def test_apply_emotion_calibration_prepends_multiple_emotions_without_mixing_state_labels(monkeypatch) -> None:
    user_id = uuid4()
    entry_id = uuid4()
    calls = {"analyses": []}
    feature_result = {
        "mood": {"value": "neutral", "confidence": 0.7},
        "energy": {"value": "neutral", "confidence": 0.7},
        "emotion_labels": ["втома"],
        "state_labels": ["втома"],
        "data_quality": "partial",
        "confidence": 0.5,
    }

    async def list_analyses_for_targets(session, *, target_type, target_ids):
        return [SimpleNamespace(task_name="extract_entry_features", result=feature_result)]

    async def add_ai_analysis(session, **kwargs):
        calls["analyses"].append(kwargs)
        return SimpleNamespace(id=uuid4())

    async def mark_day_summaries_stale(session, **kwargs):
        pass

    monkeypatch.setattr(interactions_module.repo, "list_analyses_for_targets", list_analyses_for_targets)
    monkeypatch.setattr(interactions_module.repo, "add_ai_analysis", add_ai_analysis)
    monkeypatch.setattr(interactions_module.repo, "mark_day_summaries_stale", mark_day_summaries_stale)

    reply = await apply_emotion_calibration(
        object(),
        settings=SimpleNamespace(ai_provider="deepseek", ai_live_model="live"),
        user=SimpleNamespace(id=user_id),
        entry=SimpleNamespace(id=entry_id, day_id=None),
        emotions=["тривога", "сум", "тривога"],
        intensity_level="strong",
    )

    assert calls["analyses"][0]["result"]["emotion_labels"] == ["тривога", "сум"]
    assert calls["analyses"][0]["result"]["state_labels"] == ["втома"]
    assert calls["analyses"][0]["result"]["emotions"][:2] == [
        {
            "label": "тривога",
            "intensity_level": "strong",
            "intensity": 0.8,
            "confidence": 1.0,
            "evidence": "ручне уточнення користувача",
            "time_scope": "current",
        },
        {
            "label": "сум",
            "intensity_level": "strong",
            "intensity": 0.8,
            "confidence": 1.0,
            "evidence": "ручне уточнення користувача",
            "time_scope": "current",
        },
    ]
    assert "тривога — сильно" in reply.text
    assert "сум — сильно" in reply.text


async def test_apply_emotion_calibration_keeps_mentioned_emotions_out_of_current_signals(monkeypatch) -> None:
    user_id = uuid4()
    entry_id = uuid4()
    calls = {"analyses": []}
    feature_result = {
        "mood": {"value": "neutral", "confidence": 0.7},
        "energy": {"value": "neutral", "confidence": 0.7},
        "emotion_labels": ["страх", "спокій"],
        "state_labels": ["страх", "спокій"],
        "emotions": [
            {
                "label": "страх",
                "intensity_level": "strong",
                "intensity": 0.8,
                "confidence": 0.8,
                "evidence": "опис страху",
                "time_scope": "current",
            }
        ],
        "data_quality": "partial",
        "confidence": 0.5,
    }

    async def list_analyses_for_targets(session, *, target_type, target_ids):
        return [SimpleNamespace(task_name="extract_entry_features", result=feature_result)]

    async def add_ai_analysis(session, **kwargs):
        calls["analyses"].append(kwargs)
        return SimpleNamespace(id=uuid4())

    async def mark_day_summaries_stale(session, **kwargs):
        pass

    monkeypatch.setattr(interactions_module.repo, "list_analyses_for_targets", list_analyses_for_targets)
    monkeypatch.setattr(interactions_module.repo, "add_ai_analysis", add_ai_analysis)
    monkeypatch.setattr(interactions_module.repo, "mark_day_summaries_stale", mark_day_summaries_stale)

    reply = await apply_emotion_calibration(
        object(),
        settings=SimpleNamespace(ai_provider="deepseek", ai_live_model="live"),
        user=SimpleNamespace(id=user_id),
        entry=SimpleNamespace(id=entry_id, day_id=None),
        emotions=["страх", "сум"],
        emotion_intensity_levels={"страх": "strong", "сум": "mild"},
        time_scope="mentioned_not_felt",
    )

    result = calls["analyses"][0]["result"]
    assert result["emotion_labels"] == []
    assert result["state_labels"] == ["спокій"]
    assert result["mentioned_but_not_felt"] == ["страх", "сум"]
    assert result["emotions"][:2] == [
        {
            "label": "страх",
            "intensity_level": "strong",
            "intensity": 0.8,
            "confidence": 1.0,
            "evidence": "ручне уточнення користувача",
            "time_scope": "mentioned_not_felt",
        },
        {
            "label": "сум",
            "intensity_level": "mild",
            "intensity": 0.3,
            "confidence": 1.0,
            "evidence": "ручне уточнення користувача",
            "time_scope": "mentioned_not_felt",
        },
    ]
    assert "не були поточними" in reply.text


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
        raw_text='"Назва" прямо зараз',
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
            return SimpleNamespace(text="Я почув, що “Назва” — це назва елемента проєкту, який ти опрацьовуєш."), None

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
                text="Який трек опрацьовуєш?",
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
        correction_text='Ні, "Назва" це назва елемента проєкту',
        telegram_message_id=10,
        reply_to_message_id=None,
    )

    assert result.replies[-1].text == "Я почув, що “Назва” — це назва елемента проєкту, який ти опрацьовуєш."
    assert result.replies[-1].keyboard == f"correction:{target_id}"
    assert result.entry_id == target_id
    assert result.should_embed_entry is True
    assert calls["micro_context"]["latest_prompt"] == "Який трек опрацьовуєш?"
    assert calls["micro_context"]["original_entry"]["raw_text"] == '"Назва" прямо зараз'
    assert calls["feature_context"]["entry"] is target
    assert calls["feature_context"]["extra_context"]["latest_prompt"] == "Який трек опрацьовуєш?"
    assert calls["feature_context"]["extra_context"]["correction_text"] == 'Ні, "Назва" це назва елемента проєкту'
    assert [item["task_name"] for item in calls["analyses"]] == [
        "apply_correction",
        "generate_micro_summary",
    ]
    assert {item["target_id"] for item in calls["analyses"]} == {target_id}
    assert calls["stale"] == [{"user_id": user_id, "day_id": day_id, "reason": "entry_corrected"}]


async def test_post_entry_reply_combines_interpretation_and_single_next_step(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    entry = SimpleNamespace(id=uuid4())
    settings = SimpleNamespace(settings_json={})
    updates = []

    async def interpretation(session, *, entry):
        return interactions_module.BotReply("Як я це розмітив: настрій високо.", keyboard=f"interpretation:{entry.id}")

    async def calibrations(session, *, entry):
        return [
            interactions_module.BotReply(
                "Скільки зараз енергії приблизно від 0 до 10?",
                keyboard=f"metric_score:{entry.id}:energy",
            )
        ]

    async def update_user_settings(session, *, user_id, values):
        updates.append(values["settings_json"])
        return SimpleNamespace(settings_json=values["settings_json"])

    monkeypatch.setattr(interactions_module, "interpretation_summary_reply", interpretation)
    monkeypatch.setattr(interactions_module, "metric_calibration_replies", calibrations)
    monkeypatch.setattr(interactions_module.repo, "update_user_settings", update_user_settings)

    reply = await post_entry_reply(
        object(),
        user=user,
        user_settings=settings,
        entry=entry,
        micro_summary="Я почув, що ти почуваєшся добре.",
    )

    assert reply.text.count("\n\n") == 2
    assert "настрій високо" in reply.text
    assert "Скільки зараз енергії" in reply.text
    assert reply.keyboard == f"metric_score_with_correction:{entry.id}:energy"
    assert updates[0]["pending_post_entry_followup"]["entry_id"] == str(entry.id)
    assert updates[0]["pending_post_entry_followup"]["kind"] == "metric"


async def test_post_entry_reply_prioritizes_contextual_clarification(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    entry = SimpleNamespace(id=uuid4())

    async def unexpected_interpretation(session, *, entry):
        raise AssertionError("Immediate clarification should not be preceded by a dry interpretation")

    async def unexpected_calibration(session, *, entry):
        raise AssertionError("Numeric calibration should not displace contextual clarification")

    monkeypatch.setattr(interactions_module, "interpretation_summary_reply", unexpected_interpretation)
    monkeypatch.setattr(interactions_module, "metric_calibration_replies", unexpected_calibration)

    followup = _clarification_bot_reply(
        {
            "id": "clarification-id",
            "question": "Що зараз відбувається в цьому моменті?",
            "options": ["Відпочиваю", "Застряг", "Щось інше"],
        }
    )
    reply = await post_entry_reply(
        object(),
        user=user,
        user_settings=SimpleNamespace(settings_json={}),
        entry=entry,
        micro_summary="Записав момент.",
        immediate_followup=followup,
    )

    assert reply.text == "Записав момент.\n\nЩо зараз відбувається в цьому моменті?"
    assert reply.keyboard == "clarification:clarification-id"
    assert reply.keyboard_options == ("Відпочиваю", "Застряг", "Щось інше")
