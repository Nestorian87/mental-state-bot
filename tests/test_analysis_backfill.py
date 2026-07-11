from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import mental_state_bot.services.analysis_backfill as analysis_backfill_module
from mental_state_bot.ai.schemas import (
    AffectiveStateSignal,
    EmotionSignal,
    EntryFeatures,
    FeatureValue,
)
from mental_state_bot.services.analysis_backfill import (
    _correction_history,
    _manual_metric_overrides,
    _restore_manual_metric_overrides,
    backfill_entry_features,
    entry_feature_context,
    guided_reanalyze_entry_features,
    postprocess_entry_features,
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


def test_correction_history_preserves_order_and_adds_the_current_correction() -> None:
    analyses = [
        SimpleNamespace(
            task_name="apply_correction",
            result={
                "correction_text": "Перше уточнення.",
                "kind": "clarification_answer",
                "question": "Що змінилося?",
                "corrected_at": "2026-07-11T10:00:00+03:00",
            },
        ),
        SimpleNamespace(
            task_name="extract_entry_features",
            result={},
        ),
    ]

    history = _correction_history(
        analyses,
        extra_context={"correction_text": "Пізніше уточнення.", "clarification_context": {"question": "А зараз?"}},
    )

    assert [item["text"] for item in history] == ["Перше уточнення.", "Пізніше уточнення."]
    assert history[-1]["kind"] == "clarification_answer"
    assert history[-1]["question"] == "А зараз?"


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

    async def list_user_entries(session, *, user_id, limit, descending=False):
        calls["all_entries"] += 1
        assert descending is True
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


async def test_guided_reanalysis_reports_coverage_changes(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    entry = SimpleNamespace(id=uuid4())

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

    previous = SimpleNamespace(
        task_name="extract_entry_features",
        target_id=entry.id,
        result={"should_graph_mood": False, "should_graph_energy": False, "emotion_observation": "unclear"},
    )

    async def get_user_by_telegram_id(session, telegram_user_id):
        return user

    async def list_user_entries(session, *, user_id, limit, descending=False):
        assert descending is True
        return [entry]

    async def list_analyses_for_targets(session, *, target_type, target_ids):
        return [previous]

    async def get_entry(session, entry_id):
        return entry

    async def analyze_entry_features(session, **kwargs):
        return EntryFeatures(
            should_graph_mood=True,
            should_graph_energy=True,
            emotion_observation="observed",
            emotions=[EmotionSignal(label="радість", intensity_level="moderate")],
        )

    monkeypatch.setattr(analysis_backfill_module.repo, "get_user_by_telegram_id", get_user_by_telegram_id)
    monkeypatch.setattr(analysis_backfill_module.repo, "list_user_entries", list_user_entries)
    monkeypatch.setattr(analysis_backfill_module.repo, "list_analyses_for_targets", list_analyses_for_targets)
    monkeypatch.setattr(analysis_backfill_module.repo, "get_entry", get_entry)
    monkeypatch.setattr(analysis_backfill_module, "analyze_entry_features", analyze_entry_features)

    result = await guided_reanalyze_entry_features(
        settings=SimpleNamespace(),
        ai_service=object(),
        sessionmaker=FakeSessionMaker(),
        telegram_user_id=123,
        scope="recent",
        limit=10,
    )

    assert result.processed == 1
    assert result.changed == 1
    assert result.before.mood_points == 0
    assert result.after.mood_points == 1
    assert result.after.energy_points == 1
    assert result.after.observed_emotion_points == 1


def test_manual_metric_override_survives_later_ai_analysis() -> None:
    manual = SimpleNamespace(
        task_name="extract_entry_features",
        result={
            "energy": {"value": "8", "confidence": 1.0},
            "energy_evidence": "ручне уточнення користувача",
            "energy_reasoning_type": "user_manual",
            "should_graph_energy": True,
        },
    )
    later_ai = SimpleNamespace(
        task_name="extract_entry_features",
        result={
            "energy": {"value": "unclear", "confidence": 0.0},
            "energy_reasoning_type": "unclear",
            "should_graph_energy": False,
        },
    )
    features = EntryFeatures()

    _restore_manual_metric_overrides(features, _manual_metric_overrides([manual, later_ai]))

    assert features.energy.value == "8"
    assert features.energy.confidence == 1.0
    assert features.energy_reasoning_type == "user_manual"
    assert features.should_graph_energy is True


def test_postprocess_normalizes_observation_cadence_window() -> None:
    features = EntryFeatures(
        observation_cadence={
            "next_checkin_min_minutes": 80,
            "next_checkin_max_minutes": 30,
            "confidence": 0.7,
        }
    )

    result = postprocess_entry_features(features, "звичайний запис")

    assert result.observation_cadence.next_checkin_min_minutes == 30
    assert result.observation_cadence.next_checkin_max_minutes == 80


def test_postprocess_clears_metrics_without_evidence() -> None:
    features = EntryFeatures(
        entry_type="photo_only",
        activity_labels=["прогулянка"],
        state_labels=["спокій"],
        emotion_labels=["радість"],
        mood=FeatureValue(value="8", confidence=0.8),
        energy=FeatureValue(value="7", confidence=0.8),
        anxiety=FeatureValue(value="low", confidence=0.7),
        pleasant_moments=["фото"],
        data_quality="enough",
        confidence=0.9,
    )

    result = postprocess_entry_features(features, "[photo]")

    assert result.mood.value == "unclear"
    assert result.energy.value == "unclear"
    assert result.anxiety.value == "unclear"
    assert result.entry_type == "photo_only"
    assert result.mood_evidence is None
    assert result.should_graph_mood is False
    assert result.should_graph_energy is False
    assert result.observation_cadence.next_checkin_min_minutes is None


def test_postprocess_respects_ai_entry_type_without_keyword_inference() -> None:
    features = EntryFeatures(
        entry_type="reply_fragment",
        activity_labels=["сон"],
        state_labels=["виснаження"],
        emotion_labels=["тривога"],
        mood=FeatureValue(value="very_low", confidence=0.8),
        energy=FeatureValue(value="low", confidence=0.8),
        anxiety=FeatureValue(value="high", confidence=0.8),
        data_quality="partial",
        confidence=0.8,
    )

    result = postprocess_entry_features(features, "Не знаю")

    assert result.mood.value == "unclear"
    assert result.energy.value == "unclear"
    assert result.anxiety.value == "unclear"
    assert result.entry_type == "reply_fragment"
    assert result.should_graph_mood is False
    assert result.should_graph_energy is False


def test_postprocess_keeps_only_current_emotions_with_evidence() -> None:
    features = EntryFeatures(
        entry_type="current_state",
        emotions=[
            EmotionSignal(
                label="тривога",
                intensity_level="mild",
                intensity=0.1,
                confidence=0.8,
                evidence="не дуже сильно тривожить",
                time_scope="current",
            ),
            EmotionSignal(
                label="страх",
                intensity_level="strong",
                intensity=0.8,
                confidence=0.9,
                evidence="говорили про страх",
                time_scope="mentioned_not_felt",
            ),
            EmotionSignal(
                label="радість",
                intensity_level="moderate",
                intensity=0.55,
                confidence=0.8,
                evidence=None,
                time_scope="current",
            ),
        ],
        data_quality="enough",
        confidence=0.8,
    )

    result = postprocess_entry_features(features, "не дуже сильно тривожить, говорили про страх")

    assert [emotion.label for emotion in result.emotions] == ["тривога"]
    assert result.emotions[0].intensity == 0.3
    assert result.emotion_labels == ["тривога"]
    assert "страх" in result.mentioned_but_not_felt
    assert result.emotion_observation == "observed"


def test_postprocess_does_not_turn_invalid_or_photo_emotions_into_zero_observations() -> None:
    invalid = EntryFeatures(
        entry_type="current_state",
        emotions=[
            EmotionSignal(
                label="радість",
                intensity_level="moderate",
                confidence=0.8,
                evidence=None,
                time_scope="current",
            )
        ],
        emotion_observation="observed",
    )
    photo = EntryFeatures(entry_type="photo_only", emotion_observation="no_current_emotion")

    assert postprocess_entry_features(invalid, "звичайний запис").emotion_observation == "unclear"
    assert postprocess_entry_features(photo, "[photo]").emotion_observation == "unclear"


def test_postprocess_separates_controlled_affective_states_from_emotions() -> None:
    features = EntryFeatures(
        entry_type="current_state",
        emotions=[
            EmotionSignal(
                label="спокій",
                intensity_level="moderate",
                confidence=0.8,
                evidence="зараз спокійно",
                time_scope="current",
            ),
            EmotionSignal(
                label="радість",
                intensity_level="strong",
                confidence=0.8,
                evidence="радісно",
                time_scope="current",
            ),
            EmotionSignal(
                label="вигадана мітка",
                intensity_level="strong",
                confidence=0.8,
                evidence="щось відчуваю",
                time_scope="current",
            ),
        ],
        affective_states=[
            AffectiveStateSignal(
                label="напруга",
                intensity_level="mild",
                confidence=0.8,
                evidence="трохи напружено",
                time_scope="current",
            )
        ],
    )

    result = postprocess_entry_features(features, "спокійно і радісно, але трохи напружено")

    assert [emotion.label for emotion in result.emotions] == ["радість"]
    assert [state.label for state in result.affective_states] == ["напруга", "спокій"]
    assert result.emotion_labels == ["радість"]
    assert result.state_labels == ["напруга", "спокій"]
    assert any("вигадана мітка відхилено" in note for note in result.uncertainty_notes)


def test_postprocess_blocks_graphing_for_ai_blocked_entry_type() -> None:
    features = EntryFeatures(
        entry_type="dream",
        activity_labels=["sleeping"],
        state_labels=["спокій"],
        mood=FeatureValue(value="8", confidence=0.8),
        mood_evidence="було страшно уві сні",
        mood_reasoning_type="direct_text",
        energy=FeatureValue(value="9", confidence=0.8),
        energy_evidence="після сну опису сил немає",
        energy_reasoning_type="weak_text",
        data_quality="enough",
        confidence=0.8,
    )

    result = postprocess_entry_features(features, "Снився дуже реалістичний сон")

    assert result.mood.value == "high"
    assert result.energy.value == "very_high"
    assert result.entry_type == "dream"
    assert result.should_graph_mood is False
    assert result.should_graph_energy is False


def test_postprocess_clears_metadata_only_metric_even_with_evidence() -> None:
    features = EntryFeatures(
        entry_type="sleep",
        activity_labels=["прогулянка"],
        mood=FeatureValue(value="somewhat_low", confidence=0.7),
        mood_evidence="8 годин сну",
        mood_reasoning_type="metadata_only",
        energy=FeatureValue(value="high", confidence=0.7),
        energy_evidence="8 годин сну",
        energy_reasoning_type="metadata_only",
        anxiety=FeatureValue(value="low", confidence=0.7),
        anxiety_evidence="8 годин сну",
        anxiety_reasoning_type="metadata_only",
        data_quality="enough",
        confidence=0.8,
    )

    result = postprocess_entry_features(features, "8 годин спав")

    assert result.mood.value == "unclear"
    assert result.energy.value == "unclear"
    assert result.anxiety.value == "unclear"
    assert result.entry_type == "sleep"
    assert result.should_graph_mood is False
    assert result.should_graph_energy is False


def test_postprocess_keeps_direct_mood_and_energy_evidence() -> None:
    features = EntryFeatures(
        entry_type="current_state",
        mood=FeatureValue(value="8", confidence=0.75),
        mood_evidence="Настрій дуже гарний",
        mood_reasoning_type="direct_text",
        energy=FeatureValue(value="9", confidence=0.75),
        energy_evidence="повно сил",
        energy_reasoning_type="direct_text",
        anxiety=FeatureValue(value="low", confidence=0.6),
        anxiety_evidence="тривоги не описано прямо",
        anxiety_reasoning_type="weak_text",
        data_quality="partial",
        confidence=0.75,
    )

    result = postprocess_entry_features(features, "Настрій дуже гарний і повно сил")

    assert result.mood.value == "high"
    assert result.energy.value == "very_high"
    assert result.anxiety.value == "low"
    assert result.entry_type == "current_state"
    assert result.mood_evidence == "Настрій дуже гарний"
    assert result.energy_evidence == "повно сил"
    assert result.should_graph_mood is True
    assert result.should_graph_energy is True
