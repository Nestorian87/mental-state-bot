from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.ai.schemas import (
    AffectiveStateSignal,
    EmotionSignal,
    EntryFeatures,
    FeatureValue,
    ObservationCadence,
)
from mental_state_bot.ai.service import AIService
from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import Entry
from mental_state_bot.emotions import (
    CANONICAL_AFFECTIVE_STATES,
    CANONICAL_EMOTIONS,
    EMOTION_INTENSITY_VALUES,
)

ENTRY_FEATURES_TASK = "extract_entry_features"
ENTRY_FEATURES_SCHEMA_VERSION = "entry_features.v5"
_FEATURE_VALUES = {
    "very_low",
    "low",
    "somewhat_low",
    "neutral",
    "somewhat_high",
    "high",
    "very_high",
    "unclear",
}
_GRAPH_BLOCKED_ENTRY_TYPES = {
    "activity_only",
    "dream",
    "photo_only",
    "reply_fragment",
    "command_or_system",
}
_GRAPH_REASONING_TYPES = {"direct_text", "user_manual"}
_METRICS = ("mood", "energy", "anxiety")
_EMOTION_INTENSITY = EMOTION_INTENSITY_VALUES
_EMOTION_LABELS = set(CANONICAL_EMOTIONS)
_AFFECTIVE_STATE_LABELS = set(CANONICAL_AFFECTIVE_STATES)


@dataclass(frozen=True)
class FeatureBackfillResult:
    selected: int
    processed: int
    skipped_missing: int = 0


@dataclass(frozen=True)
class FeatureCoverage:
    mood_points: int = 0
    energy_points: int = 0
    observed_emotion_points: int = 0


@dataclass(frozen=True)
class GuidedReanalysisResult(FeatureBackfillResult):
    before: FeatureCoverage = FeatureCoverage()
    after: FeatureCoverage = FeatureCoverage()
    changed: int = 0


async def backfill_entry_features(
    *,
    settings: Settings,
    ai_service: AIService,
    sessionmaker,
    telegram_user_id: int,
    limit: int,
    force: bool = False,
) -> FeatureBackfillResult:
    async with sessionmaker() as session, session.begin():
        user = await repo.get_user_by_telegram_id(session, telegram_user_id)
        if user is None:
            raise ValueError(f"Unknown Telegram user id: {telegram_user_id}")
        if force:
            entries = await repo.list_user_entries(session, user_id=user.id, limit=limit, descending=True)
        else:
            entries = await repo.list_entries_without_analysis(
                session,
                user_id=user.id,
                task_name=ENTRY_FEATURES_TASK,
                limit=limit,
            )
        entry_ids = [entry.id for entry in entries]
        user_id = user.id

    processed = 0
    skipped_missing = 0
    for entry_id in entry_ids:
        async with sessionmaker() as session, session.begin():
            entry = await repo.get_entry(session, entry_id=entry_id)
            if entry is None:
                skipped_missing += 1
                continue
            await analyze_entry_features(
                session,
                settings=settings,
                ai_service=ai_service,
                user_id=user_id,
                entry=entry,
            )
            processed += 1
    return FeatureBackfillResult(
        selected=len(entry_ids),
        processed=processed,
        skipped_missing=skipped_missing,
    )


async def count_entry_feature_reanalysis(
    *,
    sessionmaker,
    telegram_user_id: int,
    scope: Literal["recent", "range", "all"],
    limit: int | None = None,
    start_date=None,
    end_date=None,
) -> int:
    async with sessionmaker() as session, session.begin():
        user = await repo.get_user_by_telegram_id(session, telegram_user_id)
        if user is None:
            raise ValueError(f"Unknown Telegram user id: {telegram_user_id}")
        entries = await _entries_for_scope(
            session,
            user_id=user.id,
            scope=scope,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
        )
        return len(entries)


async def guided_reanalyze_entry_features(
    *,
    settings: Settings,
    ai_service: AIService,
    sessionmaker,
    telegram_user_id: int,
    scope: Literal["recent", "range", "all"],
    limit: int | None = None,
    start_date=None,
    end_date=None,
) -> GuidedReanalysisResult:
    async with sessionmaker() as session, session.begin():
        user = await repo.get_user_by_telegram_id(session, telegram_user_id)
        if user is None:
            raise ValueError(f"Unknown Telegram user id: {telegram_user_id}")
        entries = await _entries_for_scope(
            session,
            user_id=user.id,
            scope=scope,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
        )
        entry_ids = [entry.id for entry in entries]
        analyses = await repo.list_analyses_for_targets(session, target_type="entry", target_ids=entry_ids)
        before_by_id = _latest_feature_results(analyses)
        user_id = user.id

    processed = 0
    skipped_missing = 0
    changed = 0
    after_by_id: dict[str, dict[str, Any]] = {}
    for entry_id in entry_ids:
        async with sessionmaker() as session, session.begin():
            entry = await repo.get_entry(session, entry_id=entry_id)
            if entry is None:
                skipped_missing += 1
                continue
            features = await analyze_entry_features(
                session,
                settings=settings,
                ai_service=ai_service,
                user_id=user_id,
                entry=entry,
            )
            result = features.model_dump()
            key = str(entry_id)
            after_by_id[key] = result
            if before_by_id.get(key) != result:
                changed += 1
            processed += 1

    return GuidedReanalysisResult(
        selected=len(entry_ids),
        processed=processed,
        skipped_missing=skipped_missing,
        before=_feature_coverage(before_by_id.values()),
        after=_feature_coverage(after_by_id.values()),
        changed=changed,
    )


async def _entries_for_scope(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    scope: Literal["recent", "range", "all"],
    limit: int | None,
    start_date,
    end_date,
) -> list[Entry]:
    if scope == "recent":
        if not limit or limit < 1:
            raise ValueError("A positive limit is required for a recent reanalysis")
        return list(await repo.list_user_entries(session, user_id=user_id, limit=limit, descending=True))
    if scope == "range":
        if start_date is None or end_date is None:
            raise ValueError("A journal date range is required for a range reanalysis")
        return list(
            await repo.list_entries_for_journal_dates(
                session,
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
        )
    if scope == "all":
        return list(await repo.list_user_entries(session, user_id=user_id, limit=None))
    raise ValueError(f"Unknown reanalysis scope: {scope}")


def _latest_feature_results(analyses) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for analysis in analyses:
        if analysis.task_name == ENTRY_FEATURES_TASK and isinstance(analysis.result, dict):
            latest[str(analysis.target_id)] = analysis.result
    return latest


def _feature_coverage(results) -> FeatureCoverage:
    values = list(results)
    return FeatureCoverage(
        mood_points=sum(bool(item.get("should_graph_mood")) for item in values),
        energy_points=sum(bool(item.get("should_graph_energy")) for item in values),
        observed_emotion_points=sum(
            str(item.get("emotion_observation") or "") == "observed" for item in values
        ),
    )


async def analyze_entry_features(
    session: AsyncSession,
    *,
    settings: Settings,
    ai_service: AIService,
    user_id: uuid.UUID,
    entry: Entry,
    extra_context: dict[str, Any] | None = None,
) -> EntryFeatures:
    existing_analyses = await repo.list_analyses_for_targets(
        session,
        target_type="entry",
        target_ids=[entry.id],
    )
    manual_metric_overrides = _manual_metric_overrides(existing_analyses)
    correction_history = _correction_history(existing_analyses, extra_context=extra_context)
    context = {
        **(extra_context or {}),
        "correction_history": correction_history,
    }
    features, model_run_id = await ai_service.extract_entry_features(
        session,
        user_id=user_id,
        context=entry_feature_context(entry, extra_context=context),
    )
    features = postprocess_entry_features(features, entry.raw_text or "")
    _restore_manual_metric_overrides(features, manual_metric_overrides)
    await repo.add_ai_analysis(
        session,
        user_id=user_id,
        target_type="entry",
        target_id=entry.id,
        task_name=ENTRY_FEATURES_TASK,
        schema_version=ENTRY_FEATURES_SCHEMA_VERSION,
        provider=settings.ai_provider,
        model=settings.ai_live_model,
        result=features.model_dump(),
        confidence=Decimal(str(features.confidence)),
        uncertainty_notes=features.uncertainty_notes,
        model_run_id=model_run_id,
    )
    return features


def _manual_metric_overrides(analyses) -> dict[str, dict[str, Any]]:
    """Find explicit user metric values even if a later AI pass was saved afterwards."""
    overrides: dict[str, dict[str, Any]] = {}
    for analysis in reversed(list(analyses)):
        if analysis.task_name != ENTRY_FEATURES_TASK or not isinstance(analysis.result, dict):
            continue
        for metric in _METRICS:
            if metric in overrides:
                continue
            if str(analysis.result.get(f"{metric}_reasoning_type") or "") != "user_manual":
                continue
            value = analysis.result.get(metric)
            if not isinstance(value, dict) or str(value.get("value") or "") in {"", "unclear"}:
                continue
            overrides[metric] = {
                metric: value,
                f"{metric}_evidence": analysis.result.get(f"{metric}_evidence") or "ручне уточнення користувача",
                f"{metric}_reasoning_type": "user_manual",
                f"should_graph_{metric}": True,
            }
    return overrides


def _correction_history(analyses, *, extra_context: dict[str, Any] | None) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for analysis in analyses:
        if analysis.task_name != "apply_correction" or not isinstance(analysis.result, dict):
            continue
        text = " ".join(str(analysis.result.get("correction_text") or "").split())
        if not text:
            continue
        history.append(
            {
                "text": text[:900],
                "kind": str(analysis.result.get("kind") or "correction"),
                "question": " ".join(str(analysis.result.get("question") or "").split())[:600] or None,
                "recorded_at": analysis.result.get("corrected_at"),
            }
        )
    current_text = " ".join(str((extra_context or {}).get("correction_text") or "").split())
    if current_text and (not history or history[-1].get("text") != current_text):
        clarification = (extra_context or {}).get("clarification_context") or {}
        history.append(
            {
                "text": current_text[:900],
                "kind": "clarification_answer" if clarification else "correction",
                "question": " ".join(str(clarification.get("question") or "").split())[:600] or None,
                "recorded_at": None,
            }
        )
    return history[-16:]


def _restore_manual_metric_overrides(features: EntryFeatures, overrides: dict[str, dict[str, Any]]) -> None:
    for values in overrides.values():
        for field, value in values.items():
            if field in _METRICS and isinstance(value, dict):
                setattr(features, field, FeatureValue.model_validate(value))
            else:
                setattr(features, field, value)


def postprocess_entry_features(features: EntryFeatures, raw_text: str) -> EntryFeatures:
    """Normalize AI output without keyword-based interpretation."""
    _ = raw_text
    normalized = features.model_copy(deep=True)
    for metric in _METRICS:
        _normalize_metric(normalized, metric)
    normalized.social_activity = _normalize_feature_value(normalized.social_activity)
    normalized.used_context = bool(normalized.used_context)
    if normalized.used_context and normalized.context_inference == "none":
        normalized.context_inference = "weak"
    normalized.affective_states = _normalize_affective_states(normalized)
    normalized.emotions = _normalize_emotions(normalized)
    normalized.emotion_labels = _emotion_labels_for_compatibility(normalized)
    normalized.state_labels = _state_labels_for_compatibility(normalized)
    if normalized.emotions:
        normalized.emotion_observation = "observed"
    elif normalized.emotion_observation == "observed":
        normalized.emotion_observation = "unclear"
    if normalized.entry_type in _GRAPH_BLOCKED_ENTRY_TYPES:
        normalized.emotion_observation = "unclear"

    normalized.should_graph_mood = _should_graph_metric(normalized, "mood", min_confidence=0.65)
    normalized.should_graph_energy = _should_graph_metric(normalized, "energy", min_confidence=0.70)
    normalized.observation_cadence = (
        ObservationCadence()
        if normalized.entry_type in _GRAPH_BLOCKED_ENTRY_TYPES
        else _normalize_observation_cadence(normalized.observation_cadence)
    )
    if normalized.entry_type in _GRAPH_BLOCKED_ENTRY_TYPES:
        normalized.should_graph_mood = False
        normalized.should_graph_energy = False

    return normalized


def _normalize_observation_cadence(value: ObservationCadence) -> ObservationCadence:
    cadence = value.model_copy(deep=True)
    if cadence.next_checkin_min_minutes is None or cadence.next_checkin_max_minutes is None:
        cadence.next_checkin_min_minutes = None
        cadence.next_checkin_max_minutes = None
        return cadence
    minimum = max(5, min(int(cadence.next_checkin_min_minutes), 360))
    maximum = max(5, min(int(cadence.next_checkin_max_minutes), 360))
    cadence.next_checkin_min_minutes = min(minimum, maximum)
    cadence.next_checkin_max_minutes = max(minimum, maximum)
    return cadence


def _normalize_emotions(features: EntryFeatures) -> list[EmotionSignal]:
    normalized: list[EmotionSignal] = []
    seen: set[tuple[str, str]] = set()
    for emotion in features.emotions:
        label = _compact_text(str(emotion.label or "").lower())
        if not label:
            continue
        if label in _AFFECTIVE_STATE_LABELS:
            continue
        time_scope = emotion.time_scope if emotion.time_scope in {"current", "recent", "past_story", "dream", "mentioned_not_felt", "unclear"} else "unclear"
        level = emotion.intensity_level if emotion.intensity_level in {*_EMOTION_INTENSITY, "unclear"} else "unclear"
        evidence = _compact_text(emotion.evidence or "")[:180] or None
        if time_scope not in {"current", "recent"}:
            if label not in features.mentioned_but_not_felt:
                features.mentioned_but_not_felt.append(label)
            continue
        if not evidence:
            _add_uncertainty(features, f"emotion {label} очищено: немає evidence.")
            continue
        intensity = float(emotion.intensity or 0.0)
        if level in _EMOTION_INTENSITY:
            intensity = _EMOTION_INTENSITY[level]
        else:
            level = _intensity_level_from_number(intensity)
        if emotion.confidence < 0.25:
            _add_uncertainty(features, f"emotion {label} має низьку confidence.")
            continue
        key = (label, time_scope)
        if key in seen:
            continue
        if label not in _EMOTION_LABELS:
            _add_uncertainty(features, f"emotion {label} відхилено: поза контрольованим словником.")
            continue
        normalized.append(
            EmotionSignal(
                label=label[:60],
                intensity_level=level,
                intensity=max(0.0, min(1.0, intensity)),
                confidence=emotion.confidence,
                evidence=evidence,
                time_scope=time_scope,
            )
        )
        seen.add(key)
    return normalized[:8]


def _normalize_affective_states(features: EntryFeatures) -> list[AffectiveStateSignal]:
    normalized: list[AffectiveStateSignal] = []
    seen: set[tuple[str, str]] = set()
    candidates = [
        *((state, True) for state in features.affective_states),
        *((emotion, False) for emotion in features.emotions),
    ]
    for state, explicitly_affective in candidates:
        label = _compact_text(str(state.label or "").lower())
        if not label:
            continue
        if label not in _AFFECTIVE_STATE_LABELS:
            if explicitly_affective:
                _add_uncertainty(features, f"affective state {label} відхилено: поза контрольованим словником.")
            continue
        time_scope = state.time_scope if state.time_scope in {"current", "recent", "past_story", "dream", "mentioned_not_felt", "unclear"} else "unclear"
        if time_scope not in {"current", "recent"}:
            if label not in features.mentioned_but_not_felt:
                features.mentioned_but_not_felt.append(label)
            continue
        evidence = _compact_text(state.evidence or "")[:180] or None
        if not evidence:
            _add_uncertainty(features, f"affective state {label} очищено: немає evidence.")
            continue
        if state.confidence < 0.25:
            _add_uncertainty(features, f"affective state {label} має низьку confidence.")
            continue
        level = state.intensity_level if state.intensity_level in {*_EMOTION_INTENSITY, "unclear"} else "unclear"
        intensity = float(state.intensity or 0.0)
        if level in _EMOTION_INTENSITY:
            intensity = _EMOTION_INTENSITY[level]
        else:
            level = _intensity_level_from_number(intensity)
        key = (label, time_scope)
        if key in seen:
            continue
        normalized.append(
            AffectiveStateSignal(
                label=label,
                intensity_level=level,
                intensity=max(0.0, min(1.0, intensity)),
                confidence=state.confidence,
                evidence=evidence,
                time_scope=time_scope,
            )
        )
        seen.add(key)
    return normalized[:8]


def _emotion_labels_for_compatibility(features: EntryFeatures) -> list[str]:
    labels: list[str] = []
    for emotion in features.emotions:
        if emotion.time_scope in {"current", "recent"} and emotion.label not in labels:
            labels.append(emotion.label)
    if labels:
        return labels[:8]
    for label in features.emotion_labels:
        clean = _compact_text(str(label or "").lower())
        if clean in _EMOTION_LABELS and clean not in labels:
            labels.append(clean[:60])
    return labels[:8]


def _state_labels_for_compatibility(features: EntryFeatures) -> list[str]:
    labels: list[str] = []
    for state in features.affective_states:
        if state.time_scope in {"current", "recent"} and state.label not in labels:
            labels.append(state.label)
    for label in [*features.state_labels, *features.non_emotional_states]:
        clean = _compact_text(str(label or "").lower())
        if clean and clean not in _EMOTION_LABELS and clean not in labels:
            labels.append(clean[:60])
    return labels[:8]


def _intensity_level_from_number(value: float) -> str:
    if value <= 0:
        return "unclear"
    if value < 0.22:
        return "trace"
    if value < 0.43:
        return "mild"
    if value < 0.68:
        return "moderate"
    if value < 0.92:
        return "strong"
    return "overwhelming"


def entry_feature_context(entry: Entry, *, extra_context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = {
        "raw_text": entry.raw_text or "",
        "source": entry.source,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "local_timestamp": entry.local_timestamp.isoformat() if entry.local_timestamp else None,
        "metadata": entry.meta or {},
        "backfill": True,
    }
    if extra_context:
        context.update(extra_context)
    return context


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _normalize_feature_value(feature: FeatureValue) -> FeatureValue:
    value = str(feature.value or "unclear").strip().lower().replace(" ", "_").replace("-", "_")
    if value in _FEATURE_VALUES:
        return FeatureValue(value=value, confidence=feature.confidence)
    number = _parse_number(value)
    if number is not None:
        return FeatureValue(value=_score_to_label(number), confidence=feature.confidence)
    return FeatureValue(value="unclear", confidence=min(feature.confidence, 0.2))


def _parse_number(value: str) -> float | None:
    try:
        number = float(value.replace(",", "."))
    except ValueError:
        return None
    if 0 <= number <= 10:
        return number
    return None


def _score_to_label(score: float) -> str:
    if score <= 1:
        return "very_low"
    if score <= 2.5:
        return "low"
    if score <= 4:
        return "somewhat_low"
    if score < 6:
        return "neutral"
    if score < 7.5:
        return "somewhat_high"
    if score < 9:
        return "high"
    return "very_high"


def _normalize_metric(features: EntryFeatures, metric: str) -> None:
    feature = getattr(features, metric)
    normalized_feature = _normalize_feature_value(feature)
    setattr(features, metric, normalized_feature)
    if normalized_feature.value == "unclear":
        _clear_metric_evidence(features, metric)
        return

    evidence = _metric_evidence(features, metric)
    if not evidence:
        setattr(features, metric, _unclear_feature())
        _clear_metric_evidence(features, metric)
        _add_uncertainty(features, f"{metric} очищено: немає прямого evidence у записі.")
        return

    setattr(features, f"{metric}_evidence", evidence)
    reasoning = _metric_reasoning_type(features, metric)
    setattr(features, f"{metric}_reasoning_type", reasoning)
    if reasoning in {"unclear", "metadata_only"}:
        setattr(features, metric, _unclear_feature())
        _clear_metric_evidence(features, metric)
        _add_uncertainty(features, f"{metric} очищено: reasoning_type={reasoning} не є достатнім.")


def _metric_evidence(features: EntryFeatures, metric: str) -> str | None:
    evidence = getattr(features, f"{metric}_evidence", None)
    if not isinstance(evidence, str):
        return None
    evidence = _compact_text(evidence)
    return evidence[:180] or None


def _metric_reasoning_type(features: EntryFeatures, metric: str) -> str:
    value = getattr(features, f"{metric}_reasoning_type", "unclear")
    allowed = {"direct_text", "weak_text", "context_inferred", "user_manual", "metadata_only", "unclear"}
    return value if value in allowed else "unclear"


def _clear_metric_evidence(features: EntryFeatures, metric: str) -> None:
    setattr(features, f"{metric}_evidence", None)
    setattr(features, f"{metric}_reasoning_type", "unclear")


def _should_graph_metric(features: EntryFeatures, metric: str, *, min_confidence: float) -> bool:
    feature = getattr(features, metric)
    if feature.value == "unclear" or feature.confidence < min_confidence:
        return False
    if features.entry_type in _GRAPH_BLOCKED_ENTRY_TYPES:
        return False
    if not _metric_evidence(features, metric):
        return False
    return _metric_reasoning_type(features, metric) in _GRAPH_REASONING_TYPES


def _unclear_feature() -> FeatureValue:
    return FeatureValue(value="unclear", confidence=0.0)


def _add_uncertainty(features: EntryFeatures, note: str) -> None:
    if note not in features.uncertainty_notes:
        features.uncertainty_notes.append(note)
