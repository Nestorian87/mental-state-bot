from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.ai.schemas import EntryFeatures
from mental_state_bot.ai.service import AIService
from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import Day, Entry, Snapshot, User, UserSettings
from mental_state_bot.emotions import (
    CANONICAL_EMOTIONS,
    EMOTION_INTENSITY_LEVELS,
    EMOTION_INTENSITY_VALUES,
)
from mental_state_bot.services.analysis_backfill import (
    ENTRY_FEATURES_SCHEMA_VERSION,
    ENTRY_FEATURES_TASK,
    analyze_entry_features,
)
from mental_state_bot.services.journal_day import current_journal_date
from mental_state_bot.services.memory_graph import relevant_memory_context_for_text
from mental_state_bot.services.planned_events import planned_event_context
from mental_state_bot.services.preferences import (
    clarification_queue,
    custom_interaction_style,
    life_context_items,
    pending_post_entry_followup,
    settings_json_with_clarification_queue,
    settings_json_with_pending_clarification,
    settings_json_with_pending_post_entry_followup,
    user_profile_context,
)
from mental_state_bot.services.semantic_context import (
    semantic_memory_context,
    verified_semantic_memory_insight,
)
from mental_state_bot.time_utils import local_now, utc_now


@dataclass(frozen=True)
class BotReply:
    text: str
    keyboard: str | None = None
    keyboard_options: tuple[str, ...] = ()


@dataclass(frozen=True)
class InteractionResult:
    replies: list[BotReply]
    entry_id: uuid.UUID | None = None
    snapshot_closed: bool = False
    should_embed_entry: bool = False


class InteractionService:
    def __init__(self, settings: Settings, ai_service: AIService) -> None:
        self.settings = settings
        self.ai = ai_service

    async def handle_text_entry(
        self,
        session: AsyncSession,
        *,
        user: User,
        text: str,
        telegram_message_id: int | None,
        reply_to_message_id: int | None,
        source: str = "message",
    ) -> InteractionResult:
        day = await self._current_day(session, user)
        user_settings = await repo.get_user_settings(session, user.id)
        if pending_post_entry_followup(user_settings) is not None:
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_pending_post_entry_followup(user_settings, None)},
            )
        style_context = _style_context(user_settings)
        open_snapshot = await repo.get_open_snapshot(session, user_id=user.id)

        if open_snapshot is None or source == "manual_confirmed":
            entry = await self._save_and_analyze_entry(
                session,
                user=user,
                day=day,
                snapshot=None,
                text=text,
                telegram_message_id=telegram_message_id,
                reply_to_message_id=reply_to_message_id,
                source="manual",
            )
            immediate_clarification = await self.queue_clarification_for_entry(
                session,
                user=user,
                user_settings=user_settings,
                day=day,
                entry=entry,
                text=text,
                snapshot=None,
                style_context=style_context,
                delivery="immediate",
            )
            day_context = await _day_context(session, day=day)
            graph_context = await _graph_context_for_text(
                session,
                user_id=user.id,
                text=text,
                task_name="manual_entry_memory_graph",
            )
            semantic_memory = await _live_semantic_context(
                self,
                session=session,
                user=user,
                query_text=_live_query_text(text=text, day_context=day_context),
                task_name="micro_summary_semantic_context",
                exclude_entry_ids={entry.id},
            )
            micro_summary, _ = await self.ai.generate_micro_summary(
                session,
                user_id=user.id,
                context={
                    "raw_text": text,
                    "source": "manual",
                    "style": style_context,
                    "day_context": _bounded_day_context(day_context),
                    "relevant_memory_graph": graph_context,
                    "semantic_memory": semantic_memory,
                },
            )
            await self._store_micro_summary(
                session, user=user, entry=entry, micro_summary=micro_summary, semantic_memory=semantic_memory
            )
            return InteractionResult(
                replies=[
                    await post_entry_reply(
                        session,
                        user=user,
                        user_settings=user_settings,
                        entry=entry,
                        micro_summary=micro_summary.text,
                        allow_calibration=immediate_clarification is None,
                        immediate_followup=_clarification_bot_reply(immediate_clarification),
                    )
                ],
                entry_id=entry.id,
                snapshot_closed=True,
                should_embed_entry=True,
            )

        await repo.mark_snapshot_in_progress(session, snapshot_id=open_snapshot.id)
        entry = await self._save_and_analyze_entry(
            session,
            user=user,
            day=day,
            snapshot=open_snapshot,
            text=text,
            telegram_message_id=telegram_message_id,
            reply_to_message_id=reply_to_message_id,
            source="snapshot_response",
        )

        immediate_clarification = await self.queue_clarification_for_entry(
            session,
            user=user,
            user_settings=user_settings,
            day=day,
            entry=entry,
            text=text,
            snapshot=open_snapshot,
            style_context=style_context,
            delivery="immediate",
        )

        snapshot_conversation = await _snapshot_conversation_context(session, snapshot=open_snapshot)
        day_context = await _day_context(session, day=day)
        semantic_memory = await _live_semantic_context(
            self,
            session=session,
            user=user,
            query_text=_live_query_text(
                text=text,
                day_context=day_context,
                snapshot_conversation=snapshot_conversation,
            ),
            task_name="micro_summary_semantic_context",
            exclude_entry_ids={entry.id},
        )
        micro_summary, _ = await self.ai.generate_micro_summary(
            session,
            user_id=user.id,
            context={
                "raw_text": text,
                "snapshot_conversation": _bounded_snapshot_conversation(snapshot_conversation),
                "latest_prompt": snapshot_conversation.get("latest_prompt"),
                "day_context": _bounded_day_context(day_context),
                "semantic_memory": semantic_memory,
                "source": "snapshot_response",
                "style": style_context,
            },
        )
        await self._store_micro_summary(
            session, user=user, entry=entry, micro_summary=micro_summary, semantic_memory=semantic_memory
        )
        if immediate_clarification is not None:
            await repo.close_snapshot(session, snapshot_id=open_snapshot.id, status="clarification")
        else:
            await repo.close_snapshot(session, snapshot_id=open_snapshot.id)
        return InteractionResult(
            replies=[
                await post_entry_reply(
                    session,
                    user=user,
                    user_settings=user_settings,
                    entry=entry,
                    micro_summary=micro_summary.text,
                    allow_calibration=immediate_clarification is None,
                    immediate_followup=_clarification_bot_reply(immediate_clarification),
                )
            ],
            entry_id=entry.id,
            snapshot_closed=True,
            should_embed_entry=True,
        )

    async def queue_clarification_for_entry(
        self,
        session: AsyncSession,
        *,
        user: User,
        user_settings: UserSettings,
        day: Day,
        entry: Entry,
        text: str,
        snapshot: Snapshot | None,
        style_context: dict[str, Any],
        delivery: str = "queued",
    ) -> dict[str, Any] | None:
        should_clarify, clarification_need = await self._clarification_need(
            session,
            snapshot=snapshot,
            text=text,
            entry=entry,
        )
        if not should_clarify:
            return None
        recent_entries = await repo.get_recent_entries(session, user_id=user.id, limit=3)
        snapshot_conversation = (
            await _snapshot_conversation_context(session, snapshot=snapshot) if snapshot is not None else None
        )
        day_context = await _day_context(session, day=day)
        graph_context = await _graph_context_for_text(
            session,
            user_id=user.id,
            text=text,
            task_name="clarification_memory_graph",
        )
        semantic_memory = await _live_semantic_context(
            self,
            session=session,
            user=user,
            query_text=_live_query_text(
                text=text,
                day_context=day_context,
                snapshot_conversation=snapshot_conversation,
            ),
            task_name="clarification_semantic_context",
            exclude_entry_ids={entry.id},
        )
        clarification, model_run_id = await self.ai.generate_clarification(
            session,
            user_id=user.id,
            context={
                "current_answer": text,
                "recent_entries": [_entry_context(item) for item in recent_entries],
                "snapshot": _snapshot_context_hint(snapshot) if snapshot is not None else None,
                "snapshot_conversation": _bounded_snapshot_conversation(snapshot_conversation),
                "latest_prompt": snapshot_conversation.get("latest_prompt") if snapshot_conversation else None,
                "day_context": _bounded_day_context(day_context),
                "relevant_memory_graph": graph_context,
                "semantic_memory": semantic_memory,
                "style": style_context,
                "clarification_need": clarification_need,
            },
        )
        question = " ".join(clarification.question.split())
        queue = clarification_queue(user_settings)
        already_queued = any(
            item.get("entry_id") == str(entry.id) and item.get("status") in {"queued", "active"}
            for item in queue
        )
        recently_similar = _recent_similar_clarification_exists(
            queue,
            question=question,
        )
        if already_queued or recently_similar or not question:
            return None
        item = {
            "id": str(uuid.uuid4()),
            "entry_id": str(entry.id),
            "question": question[:600],
            "options": [" ".join(str(option).split())[:80] for option in clarification.options[:4] if str(option).strip()],
            "reason": clarification_need.get("reason"),
            "model_run_id": str(model_run_id) if model_run_id else None,
            "status": "queued",
            "created_at": utc_now().isoformat(),
        }
        if delivery == "immediate":
            delivered_at = utc_now().isoformat()
            item = {
                **item,
                "status": "active",
                "delivered_at": delivered_at,
                "delivery_source": "post_entry",
            }
        queue.append(item)
        updated = await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_with_clarification_queue(user_settings, queue)},
        )
        if delivery == "immediate":
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_pending_clarification(updated, item)},
            )
        return item

    async def record_button_action(
        self,
        session: AsyncSession,
        *,
        user: User,
        action: str,
    ) -> InteractionResult:
        open_snapshot = await repo.get_open_snapshot(session, user_id=user.id)
        if open_snapshot is None:
            return InteractionResult(
                replies=[BotReply("Цей зріз уже неактуальний. Якщо хочеш, можна почати новий.")],
                snapshot_closed=True,
            )
        snapshot_entries = await repo.list_snapshot_entries(session, snapshot_id=open_snapshot.id)
        target_entry = _latest_content_entry(snapshot_entries)
        await repo.close_snapshot(
            session,
            snapshot_id=open_snapshot.id,
            status="postponed" if action == "later" else "closed_by_user",
        )
        text = {
            "as_is": "Ок, записав як є. На цьому все.",
            "stop": "Ок, зупиняюся. Я зберіг це як частину зрізу.",
            "later": "Ок, повернуся пізніше без спаму.",
        }.get(action, "Записав.")
        return InteractionResult(
            replies=[BotReply(text, keyboard="correction" if target_entry and action in {"as_is", "stop"} else None)],
            entry_id=target_entry.id if target_entry else None,
            snapshot_closed=True,
            should_embed_entry=target_entry is not None,
        )

    async def record_correction(
        self,
        session: AsyncSession,
        *,
        user: User,
        correction_text: str,
        telegram_message_id: int | None,
        reply_to_message_id: int | None,
        target_entry_id: uuid.UUID | None = None,
        clarification_context: dict[str, Any] | None = None,
    ) -> InteractionResult:
        target = await repo.get_entry(session, entry_id=target_entry_id) if target_entry_id else None
        if target is not None and target.user_id != user.id:
            target = None
        if target is None:
            recent_entries = await repo.get_recent_entries(session, user_id=user.id, limit=20)
            target = next(
                (
                    entry
                    for entry in reversed(recent_entries)
                    if entry.source not in {"correction", "profile_context_update"}
                ),
                None,
            )
        if target is None:
            return InteractionResult(
                replies=[BotReply("Не знайшов запис, який можна виправити.")],
                snapshot_closed=True,
            )
        target_note = "останнього запису" if target else "контексту"
        user_settings = await repo.get_user_settings(session, user.id)
        style_context = _style_context(user_settings)
        day = await session.get(Day, target.day_id) if target.day_id else None
        if day is None:
            day = await self._current_day(session, user)
        day_context = await _day_context(session, day=day)
        snapshot = await session.get(Snapshot, target.snapshot_id) if target and target.snapshot_id else None
        snapshot_conversation = (
            await _snapshot_conversation_context(session, snapshot=snapshot)
            if snapshot is not None
            else None
        )
        semantic_memory = await _live_semantic_context(
            self,
            session=session,
            user=user,
            query_text=_live_query_text(
                text=correction_text,
                day_context=day_context,
                snapshot_conversation=snapshot_conversation,
            ),
            task_name="correction_semantic_context",
            exclude_entry_ids={target.id} if target else set(),
        )
        if target is not None:
            await analyze_entry_features(
                session,
                settings=self.settings,
                ai_service=self.ai,
                user_id=user.id,
                entry=target,
                extra_context={
                    "correction_text": correction_text,
                    "clarification_context": clarification_context,
                    "snapshot_conversation": _bounded_snapshot_conversation(snapshot_conversation),
                    "latest_prompt": (
                        snapshot_conversation.get("latest_prompt") if snapshot_conversation else None
                    ),
                    "day_context": _bounded_day_context(day_context, limit=20),
                    "semantic_memory": semantic_memory,
                    "backfill": False,
                },
            )
            await repo.add_ai_analysis(
                session,
                user_id=user.id,
                target_type="entry",
                target_id=target.id,
                task_name="apply_correction",
                schema_version="correction.v1",
                provider=self.settings.ai_provider,
                model=self.settings.ai_live_model,
                result={
                    "correction_text": correction_text,
                    "telegram_message_id": telegram_message_id,
                    "reply_to_message_id": reply_to_message_id,
                    "corrected_at": local_now(user.timezone).isoformat(),
                },
                confidence=None,
                uncertainty_notes=[],
                model_run_id=None,
            )
            if target.day_id is not None:
                await repo.mark_day_summaries_stale(
                    session,
                    user_id=user.id,
                    day_id=target.day_id,
                    reason="entry_corrected",
                )
        micro_summary, _ = await self.ai.generate_micro_summary(
            session,
            user_id=user.id,
            context={
                "raw_text": correction_text,
                "source": "correction",
                "correction_text": correction_text,
                "original_entry": _entry_context(target) if target else None,
                "snapshot_conversation": _bounded_snapshot_conversation(snapshot_conversation),
                "latest_prompt": snapshot_conversation.get("latest_prompt") if snapshot_conversation else None,
                "day_context": _bounded_day_context(day_context),
                "semantic_memory": semantic_memory,
                "style": style_context,
            },
        )
        await self._store_micro_summary(
            session, user=user, entry=target, micro_summary=micro_summary, semantic_memory=semantic_memory
        )
        if clarification_context is not None:
            current_settings = await repo.get_user_settings(session, user.id)
            next_clarification = await self.queue_clarification_for_entry(
                session,
                user=user,
                user_settings=current_settings,
                day=day,
                entry=target,
                text=correction_text,
                snapshot=snapshot,
                style_context=_style_context(current_settings),
                delivery="immediate",
            )
            return InteractionResult(
                replies=[
                    await post_entry_reply(
                        session,
                        user=user,
                        user_settings=current_settings,
                        entry=target,
                        micro_summary=micro_summary.text,
                        allow_calibration=next_clarification is None,
                        immediate_followup=_clarification_bot_reply(next_clarification),
                    )
                ],
                entry_id=target.id,
                snapshot_closed=True,
                should_embed_entry=True,
            )
        return InteractionResult(
            replies=[
                BotReply(f"Оновив трактування для {target_note}."),
                BotReply(micro_summary.text, keyboard=f"correction:{target.id}"),
            ],
            entry_id=target.id,
            snapshot_closed=True,
            should_embed_entry=True,
        )

    async def record_missed_reason(
        self,
        session: AsyncSession,
        *,
        user: User,
        reason_text: str,
        reason_code: str | None = None,
    ) -> InteractionResult:
        missed = await repo.get_latest_open_missed_prompt(session, user_id=user.id)
        if missed is None:
            return InteractionResult(
                replies=[BotReply("Не бачу відкритого пропущеного зрізу. Можна просто записати момент звичайним повідомленням.")]
            )

        snapshot = await session.get(Snapshot, missed.snapshot_id)
        await repo.resolve_missed_prompt(
            session,
            missed_prompt_id=missed.id,
            reason_text=reason_text,
        )
        if snapshot is not None:
            await repo.close_snapshot(session, snapshot_id=snapshot.id, status="missed_explained")

        entry = await repo.add_entry(
            session,
            user_id=user.id,
            day_id=snapshot.day_id if snapshot else None,
            snapshot_id=snapshot.id if snapshot else missed.snapshot_id,
            source="missed_reason",
            raw_text=f"причина пропуску: {reason_text}",
            telegram_message_id=None,
            reply_to_message_id=None,
            local_timestamp=local_now(user.timezone),
            meta={"missed_prompt_id": str(missed.id), "reason_code": reason_code},
        )
        await analyze_entry_features(
            session,
            settings=self.settings,
            ai_service=self.ai,
            user_id=user.id,
            entry=entry,
            extra_context={"missed_prompt": True, "reason_code": reason_code},
        )
        return InteractionResult(
            replies=[BotReply("Записав причину пропуску. На цьому все.")],
            entry_id=entry.id,
            snapshot_closed=True,
            should_embed_entry=True,
        )

    async def _save_and_analyze_entry(
        self,
        session: AsyncSession,
        *,
        user: User,
        day: Day,
        snapshot: Snapshot | None,
        text: str,
        telegram_message_id: int | None,
        reply_to_message_id: int | None,
        source: str,
    ) -> Entry:
        entry = await repo.add_entry(
            session,
            user_id=user.id,
            day_id=day.id,
            snapshot_id=snapshot.id if snapshot else None,
            source=source,
            raw_text=text,
            telegram_message_id=telegram_message_id,
            reply_to_message_id=reply_to_message_id,
            local_timestamp=local_now(user.timezone),
            meta={},
        )
        snapshot_conversation = (
            await _snapshot_conversation_context(session, snapshot=snapshot)
            if snapshot
            else None
        )
        graph_context = await _graph_context_for_text(
            session,
            user_id=user.id,
            text=text,
            task_name="entry_analysis_memory_graph",
        )
        previous_affective_context = await _previous_affective_context(
            session,
            day=day,
            exclude_entry_id=entry.id,
        )
        user_settings = await repo.get_user_settings(session, user.id)
        await analyze_entry_features(
            session,
            settings=self.settings,
            ai_service=self.ai,
            user_id=user.id,
            entry=entry,
            extra_context={
                "snapshot_context": _snapshot_context_hint(snapshot),
                "snapshot_conversation": _bounded_snapshot_conversation(snapshot_conversation),
                "latest_prompt": (
                    snapshot_conversation.get("latest_prompt") if snapshot_conversation else None
                ),
                "day_context": _bounded_day_context(await _day_context(session, day=day), limit=20),
                "relevant_memory_graph": graph_context,
                "previous_affective_context": previous_affective_context,
                "planned_events": planned_event_context(user_settings),
                "backfill": False,
            },
        )
        return entry

    async def _store_micro_summary(
        self,
        session: AsyncSession,
        *,
        user: User,
        entry: Entry,
        micro_summary,
        semantic_memory: list[dict],
    ) -> None:
        result = {"text": micro_summary.text}
        raw_insight = getattr(micro_summary, "semantic_memory_insight", None)
        insight = verified_semantic_memory_insight(
            raw_insight.model_dump() if raw_insight is not None else {}, semantic_memory
        )
        if insight is not None:
            result["semantic_memory_insight"] = insight
        await repo.add_ai_analysis(
            session,
            user_id=user.id,
            target_type="entry",
            target_id=entry.id,
            task_name="generate_micro_summary",
            schema_version="micro_summary.v1",
            provider=self.settings.ai_provider,
            model=self.settings.ai_live_model,
            result=result,
            confidence=None,
            uncertainty_notes=[],
            model_run_id=None,
        )

    async def _clarification_need(
        self,
        session: AsyncSession,
        *,
        snapshot: Snapshot | None,
        text: str,
        entry: Entry,
    ) -> tuple[bool, dict[str, Any]]:
        words = [word for word in text.strip().split() if word]
        if len(words) <= 3:
            return True, {"reason": "very_short_answer", "word_count": len(words)}
        analyses = await repo.list_analyses_for_targets(session, target_type="entry", target_ids=[entry.id])
        feature_analysis = next((item for item in analyses if item.task_name == "extract_entry_features"), None)
        if feature_analysis is None:
            return False, {"reason": "features_missing"}
        features = EntryFeatures.model_validate(feature_analysis.result)
        missing_metrics = _missing_core_metrics(features)
        if features.emotion_needs_clarification:
            return True, {
                "reason": "emotion_transition_unclear",
                "missing_metrics": missing_metrics,
                "suggested_question": features.clarification_question,
                "emotion_transition": features.emotion_transition,
                "entry_type": features.entry_type,
                "data_quality": features.data_quality,
                "feature_confidence": features.confidence,
                "uncertainty_notes": features.uncertainty_notes,
            }
        if features.needs_clarification:
            return True, {
                "reason": (
                    "missing_" + "_and_".join(missing_metrics)
                    if missing_metrics
                    else "model_requested_clarification"
                ),
                "missing_metrics": missing_metrics,
                "suggested_question": features.clarification_question,
                "entry_type": features.entry_type,
                "data_quality": features.data_quality,
                "feature_confidence": features.confidence,
                "uncertainty_notes": features.uncertainty_notes,
            }
        if missing_metrics and _should_offer_missing_metric_clarification(features):
            return True, {
                "reason": "missing_" + "_and_".join(missing_metrics),
                "missing_metrics": missing_metrics,
                "entry_type": features.entry_type,
                "data_quality": features.data_quality,
                "feature_confidence": features.confidence,
                "uncertainty_notes": features.uncertainty_notes,
            }
        if features.data_quality in {"empty", "very_low"} or features.confidence < 0.35:
            return True, {
                "reason": "low_information",
                "data_quality": features.data_quality,
                "feature_confidence": features.confidence,
                "uncertainty_notes": features.uncertainty_notes,
            }
        return False, {
            "reason": "enough_information",
            "data_quality": features.data_quality,
            "feature_confidence": features.confidence,
        }

    async def _current_day(self, session: AsyncSession, user: User) -> Day:
        user_settings = await repo.get_user_settings(session, user.id)
        target_date = await current_journal_date(session, user=user, user_settings=user_settings)
        return await repo.get_or_create_day(
            session,
            user_id=user.id,
            local_date_value=target_date,
            started_at=utc_now(),
        )


def _entry_context(entry: Entry) -> dict[str, Any]:
    return {
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "local_timestamp": entry.local_timestamp.isoformat() if entry.local_timestamp else None,
        "source": entry.source,
        "raw_text": entry.raw_text,
    }


async def _previous_affective_context(
    session: AsyncSession,
    *,
    day: Day,
    exclude_entry_id: uuid.UUID,
    limit: int = 3,
) -> list[dict[str, Any]]:
    entries = [
        entry
        for entry in await repo.list_day_entries(session, day_id=day.id)
        if entry.id != exclude_entry_id and entry.source not in {"correction", "profile_context_update"}
    ]
    if not entries:
        return []
    analyses = await repo.list_analyses_for_targets(
        session,
        target_type="entry",
        target_ids=[entry.id for entry in entries],
    )
    features_by_entry = _latest_feature_results_by_entry(analyses)
    context: list[dict[str, Any]] = []
    for entry in reversed(entries):
        result = features_by_entry.get(str(entry.id)) or {}
        signals = [
            {
                "label": str(item.get("label") or "").strip(),
                "intensity_level": str(item.get("intensity_level") or "unclear").strip(),
                "confidence": item.get("confidence"),
                "evidence": str(item.get("evidence") or "").strip()[:180],
            }
            for item in result.get("emotions") or []
            if isinstance(item, dict)
            and str(item.get("time_scope") or "").strip() in {"current", "recent"}
            and str(item.get("label") or "").strip()
            and str(item.get("evidence") or "").strip()
        ]
        if not signals:
            continue
        context.append(
            {
                "local_timestamp": entry.local_timestamp.isoformat() if entry.local_timestamp else None,
                "emotions": signals[:4],
            }
        )
        if len(context) >= limit:
            break
    return list(reversed(context))


def _latest_feature_results_by_entry(analyses) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for analysis in analyses:
        if analysis.task_name != ENTRY_FEATURES_TASK or not isinstance(analysis.result, dict):
            continue
        results[str(analysis.target_id)] = analysis.result
    return results


def _latest_content_entry(entries) -> Entry | None:
    for entry in reversed(list(entries)):
        if entry.source in {"correction", "profile_context_update"}:
            continue
        if str(entry.source).startswith("button_"):
            continue
        return entry
    return None


def _missing_core_metrics(features: EntryFeatures) -> list[str]:
    missing: list[str] = []
    if _feature_is_unclear(features.mood):
        missing.append("mood")
    if _feature_is_unclear(features.energy):
        missing.append("energy")
    return missing


def _missing_emotion_signal(features: EntryFeatures) -> bool:
    if features.data_quality not in {"partial", "enough", "rich"}:
        return False
    return not bool(features.emotion_labels)


def _should_offer_missing_metric_clarification(features: EntryFeatures) -> bool:
    """Escalate a meaningful graph gap without turning fragments into a survey."""
    return (
        features.entry_type not in {"photo_only", "dream", "reply_fragment", "command_or_system"}
        and features.data_quality in {"partial", "enough", "rich"}
        and features.confidence >= 0.4
    )


def _recent_similar_clarification_exists(
    queue: list[dict[str, Any]],
    *,
    question: str,
) -> bool:
    today = utc_now().date().isoformat()
    for item in queue:
        status = str(item.get("status") or "")
        if status in {"queued", "active"} and _question_similarity(
            str(item.get("question") or ""), question
        ) >= 0.78:
            return True
        if (
            status in {"answered", "skipped"}
            and _item_touched_on_date(item, today)
            and _question_similarity(str(item.get("question") or ""), question) >= 0.78
        ):
            return True
    return False


def _item_touched_on_date(item: dict[str, Any], date_text: str) -> bool:
    for key in ("created_at", "delivered_at", "answered_at", "skipped_at"):
        if str(item.get(key) or "").startswith(date_text):
            return True
    return False


def _question_similarity(left: str, right: str) -> float:
    left_tokens = set(_clarification_tokens(left))
    right_tokens = set(_clarification_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _clarification_tokens(text: str) -> list[str]:
    return [token for token in (" ".join(str(text or "").lower().split())).split() if len(token) >= 4]


def _feature_is_unclear(feature: object) -> bool:
    value = getattr(feature, "value", None)
    return str(value or "").strip().lower() in {"", "unclear", "unknown", "невідомо"}


async def metric_calibration_replies(session: AsyncSession, *, entry: Entry) -> list[BotReply]:
    features = await _latest_entry_features(session, entry_id=entry.id)
    if features is None:
        return []
    missing = _missing_core_metrics(features)
    if missing and (features.needs_clarification or _should_offer_missing_metric_clarification(features)):
        metric = missing[0]
        return [BotReply(_metric_calibration_prompt(metric), keyboard=f"metric_score:{entry.id}:{metric}")]
    if _missing_emotion_signal(features):
        return [
            BotReply(
                "Я не хочу домислювати емоцію з цього запису. Що найближче?",
                keyboard=f"emotion_calibration:{entry.id}",
            )
        ]
    return []


def _clarification_bot_reply(item: dict[str, Any] | None) -> BotReply | None:
    if item is None:
        return None
    item_id = str(item.get("id") or "")
    question = " ".join(str(item.get("question") or "").split())
    if not item_id or not question:
        return None
    options = tuple(
        " ".join(str(option).split())[:80]
        for option in item.get("options") or []
        if str(option).strip()
    )
    return BotReply(
        question,
        keyboard=f"clarification:{item_id}",
        keyboard_options=options,
    )


async def post_entry_reply(
    session: AsyncSession,
    *,
    user: User,
    user_settings: UserSettings,
    entry: Entry,
    micro_summary: str,
    allow_calibration: bool = True,
    immediate_followup: BotReply | None = None,
) -> BotReply:
    """Present a completed entry and one active next step, if it is useful now."""
    parts = [micro_summary]
    interpretation = None
    next_step = immediate_followup
    if next_step is None:
        interpretation = await interpretation_summary_reply(session, entry=entry)
        if interpretation is not None:
            parts.append(interpretation.text)
    if next_step is None and allow_calibration:
        calibration_replies = await metric_calibration_replies(session, entry=entry)
        next_step = calibration_replies[0] if calibration_replies else None
    if next_step is not None:
        parts.append(next_step.text)
        followup_kind = _post_entry_followup_kind(next_step.keyboard)
        if followup_kind is not None:
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={
                    "settings_json": settings_json_with_pending_post_entry_followup(
                        user_settings,
                        {
                            "entry_id": str(entry.id),
                            "kind": followup_kind,
                            "created_at": utc_now().isoformat(),
                        },
                    )
                },
            )
        return BotReply(
            "\n\n".join(part for part in parts if part.strip()),
            keyboard=_post_entry_followup_keyboard(next_step.keyboard, entry=entry),
            keyboard_options=next_step.keyboard_options,
        )

    parts.append("Записав як є. На цьому все.")
    return BotReply(
        "\n\n".join(part for part in parts if part.strip()),
        keyboard=interpretation.keyboard if interpretation is not None else f"correction:{entry.id}",
    )


def _post_entry_followup_keyboard(kind: str | None, *, entry: Entry) -> str:
    if kind and kind.startswith("metric_score:"):
        return "metric_score_with_correction:" + kind.split(":", maxsplit=1)[1]
    if kind and kind.startswith("emotion_calibration:"):
        return "emotion_calibration_with_correction:" + str(entry.id)
    return kind or f"correction:{entry.id}"


def _post_entry_followup_kind(kind: str | None) -> str | None:
    if kind and kind.startswith("metric_score:"):
        return "metric"
    if kind and kind.startswith("emotion_calibration:"):
        return "emotion"
    return None


async def interpretation_summary_reply(session: AsyncSession, *, entry: Entry) -> BotReply | None:
    features = await _latest_entry_features(session, entry_id=entry.id)
    if features is None:
        return None
    mood = _feature_interpretation("настрій", features.mood)
    energy = _feature_interpretation("енергія", features.energy)
    emotions = _emotion_interpretation(features)
    confidence = _confidence_interpretation(features)
    return BotReply(
        f"Як я це розмітив: {mood}; {energy}; {emotions}. {confidence}",
        keyboard=f"interpretation:{entry.id}",
    )


async def apply_metric_calibration(
    session: AsyncSession,
    *,
    settings: Settings,
    user: User,
    entry: Entry,
    metric: str,
    score: int,
) -> list[BotReply]:
    if metric not in {"mood", "energy"}:
        return [BotReply("Не впізнав метрику для уточнення.")]
    features = await _latest_entry_features(session, entry_id=entry.id)
    if features is None:
        return [BotReply("Не знайшов аналіз цього запису для уточнення.")]
    updated = features.model_dump()
    updated[metric] = {"value": str(score), "confidence": 1.0, "source": "user_calibration"}
    updated[f"{metric}_evidence"] = "ручне уточнення користувача"
    updated[f"{metric}_reasoning_type"] = "user_manual"
    updated[f"should_graph_{metric}"] = True
    notes = [*features.uncertainty_notes, f"user_calibrated_{metric}"]
    updated["uncertainty_notes"] = notes
    updated["confidence"] = max(float(updated.get("confidence") or 0), 0.75)
    await _store_user_calibrated_features(
        session,
        settings=settings,
        user=user,
        entry=entry,
        result=updated,
        uncertainty_notes=notes,
    )
    remaining = _missing_core_metrics(EntryFeatures.model_validate(updated))
    if remaining:
        next_metric = remaining[0]
        return [
            BotReply(
                f"Записав. Ще одне маленьке уточнення: {_metric_calibration_prompt(next_metric)}",
                keyboard=f"metric_score:{entry.id}:{next_metric}",
            )
        ]
    calibrated = EntryFeatures.model_validate(updated)
    if _missing_emotion_signal(calibrated):
        return [
            BotReply(
                "Записав уточнення метрики. Якщо хочеш, ще можна позначити емоцію цього моменту.",
                keyboard=f"emotion_calibration:{entry.id}",
            )
        ]
    return [BotReply("Записав уточнення метрики. Це піде в графіки й підсумки.")]


async def apply_emotion_calibration(
    session: AsyncSession,
    *,
    settings: Settings,
    user: User,
    entry: Entry,
    emotions: list[str],
    intensity_level: str = "moderate",
    emotion_intensity_levels: Mapping[str, str] | None = None,
    time_scope: str = "current",
) -> BotReply:
    clean_emotions = _clean_emotion_labels(emotions)
    if not clean_emotions:
        return BotReply("Ок, не уточнюю емоцію.")
    clean_time_scope = time_scope if time_scope in {"current", "mentioned_not_felt"} else "current"
    clean_intensity_level = _clean_emotion_intensity_level(intensity_level)
    per_emotion_levels = {
        label: _clean_emotion_intensity_level((emotion_intensity_levels or {}).get(label, clean_intensity_level))
        for label in clean_emotions
    }
    features = await _latest_entry_features(session, entry_id=entry.id)
    if features is None:
        return BotReply("Не знайшов аналіз цього запису для уточнення емоції.")
    updated = features.model_dump()
    existing_signals = [
        item
        for item in updated.get("emotions", [])
        if isinstance(item, dict) and str(item.get("label") or "").strip() not in clean_emotions
    ]
    manual_signals = [
        {
            "label": label,
            "intensity_level": per_emotion_levels[label],
            "intensity": _emotion_intensity_value(per_emotion_levels[label]),
            "confidence": 1.0,
            "evidence": "ручне уточнення користувача",
            "time_scope": clean_time_scope,
        }
        for label in clean_emotions
    ]
    emotion_labels = [
        str(label).strip()
        for label in updated.get("emotion_labels", [])
        if str(label).strip() in CANONICAL_EMOTIONS and str(label).strip() not in clean_emotions
    ]
    state_labels = [
        str(label).strip()
        for label in updated.get("state_labels", [])
        if str(label).strip() and str(label).strip() not in clean_emotions
    ]
    updated["emotions"] = [*manual_signals, *existing_signals][:8]
    if clean_time_scope == "current":
        updated["emotion_labels"] = [*clean_emotions, *emotion_labels][:8]
        updated["state_labels"] = state_labels[:8]
    else:
        updated["emotion_labels"] = emotion_labels[:8]
        updated["state_labels"] = state_labels[:8]
        mentioned = [
            str(label).strip()
            for label in updated.get("mentioned_but_not_felt", [])
            if str(label).strip() and str(label).strip() not in clean_emotions
        ]
        updated["mentioned_but_not_felt"] = [*clean_emotions, *mentioned][:8]
    notes = [*features.uncertainty_notes, "user_calibrated_emotion"]
    updated["uncertainty_notes"] = notes
    updated["confidence"] = max(float(updated.get("confidence") or 0), 0.75)
    await _store_user_calibrated_features(
        session,
        settings=settings,
        user=user,
        entry=entry,
        result=updated,
        uncertainty_notes=notes,
    )
    if clean_time_scope == "mentioned_not_felt":
        return BotReply(
            f"Позначив: {', '.join(clean_emotions)} тут лише згадувалися або описувалися, а не були поточними емоціями."
        )
    intensity_text = "; ".join(
        f"{label} — {_manual_emotion_intensity_reply(per_emotion_levels[label]).removeprefix(' — ')}"
        for label in clean_emotions
    )
    return BotReply(f"Записав емоції для цього моменту: {intensity_text}.")


def _clean_emotion_labels(emotions: list[str]) -> list[str]:
    cleaned: list[str] = []
    for emotion in emotions:
        label = " ".join(str(emotion or "").strip().lower().split())
        if not label or label in cleaned:
            continue
        if label not in CANONICAL_EMOTIONS:
            continue
        cleaned.append(label[:60])
    return cleaned[:8]


def _clean_emotion_intensity_level(value: str) -> str:
    level = str(value or "").strip().lower()
    if level in EMOTION_INTENSITY_LEVELS:
        return level
    return "unclear"


def _emotion_intensity_value(level: str) -> float:
    return EMOTION_INTENSITY_VALUES.get(level, 0.0)


def _manual_emotion_intensity_reply(level: str) -> str:
    return {
        "trace": " — ледь фоном",
        "mild": " — слабко",
        "moderate": " — помірно",
        "strong": " — сильно",
        "overwhelming": " — дуже сильно",
        "unclear": " без уточнення сили",
    }.get(level, " без уточнення сили")


async def _latest_entry_features(session: AsyncSession, *, entry_id: uuid.UUID) -> EntryFeatures | None:
    analyses = await repo.list_analyses_for_targets(session, target_type="entry", target_ids=[entry_id])
    for analysis in reversed(list(analyses)):
        if analysis.task_name == ENTRY_FEATURES_TASK:
            return EntryFeatures.model_validate(analysis.result)
    return None


async def _store_user_calibrated_features(
    session: AsyncSession,
    *,
    settings: Settings,
    user: User,
    entry: Entry,
    result: dict[str, Any],
    uncertainty_notes: list[str],
) -> None:
    await repo.add_ai_analysis(
        session,
        user_id=user.id,
        target_type="entry",
        target_id=entry.id,
        task_name=ENTRY_FEATURES_TASK,
        schema_version=ENTRY_FEATURES_SCHEMA_VERSION,
        provider="user",
        model="manual_metric_calibration",
        result=result,
        confidence=Decimal("1.0"),
        uncertainty_notes=uncertainty_notes,
        model_run_id=None,
    )
    if entry.day_id is not None:
        await repo.mark_day_summaries_stale(
            session,
            user_id=user.id,
            day_id=entry.day_id,
            reason="metric_calibrated",
        )


def _metric_calibration_prompt(metric: str) -> str:
    if metric == "energy":
        return "скільки зараз енергії приблизно від 0 до 10?"
    return "як би ти оцінив настрій приблизно від 0 до 10?"


def _feature_interpretation(label: str, feature: object) -> str:
    value = str(getattr(feature, "value", "unclear") or "unclear").strip().lower()
    confidence = float(getattr(feature, "confidence", 0.0) or 0.0)
    prefix = "приблизно " if 0 < confidence < 0.55 and value not in {"unclear", "unknown"} else ""
    if value.isdigit():
        return f"{label} {prefix}{value}/10"
    return f"{label} {prefix}{_feature_value_text(value)}"


def _emotion_interpretation(features: EntryFeatures) -> str:
    labels = []
    for emotion in features.emotions:
        if emotion.time_scope not in {"current", "recent"}:
            continue
        label = str(emotion.label).strip()
        if not label:
            continue
        level = _emotion_intensity_text(emotion.intensity_level)
        labels.append(f"{label} {level}" if level else label)
    if not labels:
        labels = [str(label).strip() for label in features.emotion_labels if str(label).strip()]
    if not labels:
        return "емоції неясні"
    return "емоції: " + ", ".join(labels[:4])


def _emotion_intensity_text(level: str) -> str:
    labels = {
        "trace": "ледь фоном",
        "mild": "слабко",
        "moderate": "помірно",
        "strong": "сильно",
        "overwhelming": "дуже сильно",
    }
    return labels.get(level, "")


def _confidence_interpretation(features: EntryFeatures) -> str:
    if features.confidence >= 0.75:
        return "Виглядає доволі впевнено."
    if features.confidence >= 0.45:
        return "Є трохи невпевненості."
    return "Даних замало, це радше чернетка."


def _feature_value_text(value: str) -> str:
    labels = {
        "very_low": "дуже низько",
        "low": "низько",
        "somewhat_low": "трохи низько",
        "neutral": "нейтрально",
        "mixed": "змішано",
        "medium": "середньо",
        "moderate": "помірно",
        "somewhat_high": "трохи високо",
        "high": "високо",
        "very_high": "дуже високо",
        "unclear": "неясний",
        "unknown": "невідомий",
    }
    return labels.get(value, value.replace("_", " ") or "неясний")


async def _day_context(session: AsyncSession, *, day: Day, limit: int = 80) -> dict[str, Any]:
    entries = await repo.list_day_entries(session, day_id=day.id)
    visible_entries = list(entries)[-limit:]
    return {
        "entry_count": len(entries),
        "omitted_entry_count": max(0, len(entries) - len(visible_entries)),
        "entries": [_entry_context(entry) for entry in visible_entries],
    }


def _bounded_day_context(context: dict[str, Any], limit: int = 12) -> dict[str, Any]:
    entries = list(context.get("entries") or [])
    return {
        "entry_count": context.get("entry_count", len(entries)),
        "omitted_entry_count": max(0, int(context.get("entry_count", len(entries))) - min(len(entries), limit)),
        "entries": entries[-limit:],
    }


def _bounded_snapshot_conversation(context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not context:
        return None
    transcript = list(context.get("transcript") or [])
    return {
        "snapshot_id": context.get("snapshot_id"),
        "latest_prompt": context.get("latest_prompt"),
        "transcript": transcript[-8:],
    }


def _snapshot_context_hint(snapshot: Snapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    context = snapshot.context_json or {}
    return {
        "snapshot_id": str(snapshot.id),
        "intent": snapshot.intent,
        "day_phase": context.get("day_phase"),
        "current_local_datetime": context.get("current_local_datetime"),
        "current_local_time": context.get("current_local_time"),
        "daily_rhythm": context.get("daily_rhythm"),
    }


async def _snapshot_conversation_context(
    session: AsyncSession, *, snapshot: Snapshot
) -> dict[str, Any]:
    prompts = await repo.get_snapshot_prompts(session, snapshot_id=snapshot.id)
    entries = await repo.list_snapshot_entries(session, snapshot_id=snapshot.id)
    prompt_context = [
        {
            "role": "bot",
            "kind": prompt.prompt_kind,
            "text": prompt.text,
            "sent_at": prompt.sent_at.isoformat() if prompt.sent_at else None,
        }
        for prompt in prompts
    ]
    entry_context = [
        {
            "role": "user",
            **_entry_context(entry),
        }
        for entry in entries
    ]
    transcript = sorted(
        [*prompt_context, *entry_context],
        key=lambda item: item.get("sent_at") or item.get("local_timestamp") or item.get("created_at") or "",
    )
    return {
        "snapshot_id": str(snapshot.id),
        "initial_context": snapshot.context_json,
        "latest_prompt": prompt_context[-1]["text"] if prompt_context else None,
        "transcript": transcript,
        "prompts": prompt_context,
        "entries": entry_context,
    }

async def _live_semantic_context(
    service: InteractionService,
    *,
    session: AsyncSession,
    user: User,
    query_text: str,
    task_name: str,
    exclude_entry_ids: set[uuid.UUID] | set[str],
    limit: int = 3,
) -> list[dict]:
    return await semantic_memory_context(
        session,
        settings=service.settings,
        ai_service=service.ai,
        user=user,
        query_text=query_text,
        task_name=task_name,
        limit=limit,
        exclude_entry_ids=exclude_entry_ids,
    )


async def _graph_context_for_text(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    text: str,
    task_name: str,
) -> dict[str, Any]:
    """Fetch a small local graph slice without adding another model call."""
    try:
        return await relevant_memory_context_for_text(
            session,
            user_id=user_id,
            text=text,
            limit=8,
            task_name=task_name,
        )
    except Exception:
        return {"nodes": [], "edges": [], "matched": []}


def _live_query_text(
    *,
    text: str,
    day_context: dict[str, Any],
    snapshot_conversation: dict[str, Any] | None = None,
) -> str:
    day_tail = " ".join(
        str(entry.get("raw_text") or "") for entry in (day_context.get("entries") or [])[-8:]
    )
    transcript_tail = ""
    if snapshot_conversation:
        transcript_tail = " ".join(
            str(turn.get("text") or turn.get("raw_text") or "")
            for turn in (snapshot_conversation.get("transcript") or [])[-8:]
        )
    return " ".join(part for part in [day_tail, transcript_tail, text] if part).strip()


def _style_context(settings: UserSettings) -> dict[str, Any]:
    return {
        "tone": settings.tone,
        "humanity_level": settings.humanity_level,
        "custom_interaction_style": custom_interaction_style(settings),
        "user_profile_context": user_profile_context(settings),
        "life_context": life_context_items(settings)[-20:],
    }
