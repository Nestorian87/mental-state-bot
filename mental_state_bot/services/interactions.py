from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.ai.schemas import EntryFeatures
from mental_state_bot.ai.service import AIService
from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import Day, Entry, Snapshot, User, UserSettings
from mental_state_bot.services.analysis_backfill import analyze_entry_features
from mental_state_bot.services.preferences import custom_interaction_style, user_profile_context
from mental_state_bot.services.semantic_context import semantic_memory_context
from mental_state_bot.time_utils import local_date, local_now, utc_now


@dataclass(frozen=True)
class BotReply:
    text: str
    keyboard: str | None = None


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
            day_context = await _day_context(session, day=day)
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
                    "day_context": day_context,
                    "semantic_memory": semantic_memory,
                },
            )
            await self._store_micro_summary(session, user=user, entry=entry, micro_summary=micro_summary.text)
            return InteractionResult(
                replies=[BotReply(micro_summary.text, keyboard=f"correction:{entry.id}")],
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

        should_clarify, clarification_need = await self._clarification_need(
            session,
            snapshot=open_snapshot,
            text=text,
            entry=entry,
        )
        if should_clarify:
            recent_entries = await repo.get_recent_entries(session, user_id=user.id, limit=5)
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
                task_name="clarification_semantic_context",
                exclude_entry_ids={entry.id},
            )
            clarification, model_run_id = await self.ai.generate_clarification(
                session,
                user_id=user.id,
                context={
                    "current_answer": text,
                    "recent_entries": [_entry_context(item) for item in recent_entries],
                    "snapshot": open_snapshot.context_json,
                    "snapshot_conversation": snapshot_conversation,
                    "latest_prompt": snapshot_conversation.get("latest_prompt"),
                    "day_context": day_context,
                    "semantic_memory": semantic_memory,
                    "style": style_context,
                    "clarification_need": clarification_need,
                },
            )
            await repo.increment_clarification_count(session, snapshot_id=open_snapshot.id)
            await repo.add_prompt(
                session,
                snapshot_id=open_snapshot.id,
                prompt_kind="clarification",
                text=clarification.question,
                sent_at=utc_now(),
                telegram_message_id=None,
                model_run_id=model_run_id,
            )
            return InteractionResult(
                replies=[BotReply(clarification.question, keyboard="snapshot_control")],
                entry_id=entry.id,
                snapshot_closed=False,
                should_embed_entry=False,
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
                "snapshot_context": open_snapshot.context_json,
                "snapshot_conversation": snapshot_conversation,
                "latest_prompt": snapshot_conversation.get("latest_prompt"),
                "day_context": day_context,
                "semantic_memory": semantic_memory,
                "source": "snapshot_response",
                "style": style_context,
            },
        )
        await self._store_micro_summary(session, user=user, entry=entry, micro_summary=micro_summary.text)
        await repo.close_snapshot(session, snapshot_id=open_snapshot.id)
        return InteractionResult(
            replies=[BotReply(micro_summary.text, keyboard=f"correction:{entry.id}")],
            entry_id=entry.id,
            snapshot_closed=True,
            should_embed_entry=True,
        )

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
                    "snapshot_context": snapshot.context_json if snapshot else None,
                    "snapshot_conversation": snapshot_conversation,
                    "latest_prompt": (
                        snapshot_conversation.get("latest_prompt") if snapshot_conversation else None
                    ),
                    "day_context": day_context,
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
                "snapshot_context": snapshot.context_json if snapshot else None,
                "snapshot_conversation": snapshot_conversation,
                "latest_prompt": snapshot_conversation.get("latest_prompt") if snapshot_conversation else None,
                "day_context": day_context,
                "semantic_memory": semantic_memory,
                "style": style_context,
            },
        )
        await self._store_micro_summary(session, user=user, entry=target, micro_summary=micro_summary.text)
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
        await analyze_entry_features(
            session,
            settings=self.settings,
            ai_service=self.ai,
            user_id=user.id,
            entry=entry,
            extra_context={
                "snapshot_context": snapshot.context_json if snapshot else None,
                "snapshot_conversation": snapshot_conversation,
                "latest_prompt": (
                    snapshot_conversation.get("latest_prompt") if snapshot_conversation else None
                ),
                "day_context": await _day_context(session, day=day),
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
        micro_summary: str,
    ) -> None:
        await repo.add_ai_analysis(
            session,
            user_id=user.id,
            target_type="entry",
            target_id=entry.id,
            task_name="generate_micro_summary",
            schema_version="micro_summary.v1",
            provider=self.settings.ai_provider,
            model=self.settings.ai_live_model,
            result={"text": micro_summary},
            confidence=None,
            uncertainty_notes=[],
            model_run_id=None,
        )

    async def _clarification_need(
        self,
        session: AsyncSession,
        *,
        snapshot: Snapshot,
        text: str,
        entry: Entry,
    ) -> tuple[bool, dict[str, Any]]:
        if snapshot.clarification_count >= self.settings.max_clarifications_per_snapshot:
            return False, {"reason": "clarification_limit_reached"}
        words = [word for word in text.strip().split() if word]
        if len(words) <= 3:
            return True, {"reason": "very_short_answer", "word_count": len(words)}
        analyses = await repo.list_analyses_for_targets(session, target_type="entry", target_ids=[entry.id])
        feature_analysis = next((item for item in analyses if item.task_name == "extract_entry_features"), None)
        if feature_analysis is None:
            return False, {"reason": "features_missing"}
        features = EntryFeatures.model_validate(feature_analysis.result)
        missing_metrics = _missing_core_metrics(features)
        if missing_metrics:
            return True, {
                "reason": "missing_" + "_and_".join(missing_metrics),
                "missing_metrics": missing_metrics,
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
        return await repo.get_or_create_day(
            session,
            user_id=user.id,
            local_date_value=local_date(user.timezone),
            started_at=utc_now(),
        )


def _entry_context(entry: Entry) -> dict[str, Any]:
    return {
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "local_timestamp": entry.local_timestamp.isoformat() if entry.local_timestamp else None,
        "source": entry.source,
        "raw_text": entry.raw_text,
    }


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


def _feature_is_unclear(feature: object) -> bool:
    value = getattr(feature, "value", None)
    return str(value or "").strip().lower() in {"", "unclear", "unknown", "невідомо"}


async def _day_context(session: AsyncSession, *, day: Day, limit: int = 80) -> dict[str, Any]:
    entries = await repo.list_day_entries(session, day_id=day.id)
    visible_entries = list(entries)[-limit:]
    return {
        "entry_count": len(entries),
        "omitted_entry_count": max(0, len(entries) - len(visible_entries)),
        "entries": [_entry_context(entry) for entry in visible_entries],
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
) -> list[dict]:
    return await semantic_memory_context(
        session,
        settings=service.settings,
        ai_service=service.ai,
        user=user,
        query_text=query_text,
        task_name=task_name,
        limit=6,
        exclude_entry_ids=exclude_entry_ids,
    )


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


def _style_context(settings: UserSettings) -> dict[str, str | None]:
    return {
        "tone": settings.tone,
        "humanity_level": settings.humanity_level,
        "custom_interaction_style": custom_interaction_style(settings),
        "user_profile_context": user_profile_context(settings),
    }
