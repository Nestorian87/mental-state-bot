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


STOP_PHRASES = (
    "не хочу",
    "досить",
    "не зараз",
    "записати як є",
    "пізніше",
    "потім",
)

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
        wants_stop = _contains_stop_phrase(text)

        if open_snapshot is None:
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
            micro_summary, _ = await self.ai.generate_micro_summary(
                session,
                user_id=user.id,
                context={"raw_text": text, "source": "manual", "style": style_context},
            )
            await self._store_micro_summary(session, user=user, entry=entry, micro_summary=micro_summary.text)
            return InteractionResult(
                replies=[BotReply(micro_summary.text, keyboard="correction")],
                entry_id=entry.id,
                snapshot_closed=True,
                should_embed_entry=True,
            )

        await repo.mark_snapshot_in_progress(session, snapshot_id=open_snapshot.id)
        if wants_stop:
            entry = await repo.add_entry(
                session,
                user_id=user.id,
                day_id=day.id,
                snapshot_id=open_snapshot.id,
                source="user_stop",
                raw_text=text,
                telegram_message_id=telegram_message_id,
                reply_to_message_id=reply_to_message_id,
                local_timestamp=local_now(user.timezone),
                meta={"stop_phrase": True},
            )
            await repo.close_snapshot(session, snapshot_id=open_snapshot.id, status="closed_by_user")
            return InteractionResult(
                replies=[BotReply("Я почув, що зараз краще зупинитися. Записав як є, на цьому все.")],
                entry_id=entry.id,
                snapshot_closed=True,
                should_embed_entry=True,
            )

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

        if await self._should_clarify(session, snapshot=open_snapshot, text=text, entry=entry):
            recent_entries = await repo.get_recent_entries(session, user_id=user.id, limit=5)
            clarification, model_run_id = await self.ai.generate_clarification(
                session,
                user_id=user.id,
                context={
                    "current_answer": text,
                    "recent_entries": [_entry_context(item) for item in recent_entries],
                    "snapshot": open_snapshot.context_json,
                    "style": style_context,
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

        micro_summary, _ = await self.ai.generate_micro_summary(
            session,
            user_id=user.id,
            context={
                "raw_text": text,
                "snapshot_context": open_snapshot.context_json,
                "source": "snapshot_response",
                "style": style_context,
            },
        )
        await self._store_micro_summary(session, user=user, entry=entry, micro_summary=micro_summary.text)
        await repo.close_snapshot(session, snapshot_id=open_snapshot.id)
        return InteractionResult(
            replies=[BotReply(micro_summary.text, keyboard="correction")],
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
        day = await self._current_day(session, user)
        open_snapshot = await repo.get_open_snapshot(session, user_id=user.id)
        if open_snapshot is None:
            return InteractionResult(
                replies=[BotReply("Цей зріз уже неактуальний. Якщо хочеш, можна почати новий.")],
                snapshot_closed=True,
            )
        raw_text = {
            "as_is": "записати як є",
            "stop": "не хочу більше зараз говорити",
            "later": "пізніше",
        }.get(action, action)
        entry = await repo.add_entry(
            session,
            user_id=user.id,
            day_id=day.id,
            snapshot_id=open_snapshot.id,
            source=f"button_{action}",
            raw_text=raw_text,
            telegram_message_id=None,
            reply_to_message_id=None,
            local_timestamp=local_now(user.timezone),
            meta={"button_action": action},
        )
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
            replies=[BotReply(text)],
            entry_id=entry.id,
            snapshot_closed=True,
            should_embed_entry=True,
        )

    async def record_correction(
        self,
        session: AsyncSession,
        *,
        user: User,
        correction_text: str,
        telegram_message_id: int | None,
        reply_to_message_id: int | None,
    ) -> InteractionResult:
        day = await self._current_day(session, user)
        recent_entries = await repo.get_recent_entries(session, user_id=user.id, limit=20)
        target = next(
            (
                entry
                for entry in reversed(recent_entries)
                if entry.source not in {"correction", "profile_context_update"}
            ),
            None,
        )
        entry = await repo.add_entry(
            session,
            user_id=user.id,
            day_id=day.id,
            snapshot_id=target.snapshot_id if target else None,
            source="correction",
            raw_text=f"виправлення: {correction_text}",
            telegram_message_id=telegram_message_id,
            reply_to_message_id=reply_to_message_id,
            local_timestamp=local_now(user.timezone),
            meta={
                "correction_for_entry_id": str(target.id) if target else None,
                "correction_for_source": target.source if target else None,
            },
        )
        target_note = "останнього запису" if target else "контексту"
        return InteractionResult(
            replies=[BotReply(f"Записав виправлення для {target_note}. Я враховуватиму це далі.")],
            entry_id=entry.id,
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
        await analyze_entry_features(
            session,
            settings=self.settings,
            ai_service=self.ai,
            user_id=user.id,
            entry=entry,
            extra_context={"snapshot_context": snapshot.context_json if snapshot else None, "backfill": False},
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

    async def _should_clarify(
        self,
        session: AsyncSession,
        *,
        snapshot: Snapshot,
        text: str,
        entry: Entry,
    ) -> bool:
        if snapshot.clarification_count >= self.settings.max_clarifications_per_snapshot:
            return False
        if _contains_stop_phrase(text):
            return False
        words = [word for word in text.strip().split() if word]
        if len(words) <= 3:
            return True
        analyses = await repo.list_analyses_for_targets(session, target_type="entry", target_ids=[entry.id])
        feature_analysis = next((item for item in analyses if item.task_name == "extract_entry_features"), None)
        if feature_analysis is None:
            return False
        features = EntryFeatures.model_validate(feature_analysis.result)
        return features.data_quality in {"empty", "very_low"} or features.confidence < 0.35

    async def _current_day(self, session: AsyncSession, user: User) -> Day:
        return await repo.get_or_create_day(
            session,
            user_id=user.id,
            local_date_value=local_date(user.timezone),
            started_at=utc_now(),
        )


def _contains_stop_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in STOP_PHRASES)


def _entry_context(entry: Entry) -> dict[str, Any]:
    return {
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "source": entry.source,
        "raw_text": entry.raw_text,
    }


def _style_context(settings: UserSettings) -> dict[str, str | None]:
    return {
        "tone": settings.tone,
        "humanity_level": settings.humanity_level,
        "custom_interaction_style": custom_interaction_style(settings),
        "user_profile_context": user_profile_context(settings),
    }
