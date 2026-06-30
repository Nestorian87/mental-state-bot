from __future__ import annotations

import random
from datetime import datetime
from typing import Any

from aiogram import Bot
from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.ai.service import AIService
from mental_state_bot.bot.keyboards import missed_prompt_keyboard, snapshot_initial_keyboard
from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import Day, Snapshot, User, UserSettings
from mental_state_bot.services.preferences import (
    custom_interaction_style,
    snapshots_paused,
    user_profile_context,
)
from mental_state_bot.services.semantic_context import semantic_memory_context
from mental_state_bot.time_utils import local_date, parse_hhmm, utc_now, zoneinfo


async def maybe_send_scheduled_snapshot(
    session: AsyncSession,
    *,
    bot: Bot,
    settings: Settings,
    ai_service: AIService,
    user: User,
) -> bool:
    user_settings = await repo.get_user_settings(session, user.id)
    now = utc_now()
    if not _is_active_time(now, user.timezone, user_settings):
        return False

    today = await repo.get_day_by_date(
        session,
        user_id=user.id,
        local_date_value=local_date(user.timezone),
    )
    if today and today.ended_at:
        open_snapshot = await repo.get_open_snapshot(session, user_id=user.id)
        if open_snapshot is not None:
            await _close_if_snapshot_day_is_closed(session, open_snapshot)
        return False

    open_snapshot = await repo.get_open_snapshot(session, user_id=user.id)
    if open_snapshot is not None:
        if await _close_if_snapshot_day_is_closed(session, open_snapshot):
            return False
        return await _handle_open_snapshot(
            session,
            bot=bot,
            user=user,
            snapshot_id=open_snapshot.id,
            prompted_at=open_snapshot.prompted_at,
            user_settings=user_settings,
        )

    last_snapshot = await repo.get_last_snapshot(session, user_id=user.id)
    if last_snapshot and last_snapshot.prompted_at:
        minutes_since = (now - last_snapshot.prompted_at).total_seconds() / 60
        threshold = random.randint(user_settings.min_interval_minutes, user_settings.max_interval_minutes)
        if minutes_since < threshold:
            return False

    return await send_snapshot_prompt(
        session,
        bot=bot,
        ai_service=ai_service,
        user=user,
        user_settings=user_settings,
        intent="state_and_activity",
        scheduled_for=now,
        photo_prompt_chance=settings.photo_prompt_chance,
    )


async def send_snapshot_prompt(
    session: AsyncSession,
    *,
    bot: Bot,
    ai_service: AIService,
    user: User,
    user_settings: UserSettings,
    intent: str,
    scheduled_for: datetime | None,
    photo_prompt_chance: float = 0.18,
) -> bool:
    now = utc_now()
    open_snapshot = await repo.get_open_snapshot(session, user_id=user.id)
    if open_snapshot is not None and await _close_if_snapshot_day_is_closed(session, open_snapshot):
        open_snapshot = None

    if open_snapshot is not None:
        return await _handle_open_snapshot(
            session,
            bot=bot,
            user=user,
            snapshot_id=open_snapshot.id,
            prompted_at=open_snapshot.prompted_at,
            user_settings=user_settings,
        )

    day = await repo.get_or_create_day(
        session,
        user_id=user.id,
        local_date_value=local_date(user.timezone),
        started_at=now,
    )
    recent_entries = await repo.get_recent_entries(session, user_id=user.id, limit=6)
    day_entries = await repo.list_day_entries(session, day_id=day.id)
    context = snapshot_question_context(
        recent_entries=recent_entries,
        day_entries=day_entries,
        user_settings=user_settings,
        trigger="manual" if scheduled_for is None else "scheduled",
        photo_prompt_opportunity=random.random() < photo_prompt_chance,
    )
    ai_settings = getattr(ai_service, "settings", None)
    context["semantic_memory"] = (
        await semantic_memory_context(
            session,
            settings=ai_settings,
            ai_service=ai_service,
            user=user,
            query_text=_question_query_text(context),
            task_name="snapshot_question_semantic_context",
            limit=6,
            exclude_entry_ids={str(entry.id) for entry in day_entries},
        )
        if ai_settings is not None
        else []
    )
    snapshot = await repo.create_snapshot(
        session,
        user_id=user.id,
        day_id=day.id,
        intent=intent,
        scheduled_for=scheduled_for,
        context=context,
    )
    question, model_run_id = await ai_service.generate_snapshot_question(
        session,
        user_id=user.id,
        context=snapshot.context_json,
    )
    message = await bot.send_message(
        chat_id=user.chat_id,
        text=question.question,
        reply_markup=snapshot_initial_keyboard(),
    )
    await repo.mark_snapshot_prompted(session, snapshot_id=snapshot.id, prompted_at=now)
    await repo.add_prompt(
        session,
        snapshot_id=snapshot.id,
        prompt_kind="initial",
        text=question.question,
        sent_at=now,
        telegram_message_id=message.message_id,
        model_run_id=model_run_id,
    )
    return True


async def _close_if_snapshot_day_is_closed(session: AsyncSession, snapshot: Snapshot) -> bool:
    if snapshot.day_id is None:
        return False
    day = await session.get(Day, snapshot.day_id)
    if day is None or day.ended_at is None:
        return False
    await repo.close_snapshot(session, snapshot_id=snapshot.id, status="closed_after_day_end")
    return True


def snapshot_question_context(
    *,
    recent_entries,
    day_entries=None,
    user_settings: UserSettings,
    trigger: str,
    photo_prompt_opportunity: bool = False,
) -> dict[str, Any]:
    photo_prompt_style = "Не проси фото."
    if user_settings.photo_prompts_enabled:
        photo_prompt_style = (
            "Цього разу можна необов'язково запропонувати фото як відповідь або доповнення. "
            "Не вимагай фото і не роби з цього окремий великий діалог."
            if photo_prompt_opportunity
            else "Цього разу краще не просити фото, якщо тільки контекст явно не просить візуального моменту."
        )
    return {
        "recent_entries": [
            _entry_context(entry)
            for entry in recent_entries
        ],
        "day_context": _day_context(day_entries or []),
        "style": {
            "tone": user_settings.tone,
            "humanity_level": user_settings.humanity_level,
            "custom_interaction_style": custom_interaction_style(user_settings),
        },
        "user_profile_context": user_profile_context(user_settings),
        "question_preferences": {
            "ask_body_signals": user_settings.ask_body_signals,
            "photo_prompts_enabled": user_settings.photo_prompts_enabled,
            "photo_prompt_opportunity": photo_prompt_opportunity,
            "photo_prompt_style": photo_prompt_style,
            "body_signal_style": (
                "Можна іноді обережно питати про тілесні сигнали, якщо це доречно."
                if user_settings.ask_body_signals
                else "Не став окремих питань про тілесні сигнали."
            ),
        },
        "trigger": trigger,
    }


def _day_context(entries, *, limit: int = 80) -> dict[str, Any]:
    entries = list(entries)
    visible_entries = entries[-limit:]
    return {
        "entry_count": len(entries),
        "omitted_entry_count": max(0, len(entries) - len(visible_entries)),
        "entries": [_entry_context(entry) for entry in visible_entries],
    }


def _entry_context(entry) -> dict[str, Any]:
    return {
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "local_timestamp": entry.local_timestamp.isoformat() if getattr(entry, "local_timestamp", None) else None,
        "source": entry.source,
        "raw_text": entry.raw_text,
    }


def _question_query_text(context: dict[str, Any]) -> str:
    day_entries = context.get("day_context", {}).get("entries") or []
    recent_entries = context.get("recent_entries") or []
    parts = [
        *(str(entry.get("raw_text") or "") for entry in day_entries[-8:]),
        *(str(entry.get("raw_text") or "") for entry in recent_entries[-4:]),
    ]
    return " ".join(part for part in parts if part).strip()


async def _handle_open_snapshot(
    session: AsyncSession,
    *,
    bot: Bot,
    user: User,
    snapshot_id,
    prompted_at: datetime | None,
    user_settings: UserSettings,
) -> bool:
    if prompted_at is None:
        return False
    now = utc_now()
    minutes_since = (now - prompted_at).total_seconds() / 60
    prompts = await repo.get_snapshot_prompts(session, snapshot_id=snapshot_id)
    has_reminder = any(prompt.prompt_kind == "reminder" for prompt in prompts)
    if not has_reminder and minutes_since >= user_settings.reminder_delay_minutes:
        text = (
            "М’яко повертаюся до попереднього зрізу. "
            "Можна відповісти на нього, відкласти, або коротко позначити причину пропуску."
        )
        message = await bot.send_message(
            chat_id=user.chat_id,
            text=text,
            reply_markup=missed_prompt_keyboard(),
        )
        await repo.add_missed_prompt(
            session,
            user_id=user.id,
            snapshot_id=snapshot_id,
            prompt_id=prompts[-1].id if prompts else None,
            missed_at=now,
        )
        await repo.add_prompt(
            session,
            snapshot_id=snapshot_id,
            prompt_kind="reminder",
            text=text,
            sent_at=now,
            telegram_message_id=message.message_id,
        )
        return True

    if has_reminder and minutes_since >= user_settings.reminder_delay_minutes * 2:
        await repo.close_snapshot(session, snapshot_id=snapshot_id, status="missed")
    return False


def _is_active_time(now_utc: datetime, timezone: str, user_settings: UserSettings) -> bool:
    if snapshots_paused(user_settings):
        return False
    local = now_utc.astimezone(zoneinfo(timezone))
    start = parse_hhmm(user_settings.active_start)
    end = parse_hhmm(user_settings.active_end)
    current = local.time()
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end
