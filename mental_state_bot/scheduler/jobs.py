from __future__ import annotations

import logging
from uuid import UUID

from aiogram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mental_state_bot.ai.service import AIService
from mental_state_bot.bot.keyboards import (
    deferred_clarification_keyboard,
    period_detail_keyboard,
    summary_detail_keyboard,
)
from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
from mental_state_bot.services.interaction_gate import serialized_user_interaction
from mental_state_bot.services.memory_graph import maintain_memory_graph
from mental_state_bot.services.preferences import (
    clarification_queue,
    pending_clarification,
    pending_input,
    post_entry_followup_is_active,
    quiet_is_active,
    settings_json_with_clarification_queue,
    settings_json_with_pending_clarification,
    settings_json_without_pending_input,
    snapshots_paused,
)
from mental_state_bot.services.snapshots import _is_active_time, maybe_send_scheduled_snapshot
from mental_state_bot.services.summaries import SummaryService
from mental_state_bot.time_utils import local_now, utc_now

logger = logging.getLogger(__name__)


def build_scheduler(
    *,
    bot: Bot,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai_service: AIService,
    summary_service: SummaryService,
) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=settings.app_timezone)
    scheduler.add_job(
        scheduled_snapshot_tick,
        "interval",
        minutes=5,
        kwargs={
            "bot": bot,
            "settings": settings,
            "sessionmaker": sessionmaker,
            "ai_service": ai_service,
        },
        id="scheduled_snapshot_tick",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.add_job(
        morning_summary_tick,
        "interval",
        minutes=30,
        kwargs={
            "bot": bot,
            "settings": settings,
            "sessionmaker": sessionmaker,
            "summary_service": summary_service,
        },
        id="morning_summary_tick",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.add_job(
        period_summary_tick,
        "interval",
        minutes=60,
        kwargs={
            "bot": bot,
            "settings": settings,
            "sessionmaker": sessionmaker,
            "summary_service": summary_service,
        },
        id="period_summary_tick",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.add_job(
        clarification_queue_tick,
        "interval",
        minutes=10,
        kwargs={"bot": bot, "settings": settings, "sessionmaker": sessionmaker, "ai_service": ai_service},
        id="clarification_queue_tick",
        replace_existing=True,
        max_instances=1,
    )
    return scheduler


async def clarification_queue_tick(
    *,
    bot: Bot,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai_service: AIService,
) -> None:
    async with sessionmaker() as session, session.begin():
        users = await repo.list_active_users(session)
        for user in users:
            if settings.telegram_allowed_user_ids and user.telegram_user_id not in settings.telegram_allowed_user_ids:
                continue
            try:
                await _maybe_send_queued_clarification(
                    session,
                    bot=bot,
                    ai_service=ai_service,
                    user=user,
                    user_settings=await repo.get_user_settings(session, user.id),
                )
            except Exception:
                logger.exception("Failed to process clarification queue", extra={"user_id": str(user.id)})


@serialized_user_interaction
async def _maybe_send_queued_clarification(
    session: AsyncSession,
    *,
    bot: Bot,
    ai_service: AIService | None = None,
    user,
    user_settings,
) -> bool:
    local = local_now(user.timezone)
    if snapshots_paused(user_settings) or not _is_active_time(
        utc_now(), user.timezone, user_settings
    ):
        return False
    if (
        pending_clarification(user_settings) is not None
        or post_entry_followup_is_active(user_settings)
        or pending_input(user_settings) is not None
    ):
        return False
    if quiet_is_active(user_settings):
        return False
    if await repo.get_open_snapshot(session, user_id=user.id) is not None:
        return False

    queue = clarification_queue(user_settings)
    today = local.date().isoformat()
    item = await _select_queued_clarification(
        session,
        user=user,
        ai_service=ai_service,
        queue=queue,
        local_date=today,
    )
    if item is None:
        return False
    if item.get("status") == "skipped":
        next_queue = [
            item if other.get("id") == item.get("id") else other
            for other in queue
        ]
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_with_clarification_queue(user_settings, next_queue)},
        )
        return False

    queued_count = sum(1 for candidate in queue if candidate.get("status") == "queued")
    delivered_at = local.isoformat()
    grouped_ids = _grouped_clarification_ids(item)
    item = {**item, "status": "active", "delivered_at": delivered_at, "delivery_source": "automatic"}
    next_queue = [
        {
            **other,
            "status": "active",
            "delivered_at": delivered_at,
            "delivery_source": "automatic_grouped",
            "grouped_by": item.get("id"),
        }
        if str(other.get("id")) in grouped_ids and str(other.get("id")) != str(item.get("id"))
        else item if other.get("id") == item.get("id") else other
        for other in queue
    ]
    next_settings_view = await repo.update_user_settings(
        session,
        user_id=user.id,
        values={"settings_json": settings_json_with_pending_clarification(user_settings, item)},
    )
    await repo.update_user_settings(
        session,
        user_id=user.id,
        values={"settings_json": settings_json_with_clarification_queue(next_settings_view, next_queue)},
    )
    await bot.send_message(
        chat_id=user.chat_id,
        text=_queued_clarification_message(item, queued_count=queued_count),
        reply_markup=deferred_clarification_keyboard(item_id=str(item["id"]), options=item.get("options") or []),
    )
    return True


async def _select_queued_clarification(
    session: AsyncSession,
    *,
    user,
    ai_service: AIService | None,
    queue: list[dict],
    local_date: str,
) -> dict | None:
    queued = [item for item in queue if item.get("status") == "queued"]
    if not queued:
        return None
    reasons_used_today = {
        str(item.get("reason") or "")
        for item in queue
        if item.get("status") in {"active", "answered", "skipped"} and _clarification_touched_on_date(item, local_date)
    }
    diverse = [item for item in queued if str(item.get("reason") or "") not in reasons_used_today]
    candidates = sorted(diverse or queued, key=lambda item: str(item.get("created_at") or ""))[:8]
    if ai_service is None or len(candidates) <= 1:
        return candidates[0]
    review = await _review_clarification_candidates(
        session,
        user=user,
        ai_service=ai_service,
        candidates=candidates,
        local_date=local_date,
        reasons_used_today=sorted(reason for reason in reasons_used_today if reason),
    )
    if review is None:
        return candidates[0]
    candidate_by_id = {str(item.get("id")): item for item in candidates}
    selected_ids = [item_id for item_id in review.item_ids if item_id in candidate_by_id]
    if not review.should_ask:
        skip_id = selected_ids[0] if selected_ids else str(candidates[0].get("id"))
        return {
            **candidate_by_id.get(skip_id, candidates[0]),
            "status": "skipped",
            "skipped_at": local_now(user.timezone).isoformat(),
            "skip_source": "automatic_queue_review",
            "skip_reason": review.reason,
        }
    if not selected_ids:
        return candidates[0]
    primary = candidate_by_id[selected_ids[0]]
    question = " ".join(str(review.question or primary.get("question") or "").split())
    if not question:
        question = str(primary.get("question") or "")
    return {
        **primary,
        "question": question[:600],
        "options": [" ".join(str(option).split())[:80] for option in getattr(review, "options", [])[:4] if str(option).strip()]
        or list(primary.get("options") or [])[:4],
        "grouped_item_ids": selected_ids,
        "queue_review_reason": review.reason,
        "queue_review_confidence": review.confidence,
    }


async def _review_clarification_candidates(
    session: AsyncSession,
    *,
    user,
    ai_service: AIService,
    candidates: list[dict],
    local_date: str,
    reasons_used_today: list[str],
):
    entry_ids = [_uuid_or_none(str(item.get("entry_id") or "")) for item in candidates]
    entries = await repo.list_entries_by_ids(session, entry_ids=[entry_id for entry_id in entry_ids if entry_id])
    text_by_entry_id = {str(entry.id): entry.raw_text for entry in entries}
    review, _ = await ai_service.review_clarification_queue(
        session,
        user_id=user.id,
        context={
            "local_date": local_date,
            "current_local_time": local_now(user.timezone).strftime("%H:%M"),
            "reasons_used_today": reasons_used_today,
            "queued_items": [
                {
                    "id": str(item.get("id") or ""),
                    "reason": item.get("reason"),
                    "question": item.get("question"),
                    "created_at": item.get("created_at"),
                    "entry_text": _compact_text(text_by_entry_id.get(str(item.get("entry_id") or ""), "")),
                }
                for item in candidates
            ],
        },
    )
    return review


def _clarification_touched_on_date(item: dict, local_date: str) -> bool:
    return any(str(item.get(key) or "").startswith(local_date) for key in ("delivered_at", "answered_at", "skipped_at"))


def _queued_clarification_message(item: dict, *, queued_count: int) -> str:
    question = str(item.get("question") or "Є одне необов’язкове уточнення до попереднього запису.")
    if queued_count <= 1:
        return question
    return f"Є {queued_count} відкладені уточнення. Почну з одного:\n\n{question}"


def _grouped_clarification_ids(item: dict) -> set[str]:
    ids = {str(item.get("id") or "")}
    ids.update(str(value) for value in item.get("grouped_item_ids") or [] if value)
    return {value for value in ids if value}


def _uuid_or_none(value: str) -> UUID | None:
    try:
        return UUID(value)
    except ValueError:
        return None


def _compact_text(text: str, *, limit: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


async def scheduled_snapshot_tick(
    *,
    bot: Bot,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai_service: AIService,
) -> None:
    async with sessionmaker() as session, session.begin():
        users = await repo.list_active_users(session)
        for user in users:
            if settings.telegram_allowed_user_ids and user.telegram_user_id not in settings.telegram_allowed_user_ids:
                continue
            try:
                await maybe_send_scheduled_snapshot(
                    session,
                    bot=bot,
                    settings=settings,
                    ai_service=ai_service,
                    user=user,
                )
            except Exception:
                logger.exception("Failed to process scheduled snapshot", extra={"user_id": str(user.id)})


async def morning_summary_tick(
    *,
    bot: Bot,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    summary_service: SummaryService,
) -> None:
    async with sessionmaker() as session, session.begin():
        users = await repo.list_active_users(session)
        for user in users:
            if settings.telegram_allowed_user_ids and user.telegram_user_id not in settings.telegram_allowed_user_ids:
                continue
            current_local = local_now(user.timezone)
            if current_local.hour < 8 or current_local.hour > 12:
                continue
            try:
                summary = await summary_service.generate_yesterday_summary_if_needed(session, user=user)
            except Exception:
                logger.exception("Failed to generate morning summary", extra={"user_id": str(user.id)})
                continue
            if summary is not None:
                await bot.send_message(
                    chat_id=user.chat_id,
                    text="Ранковий підсумок за вчора:\n\n" + summary.short_text,
                    reply_markup=summary_detail_keyboard(summary_id=str(summary.id)),
                )
            user_settings = await repo.get_user_settings(session, user.id)
            if pending_input(user_settings) == "sleep_reflection":
                await repo.update_user_settings(
                    session,
                    user_id=user.id,
                    values={"settings_json": settings_json_without_pending_input(user_settings)},
                )


async def period_summary_tick(
    *,
    bot: Bot,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    summary_service: SummaryService,
) -> None:
    async with sessionmaker() as session, session.begin():
        users = await repo.list_active_users(session)
        for user in users:
            if settings.telegram_allowed_user_ids and user.telegram_user_id not in settings.telegram_allowed_user_ids:
                continue
            current_local = local_now(user.timezone)
            if current_local.hour < 9:
                continue

            if current_local.weekday() == 0:
                try:
                    weekly = await summary_service.generate_previous_week_summary_if_needed(session, user=user)
                except Exception:
                    logger.exception("Failed to generate weekly summary", extra={"user_id": str(user.id)})
                    weekly = None
                if weekly is not None:
                    await bot.send_message(
                        chat_id=user.chat_id,
                        text="Підсумок за попередній тиждень:\n\n" + weekly.short_text,
                        reply_markup=period_detail_keyboard(summary_id=str(weekly.id)),
                    )
                    await repo.mark_summary_delivered(session, summary_id=weekly.id, delivered_at=utc_now())
                    try:
                        await maintain_memory_graph(session, user_id=user.id)
                    except Exception:
                        logger.exception("Failed to maintain memory graph", extra={"user_id": str(user.id)})

            if current_local.day == 1:
                try:
                    monthly = await summary_service.generate_previous_month_summary_if_needed(session, user=user)
                except Exception:
                    logger.exception("Failed to generate monthly summary", extra={"user_id": str(user.id)})
                    monthly = None
                if monthly is not None:
                    await bot.send_message(
                        chat_id=user.chat_id,
                        text="Підсумок за попередній місяць:\n\n" + monthly.short_text,
                        reply_markup=period_detail_keyboard(summary_id=str(monthly.id)),
                    )
                    await repo.mark_summary_delivered(session, summary_id=monthly.id, delivered_at=utc_now())
