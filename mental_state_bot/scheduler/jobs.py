from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

from aiogram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mental_state_bot.ai.service import AIService
from mental_state_bot.bot.keyboards import (
    memory_graph_confirmation_keyboard,
    period_detail_keyboard,
    summary_detail_keyboard,
)
from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
from mental_state_bot.services.memory_graph import (
    MemoryGraphConfirmationCandidate,
    maintain_memory_graph,
    review_memory_graph_duplicates,
)
from mental_state_bot.services.preferences import (
    memory_graph_confirmation_last_offer_at,
    memory_graph_confirmation_queue,
    memory_graph_last_consolidated_week,
    pending_clarification,
    pending_input,
    pending_memory_graph_confirmation,
    post_entry_followup_is_active,
    quiet_is_active,
    settings_json_with_memory_graph_confirmation_last_offer,
    settings_json_with_memory_graph_confirmation_queue,
    settings_json_with_memory_graph_consolidated_week,
    settings_json_with_pending_memory_graph_confirmation,
    settings_json_without_pending_input,
)
from mental_state_bot.services.snapshots import maybe_send_scheduled_snapshot
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
            "ai_service": ai_service,
            "summary_service": summary_service,
        },
        id="period_summary_tick",
        replace_existing=True,
        max_instances=1,
    )
    return scheduler


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
                continue
            if summary is None:
                try:
                    await _maybe_offer_memory_graph_confirmation(session, bot=bot, user=user)
                except Exception:
                    logger.exception("Failed to offer memory graph confirmation", extra={"user_id": str(user.id)})


async def period_summary_tick(
    *,
    bot: Bot,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai_service: AIService,
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
                    await _run_weekly_memory_graph_consolidation(
                        session,
                        user=user,
                        ai_service=ai_service,
                        current_local=current_local,
                    )
                except Exception:
                    logger.exception("Failed to consolidate memory graph", extra={"user_id": str(user.id)})

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


async def _run_weekly_memory_graph_consolidation(
    session: AsyncSession,
    *,
    user,
    ai_service: AIService,
    current_local,
) -> None:
    """Do one bounded, silent graph consolidation per local calendar week."""
    settings = await repo.get_user_settings(session, user.id)
    iso_year, iso_week, _ = current_local.isocalendar()
    week_key = f"{iso_year}-W{iso_week:02d}"
    if memory_graph_last_consolidated_week(settings) == week_key:
        return

    maintenance = await maintain_memory_graph(session, user_id=user.id)
    review = await review_memory_graph_duplicates(
        session,
        user_id=user.id,
        ai_service=ai_service,
        pair_limit=12,
        use_embedding_candidates=True,
        use_heavy_reasoning=True,
    )
    settings_json = _settings_with_queued_memory_graph_confirmations(settings, review.confirmation_candidates)
    await repo.update_user_settings(
        session,
        user_id=user.id,
        values={
            "settings_json": settings_json_with_memory_graph_consolidated_week(
                _settings_view(settings_json),
                week_key,
            )
        },
    )
    logger.info(
        "Weekly memory graph consolidation completed",
        extra={
            "user_id": str(user.id),
            "week": week_key,
            "nodes_checked": maintenance.nodes_checked,
            "candidate_pairs": review.pairs_selected,
            "embedding_pairs": review.embedding_pairs_found,
            "decisions": review.decisions_received,
        },
    )


def _settings_with_queued_memory_graph_confirmations(
    settings,
    candidates: tuple[MemoryGraphConfirmationCandidate, ...],
) -> dict:
    queue = memory_graph_confirmation_queue(settings)
    active_pairs = {
        tuple(sorted([str(item.get("left_node_id") or ""), str(item.get("right_node_id") or "")]))
        for item in queue
        if item.get("status") in {"queued", "active"}
    }
    now = datetime.now(UTC).isoformat()
    for candidate in candidates:
        pair = tuple(sorted([str(candidate.left_node_id), str(candidate.right_node_id)]))
        if pair in active_pairs:
            continue
        queue.append(
            {
                "id": str(uuid.uuid4()),
                "left_node_id": str(candidate.left_node_id),
                "right_node_id": str(candidate.right_node_id),
                "question": candidate.question,
                "options": list(candidate.options),
                "reason": candidate.reason,
                "status": "queued",
                "created_at": now,
            }
        )
        active_pairs.add(pair)
    return settings_json_with_memory_graph_confirmation_queue(settings, queue)


async def _maybe_offer_memory_graph_confirmation(session: AsyncSession, *, bot: Bot, user) -> bool:
    settings = await repo.get_user_settings(session, user.id)
    if (
        pending_input(settings) is not None
        or pending_clarification(settings) is not None
        or pending_memory_graph_confirmation(settings) is not None
        or post_entry_followup_is_active(settings)
        or quiet_is_active(settings)
        or await repo.get_open_snapshot(session, user_id=user.id) is not None
    ):
        return False
    last_offer = memory_graph_confirmation_last_offer_at(settings)
    if last_offer is not None and datetime.now(UTC) - last_offer < timedelta(days=3):
        return False
    queue = memory_graph_confirmation_queue(settings)
    item = next((candidate for candidate in queue if candidate.get("status") == "queued"), None)
    if item is None:
        return False
    item = {**item, "status": "active", "offered_at": datetime.now(UTC).isoformat(), "delivery_source": "automatic"}
    updated_queue = [item if candidate.get("id") == item.get("id") else candidate for candidate in queue]
    settings_json = settings_json_with_memory_graph_confirmation_queue(settings, updated_queue)
    settings_json = settings_json_with_pending_memory_graph_confirmation(_settings_view(settings_json), item)
    settings_json = settings_json_with_memory_graph_confirmation_last_offer(
        _settings_view(settings_json),
        datetime.now(UTC),
    )
    await repo.update_user_settings(session, user_id=user.id, values={"settings_json": settings_json})
    await bot.send_message(
        chat_id=user.chat_id,
        text=item["question"],
        reply_markup=memory_graph_confirmation_keyboard(item_id=str(item["id"]), options=item.get("options") or []),
    )
    return True


def _settings_view(settings_json: dict):
    return type("SettingsView", (), {"settings_json": settings_json})()
