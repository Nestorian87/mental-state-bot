from __future__ import annotations

import logging

from aiogram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mental_state_bot.ai.service import AIService
from mental_state_bot.bot.keyboards import summary_detail_keyboard
from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
from mental_state_bot.services.snapshots import maybe_send_scheduled_snapshot
from mental_state_bot.services.summaries import SummaryService
from mental_state_bot.time_utils import local_now

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
                    reply_markup=summary_detail_keyboard(),
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
            if current_local.hour < 9 or current_local.hour > 12:
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
                    )

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
                    )
