from __future__ import annotations

import logging

from aiogram import Bot, Dispatcher

from mental_state_bot.ai.service import AIService
from mental_state_bot.bot.handlers import router
from mental_state_bot.config import Settings
from mental_state_bot.db.session import async_session_factory, create_async_engine_from_settings
from mental_state_bot.scheduler.jobs import build_scheduler
from mental_state_bot.services.interactions import InteractionService
from mental_state_bot.services.memory import MemoryService
from mental_state_bot.services.summaries import SummaryService

logger = logging.getLogger(__name__)


async def run_bot(settings: Settings) -> None:
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    engine = create_async_engine_from_settings(settings)
    sessionmaker = async_session_factory(engine)
    bot = Bot(token=settings.telegram_bot_token)
    dp = Dispatcher()

    ai_service = AIService(settings)
    interaction_service = InteractionService(settings, ai_service)
    memory_service = MemoryService(settings, ai_service)
    summary_service = SummaryService(settings, ai_service)

    dp["settings"] = settings
    dp["sessionmaker"] = sessionmaker
    dp["ai_service"] = ai_service
    dp["interaction_service"] = interaction_service
    dp["memory_service"] = memory_service
    dp["summary_service"] = summary_service
    dp.include_router(router)

    scheduler = build_scheduler(
        bot=bot,
        settings=settings,
        sessionmaker=sessionmaker,
        ai_service=ai_service,
        summary_service=summary_service,
    )
    scheduler.start()
    logger.info("Mental State Bot started")
    try:
        await dp.start_polling(bot)
    finally:
        scheduler.shutdown(wait=False)
        await bot.session.close()
        await engine.dispose()
