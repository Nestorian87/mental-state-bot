from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from uuid import UUID

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, CallbackQuery, FSInputFile, InputMediaPhoto, Message
from aiogram.utils.chat_action import ChatActionSender
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mental_state_bot.ai.service import AIService
from mental_state_bot.bot.keyboards import (
    correction_keyboard,
    day_detail_keyboard,
    main_reply_keyboard,
    period_detail_keyboard,
    settings_keyboard,
    sleep_confirmation_keyboard,
    snapshot_clarification_keyboard,
    summary_detail_keyboard,
    voice_transcription_keyboard,
)
from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import Day, User, UserSettings
from mental_state_bot.services.archive_audit import build_archive_audit, format_archive_audit
from mental_state_bot.services.exports import export_user_archive
from mental_state_bot.services.interactions import BotReply, InteractionResult, InteractionService
from mental_state_bot.services.memory import MemoryService
from mental_state_bot.services.preferences import (
    custom_interaction_style,
    pending_input,
    pending_voice_transcript,
    settings_json_with_custom_interaction_style,
    settings_json_with_pending_input,
    settings_json_with_pending_voice_transcript,
    settings_json_with_snapshot_pause,
    settings_json_with_user_profile_context,
    settings_json_without_pending_input,
    settings_json_without_pending_voice_transcript,
    snapshots_paused,
    user_profile_context,
)
from mental_state_bot.services.review import (
    PhotoMoment,
    build_metrics_chart_png,
    build_metrics_chart_png_for_day,
    build_period_metrics_chart_png,
    format_cost_report,
    format_day_summary_section,
    format_day_view,
    format_gaps_for_day,
    format_gaps_view,
    format_latest_summary_section,
    format_metrics_for_day,
    format_metrics_view,
    format_period_days_view,
    format_period_metrics_view,
    format_period_summary,
    format_period_timeline_view,
    format_photo_moments_view,
    format_raw_entries_for_day,
    format_raw_entries_view,
    format_similar_entries,
    format_summary_section,
    format_today_view,
    get_photo_moments_for_day,
    get_today_photo_moments,
)
from mental_state_bot.services.snapshots import send_snapshot_prompt
from mental_state_bot.services.summaries import SummaryService
from mental_state_bot.time_utils import zoneinfo

logger = logging.getLogger(__name__)
router = Router()

SLEEP_MARKER_TEXT = "лягаю спати"
_DAY_DETAIL_SECTIONS = {"timeline", "metrics", "photos", "raw", "gaps"}


@dataclass(frozen=True)
class VoiceNoteTranscription:
    text: str
    original_text: str | None
    original_path: Path | None
    transcription_path: Path | None
    transcription_run_id: UUID | None
    duration_seconds: int | None
    mime_type: str | None
    file_size: int | None
    telegram_file_id: str
    telegram_file_unique_id: str
    error: str | None = None


@router.message(CommandStart())
async def start_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        await _get_or_create_message_user(session, message, settings)
    await message.answer(
        "Готово. Я буду тихо збирати короткі зрізи дня. Можеш також писати сюди будь-який момент сам.",
        reply_markup=main_reply_keyboard(),
    )


@router.message(Command("status"))
async def status_handler(message: Message, settings: Settings) -> None:
    if not await _allowed(message, settings):
        return
    await message.answer("Я працюю. Якщо є відкритий зріз, можна відповісти текстом або кнопками.")


@router.message(Command("help"))
async def help_handler(message: Message, settings: Settings) -> None:
    if not await _allowed(message, settings):
        return
    await message.answer(_help_text(), reply_markup=main_reply_keyboard())


@router.message(Command("snapshot"))
async def snapshot_command_handler(
    message: Message,
    bot: Bot,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai_service: AIService,
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        async with _typing(message, bot):
            sent = await send_snapshot_prompt(
                session,
                bot=bot,
                ai_service=ai_service,
                user=user,
                user_settings=user_settings,
                intent="manual_snapshot",
                scheduled_for=None,
                photo_prompt_chance=settings.photo_prompt_chance,
            )
    if not sent:
        await message.answer("Зараз уже є відкритий зріз. Можна відповісти на нього або відкласти кнопкою.")


@router.message(F.text == "Новий зріз")
async def snapshot_button_handler(
    message: Message,
    bot: Bot,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai_service: AIService,
) -> None:
    await snapshot_command_handler(message, bot, settings, sessionmaker, ai_service)


@router.message(Command("today"))
async def today_command_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        text = await format_today_view(session, user=user)
    await _answer_long_text(message, text)


@router.message(F.text == "Сьогодні")
async def today_button_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    await today_command_handler(message, settings, sessionmaker)


@router.message(Command("day"))
async def day_command_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    query = _command_argument(message.text or "")
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        target_date = _parse_day_query(query, user.timezone)
        if target_date is None:
            await message.answer("Формат: /day 2026-06-30. Також можна: /day вчора або /day сьогодні.")
            return
        day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=target_date)
        if day is None:
            await message.answer(f"За {target_date.isoformat()} ще немає збереженого дня.")
            return
        day_id = str(day.id)
        text = await format_day_view(session, user=user, day=day, limit=30)
    await _answer_long_text(message, text, reply_markup=day_detail_keyboard(day_id=day_id))


@router.message(Command("costs"))
async def costs_command_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        text = await format_cost_report(session, user=user)
    await _answer_long_text(message, text)


@router.message(Command("audit"))
async def audit_command_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        audit = await build_archive_audit(session, settings=settings, user=user)
        text = format_archive_audit(audit)
    await _answer_long_text(message, text)


@router.message(Command("metrics"))
async def metrics_command_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        text = await format_metrics_view(session, user=user)
        chart = await build_metrics_chart_png(session, user=user)
    await _answer_long_text(message, text)
    if chart is not None:
        await message.answer_photo(
            BufferedInputFile(chart, filename="metrics.png"),
            caption="Графік: синя лінія — настрій, зелена — енергія.",
        )


@router.message(F.text == "Метрики")
async def metrics_button_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    await metrics_command_handler(message, settings, sessionmaker)


@router.message(Command("photos"))
async def photos_command_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        moments = await get_today_photo_moments(session, user=user)
        text = format_photo_moments_view(moments, timezone=user.timezone)
    await _answer_long_text(message, text)
    await _send_photo_moments(message, moments)


@router.message(F.text == "Фото дня")
async def photos_button_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    await photos_command_handler(message, settings, sessionmaker)


@router.message(Command("gaps"))
async def gaps_command_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        text = await format_gaps_view(session, user=user)
    await _answer_long_text(message, text)


@router.message(F.text == "Прогалини")
async def gaps_button_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    await gaps_command_handler(message, settings, sessionmaker)


@router.message(Command("raw"))
async def raw_command_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        text = await format_raw_entries_view(session, user=user)
    await _answer_long_text(message, text)


@router.message(Command("week"))
async def week_command_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    summary_service: SummaryService,
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        async with _typing(message):
            if _is_previous_period_query(_command_argument(message.text or "")):
                summary = await summary_service.generate_previous_week_summary(session, user=user)
            else:
                summary = await summary_service.generate_current_week_summary(session, user=user)
        text = format_period_summary(summary)
        summary_id = str(summary.id)
    await _answer_long_text(message, text, reply_markup=period_detail_keyboard(summary_id=summary_id))


@router.message(Command("month"))
async def month_command_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    summary_service: SummaryService,
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        async with _typing(message):
            if _is_previous_period_query(_command_argument(message.text or "")):
                summary = await summary_service.generate_previous_month_summary(session, user=user)
            else:
                summary = await summary_service.generate_current_month_summary(session, user=user)
        text = format_period_summary(summary)
        summary_id = str(summary.id)
    await _answer_long_text(message, text, reply_markup=period_detail_keyboard(summary_id=summary_id))


@router.message(Command("report"))
async def report_command_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        metrics = await format_metrics_view(session, user=user)
        timeline = await format_today_view(session, user=user, limit=24)
    await _answer_long_text(message, timeline + "\n\n" + metrics)


@router.message(Command("similar"))
async def similar_command_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    memory_service: MemoryService,
) -> None:
    if not await _allowed(message, settings):
        return
    query = _command_argument(message.text or "")
    if not query:
        await message.answer("Формат: /similar лежу порожньо не можу почати")
        return
    if not settings.embeddings_enabled or not settings.embedding_api_key:
        await message.answer("Semantic search ще не активний: треба EMBEDDING_API_KEY і EMBEDDINGS_ENABLED=true.")
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        async with _typing(message):
            records = await memory_service.similar_entries(session, user_id=user.id, query_text=query, limit=6)
        text = format_similar_entries(list(records))
    await _answer_long_text(message, text)


@router.message(Command("settings"))
async def settings_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        paused = snapshots_paused(user_settings)
    await message.answer(
        _format_settings_text(user_settings=user_settings, snapshots_are_paused=paused),
        reply_markup=settings_keyboard(user_settings=user_settings),
    )


@router.message(F.text == "Налаштування")
async def settings_button_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    await settings_handler(message, settings, sessionmaker)


@router.message(F.text.regexp(r"(?is)^\s*(стиль|style)\s*:"))
async def custom_style_message_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    style = _custom_style_text(message.text or "")
    if not style:
        await message.answer(
            "Напиши після `стиль:` як саме хочеш, щоб бот формулював питання й короткі відповіді.",
            reply_markup=main_reply_keyboard("Опиши бажаний стиль"),
        )
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        next_settings_json = settings_json_with_custom_interaction_style(user_settings, style)
        next_settings_json.pop("pending_input", None)
        user_settings = await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": next_settings_json},
        )
        text = _format_settings_text(
            user_settings=user_settings,
            snapshots_are_paused=snapshots_paused(user_settings),
        )
    await message.answer("Записав власний стиль взаємодії.")
    await message.answer(text, reply_markup=settings_keyboard(user_settings=user_settings))


@router.message(F.text.regexp(r"(?is)^\s*(скинути стиль|reset style)\s*$"))
async def reset_custom_style_message_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        user_settings = await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": _without_pending(settings_json_with_custom_interaction_style(user_settings, None))},
        )
        text = _format_settings_text(
            user_settings=user_settings,
            snapshots_are_paused=snapshots_paused(user_settings),
        )
    await message.answer("Власний стиль скинуто.")
    await message.answer(text, reply_markup=settings_keyboard(user_settings=user_settings))


@router.message(F.text.regexp(r"(?is)^\s*(контекст|context)\s*:"))
async def profile_context_message_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    context = _prefixed_text(message.text or "")
    if not context:
        await message.answer(
            "Напиши після `контекст:` кілька речень про себе, роботу, побут або слова, які бот має розуміти.",
            reply_markup=main_reply_keyboard("Напиши контекст про себе"),
        )
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        next_settings_json = settings_json_with_user_profile_context(user_settings, context)
        next_settings_json.pop("pending_input", None)
        user_settings = await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": next_settings_json},
        )
        text = _format_settings_text(
            user_settings=user_settings,
            snapshots_are_paused=snapshots_paused(user_settings),
        )
    await message.answer("Записав загальний контекст про тебе.")
    await message.answer(text, reply_markup=settings_keyboard(user_settings=user_settings))


@router.message(F.text.regexp(r"(?is)^\s*(скинути контекст|reset context)\s*$"))
async def reset_profile_context_message_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        user_settings = await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": _without_pending(settings_json_with_user_profile_context(user_settings, None))},
        )
        text = _format_settings_text(
            user_settings=user_settings,
            snapshots_are_paused=snapshots_paused(user_settings),
        )
    await message.answer("Загальний контекст скинуто.")
    await message.answer(text, reply_markup=settings_keyboard(user_settings=user_settings))


@router.message(F.text.regexp(r"(?is)^\s*(виправлення|не так|correction)\s*:"))
async def correction_message_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    interaction_service: InteractionService,
    memory_service: MemoryService,
) -> None:
    if not await _allowed(message, settings):
        return
    correction = _prefixed_text(message.text or "")
    if not correction:
        await message.answer(
            "Напиши після `виправлення:` що саме бот зрозумів не так.",
            reply_markup=main_reply_keyboard("Напиши, що я зрозумів не так"),
        )
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        if pending_input(user_settings):
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_without_pending_input(user_settings)},
            )
        result = await interaction_service.record_correction(
            session,
            user=user,
            correction_text=correction,
            telegram_message_id=message.message_id,
            reply_to_message_id=message.reply_to_message.message_id if message.reply_to_message else None,
        )
        entry_id = result.entry_id
        user_id = user.id
        replies = result.replies
        should_embed = result.should_embed_entry
    for reply in replies:
        await message.answer(reply.text, reply_markup=_inline_reply_keyboard(reply.keyboard))
    if should_embed and entry_id and user_id:
        asyncio.create_task(_embed_entry_task(settings, sessionmaker, memory_service, entry_id, user_id))


@router.message(Command("pause"))
async def pause_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        await repo.set_user_active(session, user_id=user.id, is_active=True)
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_with_snapshot_pause(user_settings, True)},
        )
    await message.answer(
        "Ок, я поставив автоматичні зрізи на паузу. Ручні записи й підсумки лишаються доступними."
    )


@router.message(Command("resume"))
async def resume_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        await repo.set_user_active(session, user_id=user.id, is_active=True)
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_with_snapshot_pause(user_settings, False)},
        )
    await message.answer("Ок, автоматичні зрізи знову увімкнені.")


@router.message(Command("set_active"))
async def set_active_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    parts = (message.text or "").split()
    if len(parts) != 3 or not _valid_hhmm(parts[1]) or not _valid_hhmm(parts[2]):
        await message.answer("Формат: /set_active 09:00 23:30")
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"active_start": parts[1], "active_end": parts[2]},
        )
    await message.answer(f"Ок, активні години: {parts[1]}-{parts[2]}")


@router.message(Command("set_frequency"))
async def set_frequency_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    parts = (message.text or "").split()
    if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
        await message.answer("Формат: /set_frequency 30 70")
        return
    min_minutes = int(parts[1])
    max_minutes = int(parts[2])
    if min_minutes < 10 or max_minutes < min_minutes:
        await message.answer("Мінімум має бути від 10 хв, максимум не менший за мінімум.")
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"min_interval_minutes": min_minutes, "max_interval_minutes": max_minutes},
        )
    await message.answer(f"Ок, інтервал зрізів: {min_minutes}-{max_minutes} хв")


@router.message(Command("set_reminder"))
async def set_reminder_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    parts = (message.text or "").split()
    if len(parts) != 2 or not parts[1].isdigit():
        await message.answer("Формат: /set_reminder 25")
        return
    minutes = int(parts[1])
    if minutes < 5:
        await message.answer("Нагадування не має бути частіше ніж через 5 хв.")
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"reminder_delay_minutes": minutes},
        )
    await message.answer(f"Ок, м’яке нагадування після {minutes} хв.")


@router.message(Command("summary"))
async def summary_command_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    summary_service: SummaryService,
) -> None:
    if not await _allowed(message, settings):
        return
    summary = None
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        async with _typing(message):
            summary = await summary_service.generate_today_summary(session, user=user)
    await message.answer(summary.short_text, reply_markup=summary_detail_keyboard(summary_id=str(summary.id)))


@router.message(Command("sleep"))
async def sleep_command_handler(
    message: Message,
    settings: Settings,
) -> None:
    if not await _allowed(message, settings):
        return
    await message.answer("Закрити день і згенерувати підсумок?", reply_markup=sleep_confirmation_keyboard())


@router.message(Command("export"))
async def export_command_handler(message: Message, settings: Settings) -> None:
    if not await _allowed(message, settings):
        return
    output = Path("./data") / f"export-{message.from_user.id}.json"
    await export_user_archive(settings, message.from_user.id, output, format="json")
    await message.answer_document(FSInputFile(output), caption="Експорт архіву готовий.")


@router.message(Command("export_md"))
async def export_markdown_command_handler(message: Message, settings: Settings) -> None:
    if not await _allowed(message, settings):
        return
    output = Path("./data") / f"export-{message.from_user.id}.md"
    await export_user_archive(settings, message.from_user.id, output, format="markdown")
    await message.answer_document(FSInputFile(output), caption="Markdown-архів готовий.")


@router.message(Command("export_csv"))
async def export_csv_command_handler(message: Message, settings: Settings) -> None:
    if not await _allowed(message, settings):
        return
    output = Path("./data") / f"metrics-{message.from_user.id}.csv"
    await export_user_archive(settings, message.from_user.id, output, format="csv")
    await message.answer_document(FSInputFile(output), caption="CSV з метриками готовий.")


@router.message(Command("export_zip"))
async def export_zip_command_handler(message: Message, settings: Settings) -> None:
    if not await _allowed(message, settings):
        return
    output = Path("./data") / f"archive-{message.from_user.id}.zip"
    await export_user_archive(settings, message.from_user.id, output, format="zip")
    await message.answer_document(FSInputFile(output), caption="ZIP-архів з даними готовий.")


@router.callback_query(F.data == "snapshot:new")
async def snapshot_new_callback_handler(
    callback: CallbackQuery,
    bot: Bot,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai_service: AIService,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        sent = await send_snapshot_prompt(
            session,
            bot=bot,
            ai_service=ai_service,
            user=user,
            user_settings=user_settings,
            intent="manual_snapshot",
            scheduled_for=None,
            photo_prompt_chance=settings.photo_prompt_chance,
        )
    await callback.answer()
    if not sent:
        await callback.message.answer("Зараз уже є відкритий зріз. Можна відповісти на нього або відкласти.")


@router.callback_query(F.data == "day:today")
async def today_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        text = await format_today_view(session, user=user)
    await callback.answer()
    await _answer_long_text(callback.message, text)


@router.callback_query(F.data == "day:metrics")
async def metrics_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        text = await format_metrics_view(session, user=user)
        chart = await build_metrics_chart_png(session, user=user)
    await callback.answer()
    await _answer_long_text(callback.message, text)
    if chart is not None:
        await callback.message.answer_photo(
            BufferedInputFile(chart, filename="metrics.png"),
            caption="Графік: синя лінія — настрій, зелена — енергія.",
        )


@router.callback_query(F.data == "day:gaps")
async def gaps_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        text = await format_gaps_view(session, user=user)
    await callback.answer()
    await _answer_long_text(callback.message, text)


@router.callback_query(F.data.startswith("dayview:"))
async def day_detail_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    parsed = _parse_scoped_callback(callback.data or "", prefix="dayview")
    if parsed is None:
        await callback.answer("Не можу відкрити цей день")
        return
    day_id, section = parsed
    chart = None
    moments: list[PhotoMoment] = []
    reply_day_id = None
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        day = await repo.get_day(session, day_id=day_id)
        if day is None or day.user_id != user.id:
            text = "Не знайшов цей день в архіві."
        else:
            reply_day_id = str(day.id)
            text, chart, moments = await _format_day_detail_section(session, user=user, day=day, section=section)
    await callback.answer()
    await _answer_long_text(
        callback.message,
        text,
        reply_markup=day_detail_keyboard(day_id=reply_day_id) if reply_day_id else None,
    )
    if chart is not None:
        await callback.message.answer_photo(
            BufferedInputFile(chart, filename="metrics.png"),
            caption="Графік: синя лінія — настрій, зелена — енергія.",
        )
    if moments:
        await _send_photo_moments(callback.message, moments)


@router.callback_query(F.data.startswith("summary:"))
async def summary_detail_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    parts = (callback.data or "").split(":")
    section = parts[-1] if len(parts) >= 2 else "story"
    chart = None
    moments: list[PhotoMoment] = []
    reply_markup = None
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        summary = None
        if len(parts) == 3:
            summary_id = _uuid_or_none(parts[1])
            summary = await repo.get_summary(session, summary_id=summary_id) if summary_id else None
            if summary is None or summary.user_id != user.id:
                text = "Не знайшов цей підсумок."
            else:
                reply_markup = summary_detail_keyboard(summary_id=str(summary.id))
                day = await repo.get_day(session, day_id=summary.day_id) if summary.day_id else None
                if day is not None and day.user_id == user.id and section in _DAY_DETAIL_SECTIONS:
                    text, chart, moments = await _format_day_detail_section(
                        session,
                        user=user,
                        day=day,
                        section=section,
                    )
                else:
                    text = format_summary_section(summary, section)
        else:
            summary = await repo.get_latest_summary(session, user_id=user.id, period_type="daily")
            if summary is not None:
                reply_markup = summary_detail_keyboard(summary_id=str(summary.id))
                day = await repo.get_day(session, day_id=summary.day_id) if summary.day_id else None
                if day is not None and day.user_id == user.id and section in _DAY_DETAIL_SECTIONS:
                    text, chart, moments = await _format_day_detail_section(
                        session,
                        user=user,
                        day=day,
                        section=section,
                    )
                else:
                    text = format_summary_section(summary, section)
            elif section == "raw":
                text = await format_raw_entries_view(session, user=user)
            elif section == "timeline":
                text = await format_today_view(session, user=user, limit=30)
            elif section == "metrics":
                text = await format_metrics_view(session, user=user)
                chart = await build_metrics_chart_png(session, user=user)
            elif section == "photos":
                moments = await get_today_photo_moments(session, user=user)
                text = format_photo_moments_view(moments, timezone=user.timezone)
            else:
                text = await format_latest_summary_section(session, user=user, section=section)
    await callback.answer()
    await _answer_long_text(callback.message, text, reply_markup=reply_markup)
    if chart is not None:
        await callback.message.answer_photo(
            BufferedInputFile(chart, filename="metrics.png"),
            caption="Графік: синя лінія — настрій, зелена — енергія.",
        )
    if moments:
        await _send_photo_moments(callback.message, moments)


@router.callback_query(F.data == "nav:home")
async def home_callback_handler(callback: CallbackQuery, settings: Settings) -> None:
    if not await _allowed_callback(callback, settings):
        return
    await callback.answer()
    await callback.message.answer("Головне меню.", reply_markup=main_reply_keyboard())


@router.callback_query(F.data == "ai:costs")
async def costs_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        text = await format_cost_report(session, user=user)
    await callback.answer()
    await _answer_long_text(callback.message, text)


@router.callback_query(F.data == "archive:audit")
async def audit_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        audit = await build_archive_audit(session, settings=settings, user=user)
        text = format_archive_audit(audit)
    await callback.answer()
    await _answer_long_text(callback.message, text)


@router.callback_query(F.data.startswith("settings:"))
async def settings_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    action = callback.data or "settings:open"
    notice = "Налаштування."
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        if action == "settings:pause":
            await repo.set_user_active(session, user_id=user.id, is_active=True)
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_snapshot_pause(user_settings, True)},
            )
            notice = "Автоматичні зрізи на паузі."
        elif action == "settings:resume":
            await repo.set_user_active(session, user_id=user.id, is_active=True)
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_snapshot_pause(user_settings, False)},
            )
            notice = "Автоматичні зрізи увімкнені."
        elif action.startswith("settings:freq:"):
            preset = action.rsplit(":", maxsplit=1)[1]
            values = _frequency_preset_values(preset)
            user_settings = await repo.update_user_settings(session, user_id=user.id, values=values)
            notice = f"Інтервал: {values['min_interval_minutes']}-{values['max_interval_minutes']} хв."
        elif action.startswith("settings:reminder:"):
            minutes_text = action.rsplit(":", maxsplit=1)[1]
            minutes = int(minutes_text) if minutes_text.isdigit() else user_settings.reminder_delay_minutes
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"reminder_delay_minutes": max(5, minutes)},
            )
            notice = f"Нагадування після {user_settings.reminder_delay_minutes} хв."
        elif action == "settings:toggle:body":
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"ask_body_signals": not user_settings.ask_body_signals},
            )
            notice = "Питання про тілесні сигнали оновлено."
        elif action == "settings:toggle:photo":
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"photo_prompts_enabled": not user_settings.photo_prompts_enabled},
            )
            notice = "Фото-підказки оновлено."
        elif action.startswith("settings:tone:"):
            tone = action.rsplit(":", maxsplit=1)[1]
            if tone not in {"precise", "calm"}:
                tone = "calm"
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"tone": tone},
            )
            notice = "Тон відповідей оновлено."
        elif action.startswith("settings:humanity:"):
            humanity_level = action.rsplit(":", maxsplit=1)[1]
            if humanity_level not in {"balanced", "warm"}:
                humanity_level = "balanced"
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"humanity_level": humanity_level},
            )
            notice = "Стиль взаємодії оновлено."
        elif action == "settings:custom_style":
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_pending_input(user_settings, "custom_style")},
            )
            notice = "Можна написати власний стиль."
        elif action == "settings:profile_context":
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_pending_input(user_settings, "profile_context")},
            )
            notice = "Можна написати загальний контекст."
        text = _format_settings_text(
            user_settings=user_settings,
            snapshots_are_paused=snapshots_paused(user_settings),
        )
        keyboard = settings_keyboard(user_settings=user_settings)
    await callback.answer(notice)
    if action == "settings:custom_style":
        await callback.message.answer(
            "Напиши наступним повідомленням, як саме мені формулювати питання й короткі відповіді.\n\n"
            "Наприклад: коротко, без підбадьорювання, але не сухо; якщо відповідь нечітка — уточнюй м’яко.\n\n"
            "Також можна будь-коли написати `стиль: ...` або `скинути стиль`.",
            reply_markup=main_reply_keyboard("Опиши бажаний стиль"),
        )
        return
    if action == "settings:profile_context":
        await callback.message.answer(
            "Напиши наступним повідомленням загальний контекст про себе: хто ти, чим займаєшся, що зараз важливо знати, які слова або обставини мені варто розуміти.\n\n"
            "Також можна будь-коли написати `контекст: ...` або `скинути контекст`.",
            reply_markup=main_reply_keyboard("Напиши контекст про себе"),
        )
        return
    try:
        await callback.message.edit_text(text, reply_markup=keyboard)
    except Exception:
        await callback.message.answer(text, reply_markup=keyboard)


@router.callback_query(F.data == "correction:start")
async def correction_start_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_with_pending_input(user_settings, "correction")},
        )
    await callback.answer()
    await _clear_inline_keyboard(callback)
    await callback.message.answer(
        "Напиши наступним повідомленням, що саме я зрозумів не так і як це краще трактувати.\n\n"
        "Також можна будь-коли написати `виправлення: ...`.",
        reply_markup=main_reply_keyboard("Напиши, що я зрозумів не так"),
    )


@router.callback_query(F.data.startswith("voice:"))
async def voice_transcription_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    interaction_service: InteractionService,
    memory_service: MemoryService,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    action = (callback.data or "").split(":", maxsplit=1)[1]
    entry_id: UUID | None = None
    should_embed = False
    replies: list[BotReply] = []
    user_id: UUID | None = None

    await _clear_inline_keyboard(callback)
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_id = user.id
        user_settings = await repo.get_user_settings(session, user.id)
        if action == "cancel":
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_without_pending_voice_transcript(user_settings)},
            )
            replies = [BotReply("Ок, не записую це голосове.")]
        elif action == "fix":
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={
                    "settings_json": settings_json_with_pending_input(
                        user_settings,
                        "voice_transcript_fix",
                    )
                },
            )
            replies = [BotReply("Напиши наступним повідомленням правильний текст голосового.")]
        else:
            async with _typing(callback.message):
                result = await _confirm_pending_voice_transcript(
                    session=session,
                    user=user,
                    user_settings=user_settings,
                    interaction_service=interaction_service,
                )
            entry_id = result.entry_id
            should_embed = result.should_embed_entry
            replies = result.replies

    await callback.answer()
    for reply in replies:
        fallback_markup = (
            main_reply_keyboard("Надішли правильний текст голосового")
            if action == "fix"
            else None
        )
        await callback.message.answer(
            reply.text,
            reply_markup=_inline_reply_keyboard(reply.keyboard) or fallback_markup,
        )
    if should_embed and entry_id and user_id:
        asyncio.create_task(_embed_entry_task(settings, sessionmaker, memory_service, entry_id, user_id))


@router.callback_query(F.data.startswith("period:"))
async def period_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    summary_service: SummaryService,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    period = callback.data.split(":", maxsplit=1)[1]
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        async with _typing(callback.message):
            if period == "month":
                summary = await summary_service.generate_current_month_summary(session, user=user)
            else:
                summary = await summary_service.generate_current_week_summary(session, user=user)
        text = format_period_summary(summary)
        summary_id = str(summary.id)
    await callback.answer()
    await _answer_long_text(callback.message, text, reply_markup=period_detail_keyboard(summary_id=summary_id))


@router.callback_query(F.data.startswith("periodview:"))
async def period_detail_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    parsed = _parse_scoped_callback(callback.data or "", prefix="periodview")
    if parsed is None:
        await callback.answer("Не можу відкрити цей період")
        return
    summary_id, section = parsed
    chart = None
    reply_markup = None
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        summary = await repo.get_summary(session, summary_id=summary_id)
        if summary is None or summary.user_id != user.id or summary.period_type not in {"weekly", "monthly"}:
            text = "Не знайшов цей підсумок періоду."
        else:
            reply_markup = period_detail_keyboard(summary_id=str(summary.id))
            if section == "timeline":
                text = await format_period_timeline_view(session, user=user, summary=summary)
            elif section == "metrics":
                text = await format_period_metrics_view(session, user=user, summary=summary)
            elif section == "chart":
                text = "Графік по днях: синя лінія — настрій, зелена — енергія."
                chart = await build_period_metrics_chart_png(session, user=user, summary=summary)
                if chart is None:
                    text = "Для графіка за цей період поки замало метрик."
            elif section == "days":
                text = await format_period_days_view(session, user=user, summary=summary)
            else:
                text = format_period_summary(summary)
    await callback.answer()
    await _answer_long_text(callback.message, text, reply_markup=reply_markup)
    if chart is not None:
        await callback.message.answer_photo(
            BufferedInputFile(chart, filename="period-metrics.png"),
            caption="Графік по днях: синя лінія — настрій, зелена — енергія.",
        )


@router.message((F.text & ~F.text.startswith("/")) | F.caption | F.photo | F.voice)
async def entry_handler(
    message: Message,
    bot: Bot,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    interaction_service: InteractionService,
    memory_service: MemoryService,
    summary_service: SummaryService,
) -> None:
    if not await _allowed(message, settings):
        return
    text = message.text or message.caption or "[photo]"
    entry_id: UUID | None = None
    should_embed = False
    replies = []
    user_id = None
    voice_note: VoiceNoteTranscription | None = None
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        user_id = user.id
        if message.voice:
            async with _typing(message, bot):
                voice_note = await _transcribe_voice_note(
                    bot=bot,
                    settings=settings,
                    session=session,
                    message=message,
                    user_id=user.id,
                    ai_service=interaction_service.ai,
                )
            if voice_note.text:
                text = voice_note.text
                await repo.update_user_settings(
                    session,
                    user_id=user.id,
                    values={
                        "settings_json": settings_json_with_pending_voice_transcript(
                            await repo.get_user_settings(session, user.id),
                            _pending_voice_note_payload(
                                voice_note,
                                telegram_message_id=message.message_id,
                                reply_to_message_id=(
                                    message.reply_to_message.message_id
                                    if message.reply_to_message
                                    else None
                                ),
                            ),
                        )
                    },
                )
                replies = [
                    BotReply(
                        "Перевір транскрипцію голосового:\n"
                        f"{_voice_transcription_preview(text)}\n\n"
                        "Якщо все правильно, підтвердь. Якщо ні — натисни «Виправити текст» або просто надішли правильний текст наступним повідомленням.",
                        keyboard="voice_transcript",
                    )
                ]
            else:
                hint = (
                    "Не зміг розшифрувати голосове повідомлення. "
                    "Можна повторити голосом або написати цей момент текстом."
                )
                if voice_note.error:
                    hint += f"\n\nТехнічно: {voice_note.error}"
                replies = [BotReply(hint)]
        user_settings = await repo.get_user_settings(session, user.id)
        pending_kind = pending_input(user_settings)
        if replies:
            pass
        elif pending_kind and not message.photo:
            async with _typing(message, bot):
                result = await _handle_pending_input(
                    session=session,
                    user=user,
                    user_settings=user_settings,
                    pending_kind=pending_kind,
                    text=text,
                    message=message,
                    interaction_service=interaction_service,
                )
            entry_id = result.entry_id
            should_embed = result.should_embed_entry
            replies = result.replies
        elif not message.photo and _is_sleep_marker_text(text):
            replies = [BotReply("Закрити день і згенерувати підсумок?", keyboard="sleep_confirm")]
        elif not message.photo and (missed_reason := _missed_reason_text(text)):
            async with _typing(message, bot):
                result = await interaction_service.record_missed_reason(
                    session,
                    user=user,
                    reason_text=missed_reason,
                    reason_code="custom",
                )
            entry_id = result.entry_id
            should_embed = result.should_embed_entry
            replies = result.replies
        else:
            async with _typing(message, bot):
                result = await interaction_service.handle_text_entry(
                    session,
                    user=user,
                    text=text,
                    telegram_message_id=message.message_id,
                    reply_to_message_id=message.reply_to_message.message_id if message.reply_to_message else None,
                )
            entry_id = result.entry_id
            should_embed = result.should_embed_entry
            replies = result.replies
            if message.photo and entry_id is not None:
                await _store_photo(bot, settings, session, message, user.id, entry_id)

        if voice_note and entry_id is not None:
            await _store_voice_note(session, user_id=user.id, entry_id=entry_id, voice_note=voice_note)

    for reply in replies:
        await message.answer(
            reply.text,
            reply_markup=_inline_reply_keyboard(reply.keyboard),
        )
    if should_embed and entry_id and user_id:
        asyncio.create_task(_embed_entry_task(settings, sessionmaker, memory_service, entry_id, user_id))


@router.callback_query(F.data.startswith("missed_reason:"))
async def missed_reason_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    reason_code = callback.data.split(":", maxsplit=1)[1]
    if reason_code == "custom":
        await callback.answer()
        await _clear_inline_keyboard(callback)
        await callback.message.answer(
            "Можеш коротко написати причину у форматі: причина: ...",
            reply_markup=main_reply_keyboard("причина: що завадило відповісти"),
        )
        return
    await callback.answer("Не впізнав дію", show_alert=True)


@router.callback_query(F.data.startswith("snapshot:"))
async def snapshot_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    interaction_service: InteractionService,
    memory_service: MemoryService,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    action = callback.data.split(":", maxsplit=1)[1]
    entry_id = None
    user_id = None
    replies = []
    should_embed = False
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_id = user.id
        result = await interaction_service.record_button_action(session, user=user, action=action)
        entry_id = result.entry_id
        should_embed = result.should_embed_entry
        replies = result.replies
    await callback.answer()
    await _clear_inline_keyboard(callback)
    for reply in replies:
        await callback.message.answer(
            reply.text,
            reply_markup=_inline_reply_keyboard(reply.keyboard),
        )
    if should_embed and entry_id and user_id:
        asyncio.create_task(_embed_entry_task(settings, sessionmaker, memory_service, entry_id, user_id))


@router.callback_query((F.data == "sleep:confirm") | (F.data == "day:sleep"))
async def sleep_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    summary_service: SummaryService,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    await _clear_inline_keyboard(callback)
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        async with _typing(callback.message):
            summary = await summary_service.close_today_with_summary(session, user=user)
    await callback.answer("День закрито")
    await callback.message.answer(summary.short_text, reply_markup=summary_detail_keyboard(summary_id=str(summary.id)))


@router.callback_query(F.data == "sleep:cancel")
async def sleep_cancel_callback_handler(callback: CallbackQuery, settings: Settings) -> None:
    if not await _allowed_callback(callback, settings):
        return
    await callback.answer("Скасовано")
    await _clear_inline_keyboard(callback)
    await callback.message.answer("Ок, день не закриваю.")


@router.callback_query(F.data == "day:summary")
async def day_summary_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    summary_service: SummaryService,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        async with _typing(callback.message):
            summary = await summary_service.generate_today_summary(session, user=user)
    await callback.answer()
    await callback.message.answer(summary.short_text, reply_markup=summary_detail_keyboard(summary_id=str(summary.id)))


@router.callback_query(F.data.startswith("archive:export"))
async def export_callback_handler(callback: CallbackQuery, settings: Settings) -> None:
    if not await _allowed_callback(callback, settings):
        return
    export_format, extension, prefix, caption = _archive_export_options(callback.data or "")
    output = Path("./data") / f"{prefix}-{callback.from_user.id}.{extension}"
    await export_user_archive(settings, callback.from_user.id, output, format=export_format)
    await callback.answer("Експорт готовий")
    await callback.message.answer_document(FSInputFile(output), caption=caption)


async def _handle_pending_input(
    *,
    session: AsyncSession,
    user,
    user_settings: UserSettings,
    pending_kind: str,
    text: str,
    message: Message,
    interaction_service: InteractionService,
) -> InteractionResult:
    if pending_kind == "custom_style":
        next_settings_json = settings_json_with_custom_interaction_style(user_settings, text)
        user_settings = await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": _without_pending(next_settings_json)},
        )
        return InteractionResult(
            replies=[
                BotReply("Записав власний стиль взаємодії."),
                BotReply(_format_settings_text(user_settings=user_settings, snapshots_are_paused=snapshots_paused(user_settings))),
            ],
            snapshot_closed=True,
        )
    if pending_kind == "profile_context":
        next_settings_json = settings_json_with_user_profile_context(user_settings, text)
        user_settings = await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": _without_pending(next_settings_json)},
        )
        return InteractionResult(
            replies=[
                BotReply("Записав загальний контекст про тебе."),
                BotReply(_format_settings_text(user_settings=user_settings, snapshots_are_paused=snapshots_paused(user_settings))),
            ],
            snapshot_closed=True,
        )
    if pending_kind == "correction":
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_without_pending_input(user_settings)},
        )
        return await interaction_service.record_correction(
            session,
            user=user,
            correction_text=text,
            telegram_message_id=message.message_id,
            reply_to_message_id=message.reply_to_message.message_id if message.reply_to_message else None,
        )
    if pending_kind in {"voice_transcript", "voice_transcript_fix"}:
        return await _confirm_pending_voice_transcript(
            session=session,
            user=user,
            user_settings=user_settings,
            interaction_service=interaction_service,
            corrected_text=text,
            fallback_message_id=message.message_id,
            fallback_reply_to_message_id=(
                message.reply_to_message.message_id if message.reply_to_message else None
            ),
        )
    await repo.update_user_settings(
        session,
        user_id=user.id,
        values={"settings_json": settings_json_without_pending_input(user_settings)},
    )
    return InteractionResult(replies=[BotReply("Скинув незавершене очікування вводу.")], snapshot_closed=True)


async def _confirm_pending_voice_transcript(
    *,
    session: AsyncSession,
    user,
    user_settings: UserSettings,
    interaction_service: InteractionService,
    corrected_text: str | None = None,
    fallback_message_id: int | None = None,
    fallback_reply_to_message_id: int | None = None,
) -> InteractionResult:
    pending = pending_voice_transcript(user_settings)
    if not pending:
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_without_pending_voice_transcript(user_settings)},
        )
        return InteractionResult(
            replies=[BotReply("Не бачу голосового, яке чекає підтвердження.")],
            snapshot_closed=True,
        )

    text = " ".join((corrected_text or pending.get("text") or "").split())
    if not text:
        return InteractionResult(
            replies=[BotReply("Транскрипція порожня. Надішли правильний текст або скасуй голосове.")],
            snapshot_closed=True,
        )

    result = await interaction_service.handle_text_entry(
        session,
        user=user,
        text=text,
        telegram_message_id=_pending_int(pending.get("telegram_message_id")) or fallback_message_id,
        reply_to_message_id=(
            _pending_int(pending.get("reply_to_message_id")) or fallback_reply_to_message_id
        ),
    )
    if result.entry_id is not None:
        await _store_voice_note(
            session,
            user_id=user.id,
            entry_id=result.entry_id,
            voice_note=_voice_note_from_pending(pending, text=text),
        )
    await repo.update_user_settings(
        session,
        user_id=user.id,
        values={"settings_json": settings_json_without_pending_voice_transcript(user_settings)},
    )
    return result


async def _store_photo(
    bot: Bot,
    settings: Settings,
    session: AsyncSession,
    message: Message,
    user_id: UUID,
    entry_id: UUID,
) -> None:
    photo = message.photo[-1]
    user_dir = settings.media_root / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / f"{message.message_id}-{photo.file_unique_id}.jpg"
    try:
        await bot.download(photo.file_id, destination=path)
    except Exception:
        logger.exception("Failed to download Telegram photo")
        path = None
    await repo.add_media(
        session,
        user_id=user_id,
        entry_id=entry_id,
        media_type="photo",
        telegram_file_id=photo.file_id,
        telegram_file_unique_id=photo.file_unique_id,
        file_path=str(path) if path else None,
        meta={"width": photo.width, "height": photo.height},
    )


async def _transcribe_voice_note(
    *,
    bot: Bot,
    settings: Settings,
    session: AsyncSession,
    message: Message,
    user_id: UUID,
    ai_service: AIService,
) -> VoiceNoteTranscription:
    voice = message.voice
    if voice is None:
        raise ValueError("Message has no Telegram voice payload")

    user_dir = settings.media_root / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    original_path = user_dir / f"{message.message_id}-{voice.file_unique_id}.ogg"
    transcription_path: Path | None = None
    try:
        await bot.download(voice.file_id, destination=original_path)
        transcription_path = await _convert_voice_note_for_transcription(original_path)
        text, run_id = await ai_service.transcribe_voice(
            session,
            user_id=user_id,
            file_path=str(transcription_path),
            duration_seconds=voice.duration,
        )
        return VoiceNoteTranscription(
            text=text,
            original_text=text,
            original_path=original_path,
            transcription_path=transcription_path,
            transcription_run_id=run_id,
            duration_seconds=voice.duration,
            mime_type=voice.mime_type,
            file_size=voice.file_size,
            telegram_file_id=voice.file_id,
            telegram_file_unique_id=voice.file_unique_id,
        )
    except FileNotFoundError:
        logger.exception("ffmpeg is not available for Telegram voice transcription")
        return _failed_voice_note(
            voice,
            original_path=original_path if original_path.exists() else None,
            transcription_path=transcription_path,
            error="на сервері немає ffmpeg для підготовки Telegram voice",
        )
    except Exception as exc:
        logger.exception("Telegram voice transcription failed")
        return _failed_voice_note(
            voice,
            original_path=original_path if original_path.exists() else None,
            transcription_path=transcription_path,
            error=str(exc),
        )


async def _convert_voice_note_for_transcription(original_path: Path) -> Path:
    transcription_path = original_path.with_suffix(".webm")
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i",
        str(original_path),
        "-vn",
        "-c:a",
        "libopus",
        str(transcription_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or "ffmpeg не зміг підготувати Telegram voice")
    return transcription_path


async def _store_voice_note(
    session: AsyncSession,
    *,
    user_id: UUID,
    entry_id: UUID,
    voice_note: VoiceNoteTranscription,
) -> None:
    await repo.add_media(
        session,
        user_id=user_id,
        entry_id=entry_id,
        media_type="voice",
        telegram_file_id=voice_note.telegram_file_id,
        telegram_file_unique_id=voice_note.telegram_file_unique_id,
        file_path=str(voice_note.original_path) if voice_note.original_path else None,
        meta={
            "duration_seconds": voice_note.duration_seconds,
            "mime_type": voice_note.mime_type,
            "file_size": voice_note.file_size,
            "transcription_file_path": (
                str(voice_note.transcription_path) if voice_note.transcription_path else None
            ),
            "transcription_run_id": (
                str(voice_note.transcription_run_id) if voice_note.transcription_run_id else None
            ),
            "transcription_text": voice_note.text,
            "original_transcription_text": voice_note.original_text,
            "transcription_error": voice_note.error,
        },
    )


def _pending_voice_note_payload(
    voice_note: VoiceNoteTranscription,
    *,
    telegram_message_id: int | None,
    reply_to_message_id: int | None,
) -> dict[str, object]:
    return {
        "text": voice_note.text,
        "original_text": voice_note.original_text or voice_note.text,
        "original_path": str(voice_note.original_path) if voice_note.original_path else None,
        "transcription_path": (
            str(voice_note.transcription_path) if voice_note.transcription_path else None
        ),
        "transcription_run_id": (
            str(voice_note.transcription_run_id) if voice_note.transcription_run_id else None
        ),
        "duration_seconds": voice_note.duration_seconds,
        "mime_type": voice_note.mime_type,
        "file_size": voice_note.file_size,
        "telegram_file_id": voice_note.telegram_file_id,
        "telegram_file_unique_id": voice_note.telegram_file_unique_id,
        "telegram_message_id": telegram_message_id,
        "reply_to_message_id": reply_to_message_id,
    }


def _voice_note_from_pending(pending: dict[str, object], *, text: str) -> VoiceNoteTranscription:
    return VoiceNoteTranscription(
        text=text,
        original_text=str(pending.get("original_text") or pending.get("text") or ""),
        original_path=_optional_path(pending.get("original_path")),
        transcription_path=_optional_path(pending.get("transcription_path")),
        transcription_run_id=_optional_uuid(pending.get("transcription_run_id")),
        duration_seconds=_pending_int(pending.get("duration_seconds")),
        mime_type=_optional_str(pending.get("mime_type")),
        file_size=_pending_int(pending.get("file_size")),
        telegram_file_id=str(pending.get("telegram_file_id") or ""),
        telegram_file_unique_id=str(pending.get("telegram_file_unique_id") or ""),
    )


def _optional_path(value: object) -> Path | None:
    return Path(str(value)) if value else None


def _optional_uuid(value: object) -> UUID | None:
    if not value:
        return None
    try:
        return UUID(str(value))
    except ValueError:
        return None


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _pending_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _failed_voice_note(
    voice,
    *,
    original_path: Path | None,
    transcription_path: Path | None,
    error: str,
) -> VoiceNoteTranscription:
    return VoiceNoteTranscription(
        text="",
        original_text=None,
        original_path=original_path,
        transcription_path=transcription_path,
        transcription_run_id=None,
        duration_seconds=voice.duration,
        mime_type=voice.mime_type,
        file_size=voice.file_size,
        telegram_file_id=voice.file_id,
        telegram_file_unique_id=voice.file_unique_id,
        error=error,
    )


def _voice_transcription_preview(text: str, limit: int = 900) -> str:
    compact = " ".join((text or "").split())
    if len(compact) > limit:
        compact = compact[: limit - 1].rstrip() + "…"
    return f"«{compact}»"


async def _send_photo_moments(message: Message, moments: list[PhotoMoment], *, limit: int = 12) -> None:
    visible = moments[-limit:]
    sendable: list[FSInputFile | str] = []
    for moment in visible:
        media = moment.media
        if media.file_path and Path(media.file_path).exists():
            sendable.append(FSInputFile(media.file_path))
        elif media.telegram_file_id:
            sendable.append(media.telegram_file_id)

    if not sendable:
        if moments:
            await message.answer("Фото є в архіві, але зараз не бачу файлів для відправки.")
        return

    if len(sendable) == 1:
        await message.answer_photo(sendable[0], caption="Фото дня")
        return

    for start in range(0, len(sendable), 10):
        batch = sendable[start : start + 10]
        await message.answer_media_group([InputMediaPhoto(media=item) for item in batch])


async def _embed_entry_task(
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    memory_service: MemoryService,
    entry_id: UUID,
    user_id: UUID,
) -> None:
    try:
        async with sessionmaker() as session, session.begin():
            entry = await repo.get_entry(session, entry_id=entry_id)
            if entry is None:
                return
            await memory_service.embed_entry(session, entry=entry, user_id=user_id)
    except Exception:
        logger.exception("Background embedding task failed", extra={"entry_id": str(entry_id)})


async def _get_or_create_message_user(session: AsyncSession, message: Message, settings: Settings):
    return await repo.get_or_create_user(
        session,
        telegram_user_id=message.from_user.id,
        chat_id=message.chat.id,
        username=message.from_user.username,
        first_name=message.from_user.first_name,
        timezone=settings.app_timezone,
    )


async def _get_or_create_callback_user(session: AsyncSession, callback: CallbackQuery, settings: Settings):
    return await repo.get_or_create_user(
        session,
        telegram_user_id=callback.from_user.id,
        chat_id=callback.message.chat.id,
        username=callback.from_user.username,
        first_name=callback.from_user.first_name,
        timezone=settings.app_timezone,
    )


async def _allowed(message: Message, settings: Settings) -> bool:
    if message.from_user is None:
        return False
    if settings.telegram_allowed_user_ids and message.from_user.id not in settings.telegram_allowed_user_ids:
        await message.answer("Цей бот налаштований як персональний і не приймає записи від цього акаунта.")
        return False
    return True


async def _allowed_callback(callback: CallbackQuery, settings: Settings) -> bool:
    if settings.telegram_allowed_user_ids and callback.from_user.id not in settings.telegram_allowed_user_ids:
        await callback.answer("Немає доступу", show_alert=True)
        return False
    return True


@asynccontextmanager
async def _typing(message: Message | None, bot: Bot | None = None) -> AsyncIterator[None]:
    if message is None:
        yield
        return
    action_bot = bot or getattr(message, "bot", None)
    if action_bot is None:
        yield
        return
    sender = ChatActionSender.typing(chat_id=message.chat.id, bot=action_bot)
    try:
        await sender.__aenter__()
    except Exception:
        logger.debug("Could not send typing chat action", exc_info=True)
        yield
        return
    try:
        yield
    finally:
        try:
            await sender.__aexit__(None, None, None)
        except Exception:
            logger.debug("Could not stop typing chat action", exc_info=True)


async def _clear_inline_keyboard(callback: CallbackQuery) -> None:
    if callback.message is None:
        return
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:
        logger.debug("Could not clear inline keyboard", exc_info=True)


def _inline_reply_keyboard(kind: str | None):
    if kind == "snapshot_control":
        return snapshot_clarification_keyboard()
    if kind == "correction":
        return correction_keyboard()
    if kind == "voice_transcript":
        return voice_transcription_keyboard()
    if kind == "sleep_confirm":
        return sleep_confirmation_keyboard()
    return None


def _without_pending(settings_json: dict) -> dict:
    updated = dict(settings_json)
    updated.pop("pending_input", None)
    return updated


def _valid_hhmm(value: str) -> bool:
    parts = value.split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return False
    hour = int(parts[0])
    minute = int(parts[1])
    return 0 <= hour <= 23 and 0 <= minute <= 59


def _command_argument(text: str) -> str:
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def _custom_style_text(text: str) -> str:
    return _prefixed_text(text, limit=800)


def _prefixed_text(text: str, *, limit: int = 2000) -> str:
    _, _, value = text.partition(":")
    return " ".join(value.split())[:limit]


def _truncate_setting(value: str, limit: int = 180) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _is_sleep_marker_text(text: str) -> bool:
    normalized = " ".join(text.lower().split()).strip(" \t\r\n.,!?…")
    return normalized == SLEEP_MARKER_TEXT


async def _format_day_detail_section(
    session: AsyncSession, *, user: User, day: Day, section: str
) -> tuple[str, bytes | None, list[PhotoMoment]]:
    chart = None
    moments: list[PhotoMoment] = []
    day_title = day.local_date.isoformat()
    if section == "raw":
        text = await format_raw_entries_for_day(session, user=user, day=day, title=day_title)
    elif section == "timeline":
        text = await format_day_view(session, user=user, day=day, limit=30)
    elif section == "metrics":
        text = await format_metrics_for_day(session, user=user, day=day, title=day_title)
        chart = await build_metrics_chart_png_for_day(session, user=user, day=day)
    elif section == "photos":
        moments = await get_photo_moments_for_day(session, day=day)
        text = format_photo_moments_view(
            moments,
            timezone=user.timezone,
            title=f"Фото за {day_title}",
        )
    elif section == "gaps":
        text = await format_gaps_for_day(
            session,
            user=user,
            day=day,
            target_date=day.local_date,
            title=f"Прогалини за {day_title}",
        )
    else:
        text = await format_day_summary_section(session, user=user, day=day, section=section)
    return text, chart, moments


def _parse_day_query(query: str, timezone: str) -> date | None:
    normalized = " ".join(query.strip().lower().split())
    if not normalized:
        return None
    if normalized in {"сьогодні", "today"}:
        return datetime.now(tz=zoneinfo(timezone)).date()
    if normalized in {"вчора", "yesterday"}:
        return datetime.now(tz=zoneinfo(timezone)).date() - timedelta(days=1)
    try:
        return date.fromisoformat(normalized)
    except ValueError:
        pass
    try:
        return datetime.strptime(normalized, "%d.%m.%Y").date()
    except ValueError:
        return None


def _is_previous_period_query(query: str) -> bool:
    normalized = " ".join(query.strip().lower().split())
    return normalized in {"prev", "previous", "попередній", "попередня", "минулий", "минула"}


def _parse_scoped_callback(data: str, *, prefix: str) -> tuple[UUID, str] | None:
    parts = data.split(":")
    if len(parts) != 3 or parts[0] != prefix:
        return None
    item_id = _uuid_or_none(parts[1])
    if item_id is None or not parts[2]:
        return None
    return item_id, parts[2]


def _uuid_or_none(value: str | UUID | None) -> UUID | None:
    if isinstance(value, UUID):
        return value
    if value is None:
        return None
    try:
        return UUID(str(value))
    except ValueError:
        return None


def _missed_reason_text(text: str) -> str | None:
    normalized = text.strip()
    lowered = normalized.lower()
    for prefix in ("причина:", "reason:"):
        if lowered.startswith(prefix):
            reason = normalized[len(prefix) :].strip()
            return reason or None
    return None


def _help_text() -> str:
    return "\n".join(
        [
            "Команди:",
            "/snapshot - зробити зріз зараз",
            "/today - таймлайн сьогодні",
            "/day 2026-06-30 - переглянути будь-який день",
            "/metrics - метрики й міні-графіки",
            "/gaps - прогалини й покриття дня",
            "/raw - сирі записи за день",
            "/report - таймлайн + метрики",
            "/summary - згенерувати підсумок дня",
            "/sleep - закрити день і згенерувати підсумок",
            "/week - підсумок поточного тижня",
            "/week prev - підсумок попереднього тижня",
            "/month - підсумок поточного місяця",
            "/month prev - підсумок попереднього місяця",
            "/similar <текст> - знайти схожі моменти",
            "/costs - витрати й токени",
            "/audit - стан архіву й покриття даних",
            "/settings - налаштування",
            "Можна надсилати текст, фото або Telegram-голосові повідомлення.",
            "стиль: ... - власний стиль питань і коротких відповідей",
            "скинути стиль - прибрати власний стиль",
            "/pause - поставити автоматичні зрізи на паузу",
            "/resume - знову увімкнути автоматичні зрізи",
            "/export - експорт JSON",
            "/export_md - експорт Markdown",
            "/export_csv - експорт метрик CSV",
            "/export_zip - ZIP архів з даними й media",
        ]
    )


def _format_settings_text(*, user_settings: UserSettings, snapshots_are_paused: bool) -> str:
    status = "на паузі" if snapshots_are_paused else "увімкнені"
    body = "так" if user_settings.ask_body_signals else "ні"
    photo = "так" if user_settings.photo_prompts_enabled else "ні"
    custom_style = custom_interaction_style(user_settings)
    profile_context = user_profile_context(user_settings)
    raw_tone = getattr(user_settings, "tone", "calm")
    raw_humanity = getattr(user_settings, "humanity_level", "balanced")
    tone = {"precise": "сухий", "calm": "спокійний"}.get(raw_tone, raw_tone)
    humanity = {"balanced": "стримано", "warm": "людяніше"}.get(
        raw_humanity,
        raw_humanity,
    )
    return "\n".join(
        [
            "Поточні налаштування:",
            f"Автоматичні зрізи: {status}",
            f"Тон: {tone}",
            f"Стиль: {humanity}",
            f"Власний стиль: {custom_style or 'немає'}",
            f"Контекст про мене: {_truncate_setting(profile_context) if profile_context else 'немає'}",
            f"Активні години: {user_settings.active_start}-{user_settings.active_end}",
            f"Інтервал зрізів: {user_settings.min_interval_minutes}-{user_settings.max_interval_minutes} хв",
            f"Нагадування після: {user_settings.reminder_delay_minutes} хв",
            f"Макс. уточнень: {user_settings.max_clarifications}",
            f"Питати про тіло: {body}",
            f"Фото-підказки: {photo}",
            "",
            "Точні команди:",
            "/set_active 09:00 23:30",
            "/set_frequency 30 70",
            "/set_reminder 25",
            "стиль: коротко, без підбадьорювання, але не сухо",
            "скинути стиль",
            "контекст: хто я, чим займаюся, що важливо знати",
            "скинути контекст",
        ]
    )


def _frequency_preset_values(preset: str) -> dict[str, int]:
    presets = {
        "slow": {"min_interval_minutes": 75, "max_interval_minutes": 120},
        "normal": {"min_interval_minutes": 30, "max_interval_minutes": 70},
        "fast": {"min_interval_minutes": 20, "max_interval_minutes": 40},
    }
    return presets.get(preset, presets["normal"])


def _archive_export_options(callback_data: str) -> tuple[str, str, str, str]:
    if callback_data.endswith("_md"):
        return "markdown", "md", "export", "Markdown-архів готовий."
    if callback_data.endswith("_csv"):
        return "csv", "csv", "metrics", "CSV з метриками готовий."
    if callback_data.endswith("_zip"):
        return "zip", "zip", "archive", "ZIP-архів з даними готовий."
    return "json", "json", "export", "JSON-архів готовий."


async def _answer_long_text(message: Message, text: str, reply_markup=None) -> None:
    chunks = _split_telegram_text(text)
    for index, chunk in enumerate(chunks):
        await message.answer(chunk, reply_markup=reply_markup if index == len(chunks) - 1 else None)


def _split_telegram_text(text: str, limit: int = 3900) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines():
        line_len = len(line) + 1
        if current and current_len + line_len > limit:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        if line_len > limit:
            for start in range(0, len(line), limit):
                if current:
                    chunks.append("\n".join(current))
                    current = []
                    current_len = 0
                chunks.append(line[start : start + limit])
            continue
        current.append(line)
        current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks or [""]
