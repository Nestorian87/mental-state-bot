from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import UUID

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, CallbackQuery, FSInputFile, InputMediaPhoto, Message
from aiogram.utils.chat_action import ChatActionSender
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mental_state_bot.ai.service import AIService
from mental_state_bot.bot.keyboards import (
    EMOTION_CALIBRATION_OPTIONS,
    clarifications_menu_keyboard,
    clarifications_skip_all_confirmation_keyboard,
    correction_keyboard,
    data_menu_keyboard,
    day_detail_keyboard,
    day_menu_keyboard,
    deferred_clarification_keyboard,
    emotion_calibration_keyboard,
    emotion_intensity_keyboard,
    entry_delete_confirmation_keyboard,
    entry_management_keyboard,
    interpretation_keyboard,
    life_context_continue_keyboard,
    life_context_current_question_keyboard,
    life_context_menu_keyboard,
    life_context_offer_keyboard,
    life_context_open_question_keyboard,
    life_context_question_keyboard,
    life_context_rewrite_confirmation_keyboard,
    main_menu_keyboard,
    main_reply_keyboard,
    manual_entry_confirmation_keyboard,
    memory_ai_review_confirmation_keyboard,
    memory_graph_import_confirmation_keyboard,
    memory_maintenance_confirmation_keyboard,
    memory_menu_keyboard,
    memory_rebuild_confirmation_keyboard,
    metric_score_keyboard,
    period_choice_keyboard,
    period_detail_keyboard,
    planned_event_offer_keyboard,
    quiet_menu_keyboard,
    quiet_offer_keyboard,
    reanalysis_confirmation_keyboard,
    reanalysis_scope_keyboard,
    settings_capture_keyboard,
    settings_keyboard,
    settings_rhythm_keyboard,
    settings_style_keyboard,
    sleep_confirmation_keyboard,
    sleep_reflection_keyboard,
    snapshot_clarification_keyboard,
    summaries_menu_keyboard,
    summary_detail_keyboard,
    turning_point_detail_keyboard,
    turning_points_keyboard,
    voice_transcription_keyboard,
    wake_time_keyboard,
)
from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import Day, Entry, User, UserSettings
from mental_state_bot.emotions import EMOTION_INTENSITY_LEVELS
from mental_state_bot.services.analysis_backfill import (
    count_entry_feature_reanalysis,
    guided_reanalyze_entry_features,
)
from mental_state_bot.services.archive_audit import build_archive_audit, format_archive_audit
from mental_state_bot.services.exports import export_user_archive
from mental_state_bot.services.interactions import (
    BotReply,
    InteractionResult,
    InteractionService,
    apply_emotion_calibration,
    apply_metric_calibration,
)
from mental_state_bot.services.journal_day import current_journal_date
from mental_state_bot.services.life_context import (
    answer_life_context_candidate,
    apply_life_context_rewrite,
    cancel_life_context_rewrite,
    current_life_context_candidate,
    format_life_context_items,
    format_life_context_question,
    maybe_start_auto_life_context_review,
    start_life_context_review,
    start_life_context_rewrite,
)
from mental_state_bot.services.memory import MemoryService, backfill_entry_embeddings
from mental_state_bot.services.memory_graph import (
    MemoryGraphAIReviewResult,
    MemoryGraphMaintenanceResult,
    format_personal_lexicon_view,
    maintain_memory_graph,
    review_memory_graph_duplicates,
)
from mental_state_bot.services.memory_graph_exchange import (
    export_memory_graph_json,
    parse_memory_graph_import,
    replace_memory_graph_from_import,
)
from mental_state_bot.services.memory_graph_visualization import build_memory_graph_html
from mental_state_bot.services.planned_events import (
    confirm_pending_planned_event,
    detect_planned_event_candidate,
    planned_event_context,
    planned_event_text,
)
from mental_state_bot.services.preferences import (
    adaptive_observation_enabled,
    clarification_queue,
    context_quiet_enabled,
    context_quiet_last_check_at,
    custom_interaction_style,
    life_context_items,
    pending_clarification,
    pending_correction_entry_id,
    pending_input,
    pending_life_context_review,
    pending_manual_entry,
    pending_memory_graph_import,
    pending_planned_event,
    pending_post_entry_followup,
    pending_voice_transcript,
    quiet_is_active,
    quiet_until,
    settings_json_with_adaptive_observation,
    settings_json_with_clarification_queue,
    settings_json_with_context_quiet,
    settings_json_with_context_quiet_last_check,
    settings_json_with_context_quiet_last_offer,
    settings_json_with_custom_interaction_style,
    settings_json_with_pending_clarification,
    settings_json_with_pending_input,
    settings_json_with_pending_life_context_review,
    settings_json_with_pending_manual_entry,
    settings_json_with_pending_memory_graph_import,
    settings_json_with_pending_planned_event,
    settings_json_with_pending_post_entry_followup,
    settings_json_with_pending_voice_transcript,
    settings_json_with_planned_events,
    settings_json_with_quiet_until,
    settings_json_with_snapshot_pause,
    settings_json_with_user_profile_context,
    settings_json_with_wake_time_records,
    settings_json_without_pending_input,
    settings_json_without_pending_manual_entry,
    settings_json_without_pending_voice_transcript,
    snapshots_paused,
    user_profile_context,
)
from mental_state_bot.services.review import (
    PhotoMoment,
    build_affect_spectrum_png,
    build_affect_spectrum_png_for_day,
    build_emotion_timeline_png,
    build_emotion_timeline_png_for_day,
    build_metrics_chart_png,
    build_metrics_chart_png_for_day,
    build_period_emotion_chart_png,
    build_period_metrics_chart_png,
    format_affective_vocabulary_audit,
    format_cost_report,
    format_day_summary_section,
    format_day_turning_point,
    format_day_turning_points,
    format_day_view,
    format_gaps_for_day,
    format_gaps_view,
    format_latest_summary_section,
    format_metrics_for_day,
    format_metrics_view,
    format_period_days_view,
    format_period_emotions_view,
    format_period_metrics_view,
    format_period_patterns_view,
    format_period_summary,
    format_period_timeline_view,
    format_period_turning_points_view,
    format_photo_moments_view,
    format_raw_entries_for_day,
    format_raw_entries_view,
    format_similar_entries,
    format_summary_section,
    format_today_view,
    get_day_turning_points,
    get_photo_moments_for_day,
    get_today_photo_moments,
)
from mental_state_bot.services.snapshots import send_snapshot_prompt
from mental_state_bot.services.summaries import SummaryService
from mental_state_bot.services.visual_report import (
    MAX_VISUAL_REPORT_DAYS,
    build_visual_report_pdf,
)
from mental_state_bot.services.wake_time import (
    append_wake_time_record,
    parse_wake_time_text,
    should_offer_wake_time_question,
    skipped_wake_time_record,
)
from mental_state_bot.time_utils import utc_now, zoneinfo

logger = logging.getLogger(__name__)

FEATURE_REANALYSIS_LIMIT = 200
MEMORY_REBUILD_LIMIT = 200
MEMORY_GRAPH_REVIEW_LIMIT = 12
VOICE_TRANSCRIPTION_ACTIONS = {"confirm", "fix", "cancel"}
MANUAL_ENTRY_ACTIONS = {"save", "ignore"}
router = Router()

SLEEP_MARKER_TEXT = "лягаю спати"
_DAY_DETAIL_SECTIONS = {"timeline", "metrics", "photos", "raw", "gaps", "turning_points"}


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


@router.message(F.text == "Меню")
async def menu_button_handler(message: Message, settings: Settings) -> None:
    if not await _allowed(message, settings):
        return
    await message.answer("Головне меню.", reply_markup=main_menu_keyboard())


@router.message(F.text == "Пауза")
async def quiet_button_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        text = _quiet_status_text(user_settings, timezone=user.timezone)
        active = quiet_is_active(user_settings)
    await message.answer(text, reply_markup=quiet_menu_keyboard(active=active))


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
        user_settings = await repo.get_user_settings(session, user.id)
        today = await current_journal_date(session, user=user, user_settings=user_settings)
        target_date = _parse_day_query(query, user.timezone, today=today)
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
        emotion_chart = await build_emotion_timeline_png(session, user=user)
        spectrum_chart = await build_affect_spectrum_png(session, user=user)
    await _answer_long_text(message, text)
    if chart is not None:
        await message.answer_photo(
            BufferedInputFile(chart, filename="metrics.png"),
        )
    if emotion_chart is not None:
        await message.answer_photo(
            BufferedInputFile(emotion_chart, filename="emotions.png"),
            caption="Емоційна карта дня.",
        )
    if spectrum_chart is not None:
        await message.answer_photo(
            BufferedInputFile(spectrum_chart, filename="spectrum.png"),
            caption="Спектр стану дня.",
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
    await _mark_summary_delivered(sessionmaker, summary_id=summary.id)


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
    await _mark_summary_delivered(sessionmaker, summary_id=summary.id)


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


@router.message(Command("visual_report"))
async def visual_report_command_handler(
    message: Message,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    query = _command_argument(message.text or "")
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        today = await current_journal_date(session, user=user, user_settings=user_settings)
        parsed = _parse_date_range_query(query, user.timezone, today=today)
        if parsed is None:
            await message.answer(_visual_report_range_hint())
            return
        start_date, end_date = parsed
        async with _typing(message):
            try:
                pdf = await build_visual_report_pdf(
                    session,
                    user=user,
                    start_date=start_date,
                    end_date=end_date,
                )
            except ValueError as error:
                await message.answer(str(error))
                return
    filename = f"visual-report-{start_date.isoformat()}-{end_date.isoformat()}.pdf"
    await message.answer_document(
        BufferedInputFile(pdf, filename=filename),
        caption=f"Візуальний звіт: {start_date.isoformat()} - {end_date.isoformat()}",
    )


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
            text = await _format_similar_memory_query(
                session,
                user=user,
                memory_service=memory_service,
                query=query,
            )
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
        await message.answer(reply.text, reply_markup=_inline_reply_keyboard(reply.keyboard, options=reply.keyboard_options))
    if should_embed and entry_id and user_id:
        asyncio.create_task(
            _embed_entry_task(
                settings,
                sessionmaker,
                memory_service,
                entry_id,
                user_id,
                replace_existing=True,
            )
        )


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


@router.callback_query(F.data.startswith("menu:"))
async def menu_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    summary_service: SummaryService,
    memory_service: MemoryService,
    ai_service: AIService,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    action = callback.data or "menu:main"
    if action == "menu:main":
        await callback.answer()
        await _edit_or_answer_menu(callback, "Головне меню.", main_menu_keyboard())
        return
    if action == "menu:day":
        await callback.answer()
        await _edit_or_answer_menu(callback, "День.", day_menu_keyboard())
        return
    if action == "menu:summaries":
        await callback.answer()
        await _edit_or_answer_menu(callback, "Підсумки.", summaries_menu_keyboard())
        return
    if action == "menu:summaries:week":
        await callback.answer()
        await _edit_or_answer_menu(callback, "Тижневий підсумок.", period_choice_keyboard(period="week"))
        return
    if action == "menu:summaries:month":
        await callback.answer()
        await _edit_or_answer_menu(callback, "Місячний підсумок.", period_choice_keyboard(period="month"))
        return
    if action == "menu:memory":
        await callback.answer()
        await _edit_or_answer_menu(
            callback,
            "Пам’ять.",
            memory_menu_keyboard(embeddings_enabled=_embeddings_ready(settings)),
        )
        return
    if action == "menu:life_context":
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            user_settings = await repo.get_user_settings(session, user.id)
            has_items = bool(life_context_items(user_settings))
        await callback.answer()
        await _edit_or_answer_menu(
            callback,
            "Живий контекст. Це окрема обережна пам’ять про людей, місця, проєкти, рутини й назви, які краще не плутати.",
            life_context_menu_keyboard(has_items=has_items),
        )
        return
    if action == "menu:data":
        await callback.answer()
        await _edit_or_answer_menu(callback, "Дані й архів.", data_menu_keyboard())
        return
    if action == "menu:clarifications":
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            user_settings = await repo.get_user_settings(session, user.id)
            text = _format_clarification_queue_status(user_settings)
            keyboard = _clarification_queue_menu_keyboard(user_settings)
        await callback.answer()
        await _edit_or_answer_menu(callback, text, keyboard)
        return

    if action == "menu:data:reanalyze":
        await callback.answer()
        text = (
            "Переаналіз AI заново витягне метрики, емоції, стани й активності.\n\n"
            "Спершу вибери невеликий пробний прохід, період або весь архів. Перед запуском покажу "
            "точну кількість записів. Старі AI-версії залишаться в історії, а нова стане поточною."
        )
        await _edit_or_answer_menu(
            callback,
            text,
            reanalysis_scope_keyboard(),
        )
        return

    if action == "menu:data:visual_report":
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            user_settings = await repo.get_user_settings(session, user.id)
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_pending_input(user_settings, "visual_report_range")},
            )
        await callback.answer()
        await callback.message.answer(
            _visual_report_range_hint(),
            reply_markup=main_reply_keyboard("Наприклад: 2026-07-01 2026-07-09"),
        )
        return

    if action == "menu:data:affect_audit":
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            text = await format_affective_vocabulary_audit(session, user=user)
        await callback.answer()
        await _answer_long_text(callback.message, text, reply_markup=data_menu_keyboard())
        return

    if action in {"menu:day:today", "menu:day:yesterday"}:
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            user_settings = await repo.get_user_settings(session, user.id)
            target_date = await current_journal_date(session, user=user, user_settings=user_settings)
            if action == "menu:day:yesterday":
                target_date -= timedelta(days=1)
            text, day_id = await _format_day_for_date(session, user=user, target_date=target_date)
        await callback.answer()
        await _answer_long_text(
            callback.message,
            text,
            reply_markup=day_detail_keyboard(day_id=day_id) if day_id else day_menu_keyboard(),
        )
        return

    if action == "menu:day:date":
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            user_settings = await repo.get_user_settings(session, user.id)
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_pending_input(user_settings, "day_date")},
            )
        await callback.answer()
        await callback.message.answer(
            "Напиши дату, яку відкрити. Наприклад: 2026-06-30, 30.06.2026, сьогодні або вчора.",
            reply_markup=main_reply_keyboard("Напиши дату: 2026-06-30"),
        )
        return

    if action == "menu:summaries:day":
        await callback.answer("Генерую підсумок")
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            async with _typing(callback.message):
                summary = await summary_service.generate_today_summary(session, user=user)
        await callback.message.answer(summary.short_text, reply_markup=summary_detail_keyboard(summary_id=str(summary.id)))
        return

    if action.startswith("menu:period:"):
        parts = action.split(":")
        period = parts[2] if len(parts) > 2 else "week"
        choice = parts[3] if len(parts) > 3 else "current"
        await callback.answer("Генерую підсумок")
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            async with _typing(callback.message):
                if period == "month" and choice == "previous":
                    summary = await summary_service.generate_previous_month_summary(session, user=user)
                elif period == "month":
                    summary = await summary_service.generate_current_month_summary(session, user=user)
                elif choice == "previous":
                    summary = await summary_service.generate_previous_week_summary(session, user=user)
                else:
                    summary = await summary_service.generate_current_week_summary(session, user=user)
            text = format_period_summary(summary)
            summary_id = str(summary.id)
        await _answer_long_text(callback.message, text, reply_markup=period_detail_keyboard(summary_id=summary_id))
        await _mark_summary_delivered(sessionmaker, summary_id=summary.id)
        return

    if action == "menu:memory:search":
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            user_settings = await repo.get_user_settings(session, user.id)
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_pending_input(user_settings, "memory_search")},
            )
        await callback.answer()
        await callback.message.answer(
            "Що пошукати в пам’яті?",
            reply_markup=main_reply_keyboard("Наприклад: гуляю, не можу почати, спокійний вечір"),
        )
        return

    if action == "menu:memory:graph":
        await callback.answer("Готую граф")
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            async with _typing(callback.message):
                graph_html = await build_memory_graph_html(session, user_id=user.id)
        await callback.message.answer_document(
            BufferedInputFile(graph_html, filename="memory-graph.html"),
            caption="Інтерактивний граф пам’яті. Відкрий HTML-файл у браузері.",
        )
        return

    if action == "menu:memory:export":
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            payload = await export_memory_graph_json(session, user_id=user.id)
        await callback.answer("Готую експорт")
        await callback.message.answer_document(
            BufferedInputFile(payload, filename="memory-graph.json"),
            caption="JSON графа пам’яті. Його можна очистити й імпортувати назад через меню пам’яті.",
        )
        return

    if action == "menu:memory:import":
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            user_settings = await repo.get_user_settings(session, user.id)
            next_json = settings_json_with_pending_memory_graph_import(user_settings, None)
            next_json = settings_json_with_pending_input(_settings_view(next_json), "memory_graph_import")
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": next_json},
            )
        await callback.answer()
        await callback.message.answer(
            "Надішли сюди очищений JSON-експорт графа. Я спершу покажу, скільки вузлів, зв’язків і доказів "
            "буде імпортовано; заміна відбудеться лише після підтвердження.",
        )
        return

    if action == "menu:memory:lexicon":
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            text = await format_personal_lexicon_view(session, user_id=user.id)
        await callback.answer()
        await _answer_long_text(
            callback.message,
            text,
            reply_markup=memory_menu_keyboard(embeddings_enabled=_embeddings_ready(settings)),
        )
        return

    if action == "menu:memory:last":
        if not _embeddings_ready(settings):
            await callback.answer("Embeddings не активні", show_alert=True)
            return
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            recent = await repo.get_recent_entries(session, user_id=user.id, limit=1)
            if not recent or not (recent[-1].raw_text or "").strip():
                text = "Поки немає текстового останнього запису, від якого можна шукати схожі моменти."
            else:
                async with _typing(callback.message):
                    text = await _format_similar_memory_query(
                        session,
                        user=user,
                        memory_service=memory_service,
                        query=recent[-1].raw_text or "",
                    )
        await callback.answer()
        await _answer_long_text(callback.message, text, reply_markup=memory_menu_keyboard(embeddings_enabled=True))
        return

    if action == "menu:memory:influences":
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            snapshots = await repo.list_recent_snapshots(session, user_id=user.id, limit=16)
            prompts = await repo.list_prompts_for_snapshots(
                session,
                snapshot_ids=[snapshot.id for snapshot in snapshots],
            )
            text = _format_semantic_memory_influences(snapshots, prompts, timezone=user.timezone)
        await callback.answer()
        await _answer_long_text(
            callback.message,
            text,
            reply_markup=memory_menu_keyboard(embeddings_enabled=_embeddings_ready(settings)),
        )
        return

    if action == "menu:memory:maintain":
        await callback.answer()
        text = (
            "Перевірка графа послабить старі непідтверджені зв’язки й позначить можливі дублікати "
            "як кандидати в `metadata`.\n\n"
            "Це не зливає вузли, не видаляє докази й не вирішує за AI, що є справжнім дублем. "
            "Таку саму легку перевірку бот також запускає фоном разом із тижневим обслуговуванням."
        )
        await _edit_or_answer_menu(callback, text, memory_maintenance_confirmation_keyboard())
        return

    if action == "menu:memory:review":
        await callback.answer()
        text = (
            "AI-ревізія візьме можливі дублікати з графа й обережно перевірить, чи це один концепт, "
            "альтернативна назва, різні речі або невпевнено.\n\n"
            "Автоматично застосовую тільки високовпевнені рішення для непідтверджених вузлів. "
            "Підтверджені, чутливі та невпевнені пари лишаються для майбутнього підтвердження."
        )
        await _edit_or_answer_menu(
            callback,
            text,
            memory_ai_review_confirmation_keyboard(limit=MEMORY_GRAPH_REVIEW_LIMIT),
        )
        return

    if action == "menu:memory:rebuild":
        if not _embeddings_ready(settings):
            await callback.answer("Embeddings не активні", show_alert=True)
            return
        await callback.answer()
        text = (
            "Перебудова пам’яті заново створить contextual memory capsules, embeddings і graph-зв’язки "
            f"для останніх {MEMORY_REBUILD_LIMIT} записів.\n\n"
            "Це потрібно після нової моделі пам’яті, бо старі embeddings були менш контекстними. "
            "Дія витрачає AI-запити й може тривати кілька хвилин."
        )
        await _edit_or_answer_menu(
            callback,
            text,
            memory_rebuild_confirmation_keyboard(limit=MEMORY_REBUILD_LIMIT),
        )
        return

    if action == "menu:memory:status":
        await callback.answer()
        text = (
            "Embeddings увімкнені й ключ налаштований."
            if _embeddings_ready(settings)
            else "Embeddings зараз не активні. Для пошуку потрібні EMBEDDINGS_ENABLED=true і EMBEDDING_API_KEY."
        )
        await _edit_or_answer_menu(callback, text, memory_menu_keyboard(embeddings_enabled=_embeddings_ready(settings)))
        return

    await callback.answer("Не впізнав дію", show_alert=True)


@router.callback_query(F.data.startswith("memory:import:"))
async def memory_graph_import_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    action = (callback.data or "").rsplit(":", maxsplit=1)[-1]
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        pending = pending_memory_graph_import(user_settings)
        if action == "cancel":
            next_json = settings_json_with_pending_memory_graph_import(user_settings, None)
            next_json = settings_json_without_pending_input(_settings_view(next_json))
            await repo.update_user_settings(session, user_id=user.id, values={"settings_json": next_json})
            text = "Імпорт графа скасовано. Поточний граф не змінювався."
        elif pending is None:
            next_json = settings_json_with_pending_memory_graph_import(user_settings, None)
            next_json = settings_json_without_pending_input(_settings_view(next_json))
            await repo.update_user_settings(session, user_id=user.id, values={"settings_json": next_json})
            text = "Не знайшов підготовлений імпорт. Надішли JSON ще раз через «Імпорт графа»."
        else:
            try:
                result = await replace_memory_graph_from_import(session, user_id=user.id, payload=pending)
            except ValueError as error:
                text = f"Не імпортував граф: {error}"
            else:
                next_json = settings_json_with_pending_memory_graph_import(user_settings, None)
                next_json = settings_json_without_pending_input(_settings_view(next_json))
                await repo.update_user_settings(session, user_id=user.id, values={"settings_json": next_json})
                text = (
                    f"Граф замінено: {result.nodes} вузлів, {result.edges} зв’язків, {result.evidence} доказів."
                    + (f" Пропущено некоректних елементів: {result.skipped}." if result.skipped else "")
                )
    await callback.answer()
    await _clear_inline_keyboard(callback)
    await callback.message.answer(text, reply_markup=memory_menu_keyboard(embeddings_enabled=_embeddings_ready(settings)))


@router.callback_query(F.data.startswith("memory:maintain:"))
async def maintain_memory_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    action = (callback.data or "").rsplit(":", maxsplit=1)[-1]
    if action == "cancel":
        await callback.answer("Скасовано")
        await _edit_or_answer_menu(
            callback,
            "Ок, граф не чіпаю.",
            memory_menu_keyboard(embeddings_enabled=_embeddings_ready(settings)),
        )
        return
    if action != "confirm":
        await callback.answer("Не впізнав дію", show_alert=True)
        return

    await callback.answer("Перевіряю граф")
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        async with _typing(callback.message):
            result = await maintain_memory_graph(session, user_id=user.id)
    await _edit_or_answer_menu(
        callback,
        _format_memory_maintenance_result(result),
        memory_menu_keyboard(embeddings_enabled=_embeddings_ready(settings)),
    )


@router.callback_query(F.data.startswith("memory:review:"))
async def review_memory_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai_service: AIService,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    action = (callback.data or "").rsplit(":", maxsplit=1)[-1]
    if action == "cancel":
        await callback.answer("Скасовано")
        await _edit_or_answer_menu(
            callback,
            "Ок, AI-ревізію графа не запускаю.",
            memory_menu_keyboard(embeddings_enabled=_embeddings_ready(settings)),
        )
        return
    try:
        limit = max(1, min(int(action), 30))
    except ValueError:
        await callback.answer("Не можу прочитати ліміт", show_alert=True)
        return

    await callback.answer("Перевіряю граф")
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        async with _typing(callback.message):
            maintenance = await maintain_memory_graph(session, user_id=user.id)
            review = await review_memory_graph_duplicates(
                session,
                user_id=user.id,
                ai_service=ai_service,
                pair_limit=limit,
            )
    await _edit_or_answer_menu(
        callback,
        _format_memory_ai_review_result(review, maintenance=maintenance),
        memory_menu_keyboard(embeddings_enabled=_embeddings_ready(settings)),
    )


@router.callback_query(F.data.startswith("life_context:"))
async def life_context_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai_service: AIService,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    action = callback.data or ""
    answered = False
    try:
        if action in {"life_context:scan", "life_context:rewrite"}:
            await callback.answer("Дивлюся живий контекст")
            answered = True
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            user_settings = await repo.get_user_settings(session, user.id)
            if action == "life_context:list":
                text = format_life_context_items(life_context_items(user_settings))
                keyboard = life_context_menu_keyboard(has_items=bool(life_context_items(user_settings)))
            elif action == "life_context:scan":
                async with _typing(callback.message):
                    lead_text, review = await start_life_context_review(
                        session,
                        user=user,
                        user_settings=user_settings,
                        ai_service=ai_service,
                    )
                if review:
                    text = f"{lead_text}\n\nМожемо швидко пройти 1-3 питання. Якщо не хочеться зараз, можна відкласти."
                    keyboard = life_context_offer_keyboard()
                else:
                    text = lead_text
                    keyboard = life_context_menu_keyboard(has_items=bool(life_context_items(user_settings)))
            elif action == "life_context:rewrite":
                async with _typing(callback.message):
                    text, rewrite = await start_life_context_rewrite(
                        session,
                        user=user,
                        user_settings=user_settings,
                        ai_service=ai_service,
                    )
                keyboard = (
                    life_context_rewrite_confirmation_keyboard()
                    if rewrite
                    else life_context_menu_keyboard(has_items=bool(life_context_items(user_settings)))
                )
            elif action == "life_context:rewrite:apply":
                text = await apply_life_context_rewrite(session, user=user, user_settings=user_settings)
                keyboard = life_context_menu_keyboard(has_items=True)
            elif action == "life_context:rewrite:cancel":
                text = await cancel_life_context_rewrite(session, user=user, user_settings=user_settings)
                keyboard = life_context_menu_keyboard(has_items=bool(life_context_items(user_settings)))
            elif action in {"life_context:review:start", "life_context:review:next"}:
                review = pending_life_context_review(user_settings)
                candidate = current_life_context_candidate(review)
                if candidate is None:
                    text = "Не бачу активних припущень для перевірки."
                    keyboard = life_context_menu_keyboard(has_items=bool(life_context_items(user_settings)))
                else:
                    text = format_life_context_question(review or {})
                    keyboard = life_context_question_keyboard(candidate)
            elif action == "life_context:review:later":
                text = "Ок, залишив ці припущення на потім."
                keyboard = life_context_continue_keyboard()
            elif action == "life_context:review:stop":
                await repo.update_user_settings(
                    session,
                    user_id=user.id,
                    values={"settings_json": settings_json_with_pending_life_context_review(user_settings, None)},
                )
                text = "Ок, не перевіряю це зараз."
                keyboard = life_context_menu_keyboard(has_items=bool(life_context_items(user_settings)))
            elif action.startswith("life_context:answer:"):
                answer_kind = _life_context_answer_kind(action)
                review = pending_life_context_review(user_settings)
                candidate = current_life_context_candidate(review)
                if candidate is None:
                    text = "Не бачу активного питання про живий контекст."
                    keyboard = life_context_menu_keyboard(has_items=bool(life_context_items(user_settings)))
                elif answer_kind == "free":
                    await repo.update_user_settings(
                        session,
                        user_id=user.id,
                        values={"settings_json": settings_json_with_pending_input(user_settings, "life_context_free_answer")},
                    )
                    if not answered:
                        await callback.answer()
                    await _clear_inline_keyboard(callback)
                    await callback.message.answer(
                        "Напиши наступним повідомленням, як це краще запам’ятати в живому контексті.",
                        reply_markup=main_reply_keyboard("Поясни контекст своїми словами"),
                    )
                    return
                else:
                    answer = _life_context_answer_from_callback(action, candidate)
                    text, next_review = await answer_life_context_candidate(
                        session,
                        user=user,
                        user_settings=user_settings,
                        answer=answer,
                        answer_kind=answer_kind,
                    )
                    if next_review:
                        text = f"{text}\n\n{format_life_context_question(next_review)}"
                        keyboard = life_context_question_keyboard(current_life_context_candidate(next_review) or {})
                    else:
                        keyboard = life_context_menu_keyboard(has_items=True)
            else:
                text = "Не впізнав дію живого контексту."
                keyboard = life_context_menu_keyboard(has_items=bool(life_context_items(user_settings)))
        if not answered:
            await callback.answer()
        await _edit_or_answer_menu(callback, text, keyboard)
    except Exception:
        logger.exception("Life context callback failed", extra={"callback_data": action})
        if not answered:
            await callback.answer("Не вийшло обробити дію", show_alert=True)
        if callback.message is not None:
            await callback.message.answer(
                "Щось зламалося під час роботи з живим контекстом. Я це залогував; можна спробувати ще раз трохи пізніше.",
                reply_markup=life_context_menu_keyboard(has_items=True),
            )


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
        emotion_chart = await build_emotion_timeline_png(session, user=user)
        spectrum_chart = await build_affect_spectrum_png(session, user=user)
    await callback.answer()
    await _answer_long_text(callback.message, text)
    if chart is not None:
        await callback.message.answer_photo(
            BufferedInputFile(chart, filename="metrics.png"),
        )
    if emotion_chart is not None:
        await callback.message.answer_photo(
            BufferedInputFile(emotion_chart, filename="emotions.png"),
            caption="Емоційна карта дня.",
        )
    if spectrum_chart is not None:
        await callback.message.answer_photo(
            BufferedInputFile(spectrum_chart, filename="spectrum.png"),
            caption="Спектр стану дня.",
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
    summary_service: SummaryService,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    parsed = _parse_scoped_callback(callback.data or "", prefix="dayview")
    if parsed is None:
        await callback.answer("Не можу відкрити цей день")
        return
    await callback.answer()
    day_id, section = parsed
    chart = None
    emotion_chart = None
    spectrum_chart = None
    moments: list[PhotoMoment] = []
    reply_day_id = None
    reply_markup = None
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        day = await repo.get_day(session, day_id=day_id)
        if day is None or day.user_id != user.id:
            text = "Не знайшов цей день в архіві."
        else:
            reply_day_id = str(day.id)
            if section == "entries":
                entries = list(await repo.list_day_entries(session, day_id=day.id))
                text = _format_entry_management_view(entries, timezone=user.timezone)
                reply_markup = entry_management_keyboard(
                    day_id=str(day.id),
                    entries=_entry_management_buttons(entries, timezone=user.timezone),
                )
            elif section == "turning_points":
                turning_points = await get_day_turning_points(session, day=day)
                text = format_day_turning_points(turning_points, timezone=user.timezone)
                reply_markup = turning_points_keyboard(
                    day_id=str(day.id),
                    labels=_turning_point_labels(turning_points, timezone=user.timezone),
                )
            elif section == "refresh":
                async with _typing(callback.message):
                    summary = await summary_service.generate_day_summary(session, user=user, day=day)
                text = summary.short_text
                reply_markup = summary_detail_keyboard(summary_id=str(summary.id))
            else:
                text, chart, emotion_chart, spectrum_chart, moments = await _format_day_detail_section(
                    session,
                    user=user,
                    day=day,
                    section=section,
                )
    await _answer_long_text(
        callback.message,
        text,
        reply_markup=reply_markup or (day_detail_keyboard(day_id=reply_day_id) if reply_day_id else None),
    )
    if chart is not None:
        await callback.message.answer_photo(
            BufferedInputFile(chart, filename="metrics.png"),
        )
    if emotion_chart is not None:
        await callback.message.answer_photo(
            BufferedInputFile(emotion_chart, filename="emotions.png"),
            caption="Емоційна карта дня.",
        )
    if spectrum_chart is not None:
        await callback.message.answer_photo(
            BufferedInputFile(spectrum_chart, filename="spectrum.png"),
            caption="Спектр стану дня.",
        )
    if moments:
        await _send_photo_moments(callback.message, moments)


@router.callback_query(F.data.startswith("entry:"))
async def entry_management_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    parts = (callback.data or "").split(":")
    if len(parts) != 3:
        await callback.answer("Не впізнав дію", show_alert=True)
        return
    action = parts[1]
    entry_id = _uuid_or_none(parts[2])
    if entry_id is None:
        await callback.answer("Не можу прочитати запис", show_alert=True)
        return

    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        entry = await repo.get_entry(session, entry_id=entry_id)
        if entry is None or entry.user_id != user.id:
            text = "Не знайшов цей запис."
            reply_markup = None
        elif action == "delete":
            day_id = str(entry.day_id) if entry.day_id else None
            text = (
                "Видалити цей запис?\n\n"
                f"{_format_entry_delete_preview(entry, timezone=user.timezone)}\n\n"
                "Це також прибере його AI-аналіз і embedding, щоб він не впливав на пам’ять та метрики."
            )
            reply_markup = entry_delete_confirmation_keyboard(entry_id=str(entry.id), day_id=day_id)
        elif action == "confirm_delete":
            day_id = str(entry.day_id) if entry.day_id else None
            preview = _format_entry_delete_preview(entry, timezone=user.timezone)
            should_reopen_day = entry.source == "sleep_marker" and entry.day_id is not None
            deleted = await repo.delete_entry_tree(session, entry_id=entry.id, user_id=user.id)
            if deleted is None:
                text = "Не зміг видалити запис: він уже зник або не належить тобі."
                reply_markup = None
            else:
                if deleted.day_id is not None:
                    await repo.mark_day_summaries_stale(
                        session,
                        user_id=user.id,
                        day_id=deleted.day_id,
                        reason="entry_deleted",
                    )
                if should_reopen_day and deleted.day_id is not None:
                    await repo.reopen_day(session, day_id=deleted.day_id)
                text = f"Видалив запис:\n\n{preview}"
                reply_markup = day_detail_keyboard(day_id=day_id) if day_id else main_menu_keyboard()
        else:
            await callback.answer("Не впізнав дію", show_alert=True)
            return

    await callback.answer()
    await _edit_or_answer_menu(callback, text, reply_markup)


@router.callback_query(F.data.startswith("metric:"))
async def metric_calibration_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    if (callback.data or "").startswith("metric:start:"):
        parsed_start = _parse_metric_start_callback(callback.data or "")
        if parsed_start is None:
            await callback.answer("Не впізнав метрику", show_alert=True)
            return
        entry_id, metric = parsed_start
        await callback.answer()
        await callback.message.answer(
            _metric_prompt_text(metric),
            reply_markup=metric_score_keyboard(entry_id=str(entry_id), metric=metric),
        )
        return
    parsed = _parse_metric_callback(callback.data or "")
    if parsed is None:
        await callback.answer("Не впізнав уточнення", show_alert=True)
        return
    entry_id, metric, score_text = parsed
    await callback.answer()
    await _clear_inline_keyboard(callback)
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        await _clear_pending_post_entry_followup(session, user=user, entry_id=entry_id)
        if score_text == "skip":
            replies = [BotReply("Ок, не уточнюю цю метрику.")]
        else:
            try:
                score = int(score_text)
            except ValueError:
                replies = [BotReply("Не зміг прочитати оцінку.")]
            else:
                if score < 0 or score > 10:
                    replies = [BotReply("Оцінка має бути від 0 до 10.")]
                else:
                    entry = await repo.get_entry(session, entry_id=entry_id)
                    if entry is None or entry.user_id != user.id:
                        replies = [BotReply("Не знайшов цей запис.")]
                    else:
                        replies = await apply_metric_calibration(
                            session,
                            settings=settings,
                            user=user,
                            entry=entry,
                            metric=metric,
                            score=score,
                        )
        await _set_pending_post_entry_followup_from_replies(
            session,
            user=user,
            entry_id=entry_id,
            replies=replies,
        )
    for reply in replies:
        await callback.message.answer(reply.text, reply_markup=_inline_reply_keyboard(reply.keyboard, options=reply.keyboard_options))


@router.callback_query(F.data.startswith("emotion:") | F.data.startswith("e:"))
async def emotion_calibration_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    if (callback.data or "").startswith("emotion:start:"):
        entry_id = _uuid_or_none((callback.data or "").split(":", maxsplit=2)[2])
        if entry_id is None:
            await callback.answer("Не впізнав запис", show_alert=True)
            return
        await callback.answer()
        await callback.message.answer(
            "Що ближче до емоційного тону цього моменту? Можна обрати кілька варіантів.",
            reply_markup=emotion_calibration_keyboard(entry_id=str(entry_id)),
        )
        return
    parsed_per_emotion = _parse_per_emotion_intensity_callback(callback.data or "")
    if parsed_per_emotion is not None:
        action, entry_id, emotions, intensity_levels, time_scope, position, value = parsed_per_emotion
        if not emotions:
            await callback.answer("Не бачу обраних емоцій", show_alert=True)
            return
        if action == "set":
            intensity_levels[position] = value
            next_position = min(position + 1, len(emotions) - 1)
            await callback.answer()
            await callback.message.edit_text(
                _emotion_intensity_prompt(emotions, intensity_levels, next_position, time_scope),
                reply_markup=emotion_intensity_keyboard(
                    entry_id=str(entry_id),
                    selected=emotions,
                    intensity_levels=intensity_levels,
                    position=next_position,
                    time_scope=time_scope,
                ),
            )
            return
        if action == "navigate":
            await callback.answer()
            await callback.message.edit_text(
                _emotion_intensity_prompt(emotions, intensity_levels, position, time_scope),
                reply_markup=emotion_intensity_keyboard(
                    entry_id=str(entry_id),
                    selected=emotions,
                    intensity_levels=intensity_levels,
                    position=position,
                    time_scope=time_scope,
                ),
            )
            return
        if action == "scope":
            next_scope = "current" if time_scope == "mentioned_not_felt" else "mentioned_not_felt"
            await callback.answer()
            await callback.message.edit_text(
                _emotion_intensity_prompt(emotions, intensity_levels, position, next_scope),
                reply_markup=emotion_intensity_keyboard(
                    entry_id=str(entry_id),
                    selected=emotions,
                    intensity_levels=intensity_levels,
                    position=position,
                    time_scope=next_scope,
                ),
            )
            return
        await callback.answer()
        await _clear_inline_keyboard(callback)
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            await _clear_pending_post_entry_followup(session, user=user, entry_id=entry_id)
            entry = await repo.get_entry(session, entry_id=entry_id)
            if entry is None or entry.user_id != user.id:
                reply = BotReply("Не знайшов цей запис.")
            else:
                reply = await apply_emotion_calibration(
                    session,
                    settings=settings,
                    user=user,
                    entry=entry,
                    emotions=emotions,
                    emotion_intensity_levels=dict(zip(emotions, intensity_levels, strict=True)),
                    time_scope=time_scope,
                )
        await callback.message.answer(reply.text, reply_markup=_inline_reply_keyboard(reply.keyboard, options=reply.keyboard_options))
        return
    parsed_intensity = _parse_emotion_intensity_callback(callback.data or "")
    if parsed_intensity is not None:
        entry_id, emotions, intensity_level = parsed_intensity
        await callback.answer()
        await _clear_inline_keyboard(callback)
        if not emotions:
            await callback.message.answer("Не бачу обраних емоцій для уточнення.")
            return
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            await _clear_pending_post_entry_followup(session, user=user, entry_id=entry_id)
            entry = await repo.get_entry(session, entry_id=entry_id)
            if entry is None or entry.user_id != user.id:
                reply = BotReply("Не знайшов цей запис.")
            else:
                reply = await apply_emotion_calibration(
                    session,
                    settings=settings,
                    user=user,
                    entry=entry,
                    emotions=emotions,
                    intensity_level=intensity_level,
                )
        await callback.message.answer(reply.text, reply_markup=_inline_reply_keyboard(reply.keyboard, options=reply.keyboard_options))
        return
    parsed = _parse_emotion_callback(callback.data or "")
    if parsed is None:
        await callback.answer("Не впізнав емоцію", show_alert=True)
        return
    action, entry_id, emotions = parsed
    if action == "toggle":
        if len(emotions) > 8:
            await callback.answer("За раз можна уточнити до 8 емоцій.", show_alert=True)
            return
        await callback.answer()
        await callback.message.edit_reply_markup(
            reply_markup=emotion_calibration_keyboard(entry_id=str(entry_id), selected=emotions)
        )
        return
    if action == "skip":
        await callback.answer()
        await _clear_inline_keyboard(callback)
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            await _clear_pending_post_entry_followup(session, user=user, entry_id=entry_id)
        await callback.message.answer("Ок, не уточнюю емоцію.")
        return
    if action == "custom":
        await callback.answer()
        await _clear_inline_keyboard(callback)
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            user_settings = await repo.get_user_settings(session, user.id)
            next_settings_json = settings_json_with_pending_input(user_settings, "correction")
            next_settings_json["pending_correction_entry_id"] = str(entry_id)
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": next_settings_json},
            )
        await callback.message.answer(
            "Опиши своїми словами, що було емоційно важливим у тому моменті. Я переосмислю саме цей запис, а не створю новий."
        )
        return
    if action != "save":
        await callback.answer("Не впізнав дію", show_alert=True)
        return
    await callback.answer()
    if not emotions:
        await callback.message.answer("Обери хоча б одну емоцію зі списку або натисни «Не уточнювати».")
        return
    await _clear_inline_keyboard(callback)
    await callback.message.answer(
        _emotion_intensity_prompt(emotions, ["unclear"] * len(emotions), 0, "current"),
        reply_markup=emotion_intensity_keyboard(entry_id=str(entry_id), selected=emotions),
    )


@router.callback_query(F.data == "interpretation:ok")
async def interpretation_ok_callback_handler(callback: CallbackQuery, settings: Settings) -> None:
    if not await _allowed_callback(callback, settings):
        return
    await callback.answer("Ок")
    await _clear_inline_keyboard(callback)


@router.callback_query(F.data.startswith("summary:"))
async def summary_detail_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    summary_service: SummaryService,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    await callback.answer()
    parts = (callback.data or "").split(":")
    section = parts[-1] if len(parts) >= 2 else "story"
    chart = None
    emotion_chart = None
    spectrum_chart = None
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
                if section == "refresh" and day is not None and day.user_id == user.id:
                    async with _typing(callback.message):
                        summary = await summary_service.generate_day_summary(session, user=user, day=day)
                    text = summary.short_text
                    reply_markup = summary_detail_keyboard(summary_id=str(summary.id))
                elif day is not None and day.user_id == user.id and section == "turning_points":
                    turning_points = await get_day_turning_points(session, day=day)
                    text = format_day_turning_points(turning_points, timezone=user.timezone)
                    reply_markup = turning_points_keyboard(
                        day_id=str(day.id),
                        labels=_turning_point_labels(turning_points, timezone=user.timezone),
                    )
                elif day is not None and day.user_id == user.id and section in _DAY_DETAIL_SECTIONS:
                    text, chart, emotion_chart, spectrum_chart, moments = await _format_day_detail_section(
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
                if section == "refresh" and day is not None and day.user_id == user.id:
                    async with _typing(callback.message):
                        summary = await summary_service.generate_day_summary(session, user=user, day=day)
                    text = summary.short_text
                    reply_markup = summary_detail_keyboard(summary_id=str(summary.id))
                elif day is not None and day.user_id == user.id and section == "turning_points":
                    turning_points = await get_day_turning_points(session, day=day)
                    text = format_day_turning_points(turning_points, timezone=user.timezone)
                    reply_markup = turning_points_keyboard(
                        day_id=str(day.id),
                        labels=_turning_point_labels(turning_points, timezone=user.timezone),
                    )
                elif day is not None and day.user_id == user.id and section in _DAY_DETAIL_SECTIONS:
                    text, chart, emotion_chart, spectrum_chart, moments = await _format_day_detail_section(
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
                emotion_chart = await build_emotion_timeline_png(session, user=user)
                spectrum_chart = await build_affect_spectrum_png(session, user=user)
            elif section == "photos":
                moments = await get_today_photo_moments(session, user=user)
                text = format_photo_moments_view(moments, timezone=user.timezone)
            else:
                text = await format_latest_summary_section(session, user=user, section=section)
    await _answer_long_text(callback.message, text, reply_markup=reply_markup)
    if chart is not None:
        await callback.message.answer_photo(
            BufferedInputFile(chart, filename="metrics.png"),
        )
    if emotion_chart is not None:
        await callback.message.answer_photo(
            BufferedInputFile(emotion_chart, filename="emotions.png"),
            caption="Емоційна карта дня.",
        )
    if spectrum_chart is not None:
        await callback.message.answer_photo(
            BufferedInputFile(spectrum_chart, filename="spectrum.png"),
            caption="Спектр стану дня.",
        )
    if moments:
        await _send_photo_moments(callback.message, moments)


@router.callback_query(F.data.startswith("turning:"))
async def turning_points_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    parts = (callback.data or "").split(":")
    if len(parts) not in {3, 4} or parts[1] not in {"list", "detail", "entry"}:
        await callback.answer("Не впізнав поворот", show_alert=True)
        return
    action, day_id = parts[1], parts[2]
    index = None
    if action in {"detail", "entry"}:
        try:
            index = int(parts[3])
        except (IndexError, ValueError):
            await callback.answer("Не впізнав поворот", show_alert=True)
            return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        day = await repo.get_day(session, day_id=day_id)
        if day is None or day.user_id != user.id:
            text = "Не знайшов цей день в архіві."
            reply_markup = None
        else:
            points = await get_day_turning_points(session, day=day)
            if action == "list" or index is None:
                text = format_day_turning_points(points, timezone=user.timezone)
                reply_markup = turning_points_keyboard(
                    day_id=str(day.id),
                    labels=_turning_point_labels(points, timezone=user.timezone),
                )
            elif not 0 <= index < len(points):
                text = "Цей поворот уже не доступний. Можливо, підсумок дня оновився."
                reply_markup = turning_points_keyboard(
                    day_id=str(day.id),
                    labels=_turning_point_labels(points, timezone=user.timezone),
                )
            elif action == "detail":
                text = format_day_turning_point(points[index], timezone=user.timezone, index=index + 1)
                reply_markup = turning_point_detail_keyboard(day_id=str(day.id), index=index)
            else:
                entry = points[index].entry
                text = "\n".join(
                    [
                        f"Запис · {_entry_time_label(entry, user.timezone)}",
                        "",
                        entry.raw_text or "[без тексту]",
                    ]
                )
                reply_markup = turning_point_detail_keyboard(day_id=str(day.id), index=index)
    await callback.answer()
    await _answer_long_text(callback.message, text, reply_markup=reply_markup)


@router.callback_query(F.data == "nav:home")
async def home_callback_handler(callback: CallbackQuery, settings: Settings) -> None:
    if not await _allowed_callback(callback, settings):
        return
    await callback.answer()
    await callback.message.answer("Головне меню.", reply_markup=main_menu_keyboard())


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


@router.callback_query(F.data.startswith("features:scope:"))
async def reanalysis_scope_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    parts = (callback.data or "").split(":")
    if len(parts) < 3:
        await callback.answer("Не впізнав вибір", show_alert=True)
        return
    choice = parts[2]
    if choice == "range":
        async with sessionmaker() as session, session.begin():
            user = await _get_or_create_callback_user(session, callback, settings)
            user_settings = await repo.get_user_settings(session, user.id)
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_pending_input(user_settings, "reanalysis_range")},
            )
        await callback.answer()
        await callback.message.answer(_reanalysis_range_hint(), reply_markup=main_reply_keyboard("Наприклад: 2026-07-01 2026-07-09"))
        return

    scope: str
    limit: int | None = None
    start_date = end_date = None
    label: str
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        today = await current_journal_date(session, user=user, user_settings=user_settings)
    if choice == "recent" and len(parts) == 4:
        try:
            limit = max(1, min(int(parts[3]), FEATURE_REANALYSIS_LIMIT))
        except ValueError:
            await callback.answer("Не можу прочитати обсяг", show_alert=True)
            return
        scope = "recent"
        label = f"останні {limit} записів"
        action = f"recent:{limit}"
    elif choice == "days" and len(parts) == 4:
        try:
            days = max(1, min(int(parts[3]), 31))
        except ValueError:
            await callback.answer("Не можу прочитати період", show_alert=True)
            return
        scope = "range"
        start_date, end_date = today - timedelta(days=days - 1), today
        label = f"журнальні дні {start_date.isoformat()} - {end_date.isoformat()}"
        action = f"range:{start_date.strftime('%Y%m%d')}:{end_date.strftime('%Y%m%d')}"
    elif choice == "all" and len(parts) == 3:
        scope = "all"
        label = "увесь архів"
        action = "all"
    else:
        await callback.answer("Не впізнав вибір", show_alert=True)
        return

    count = await count_entry_feature_reanalysis(
        sessionmaker=sessionmaker,
        telegram_user_id=callback.from_user.id,
        scope=scope,  # type: ignore[arg-type]
        limit=limit,
        start_date=start_date,
        end_date=end_date,
    )
    await callback.answer()
    if count == 0:
        await _edit_or_answer_menu(callback, f"У виборі «{label}» ще немає записів.", reanalysis_scope_keyboard())
        return
    await _edit_or_answer_menu(
        callback,
        (
            f"Вибрано: {label}.\n"
            f"Записів для переаналізу: {count}.\n\n"
            "Після завершення покажу, як змінилося покриття графіків. Це витратить AI-запити."
        ),
        reanalysis_confirmation_keyboard(action=action, selected=count),
    )


@router.callback_query(F.data.startswith("features:reanalyze:"))
async def reanalyze_features_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai_service: AIService,
    bot: Bot,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    action = (callback.data or "").removeprefix("features:reanalyze:")
    if action == "cancel":
        await callback.answer("Скасовано")
        await _edit_or_answer_menu(callback, "Ок, переаналіз не запускаю.", data_menu_keyboard())
        return
    parsed = _parse_reanalysis_action(action)
    if parsed is None:
        await callback.answer("Не можу прочитати вибір", show_alert=True)
        return
    scope, limit, start_date, end_date = parsed
    label = _reanalysis_scope_label(scope=scope, limit=limit, start_date=start_date, end_date=end_date)

    await callback.answer("Запускаю переаналіз")
    await _clear_inline_keyboard(callback)
    await callback.message.answer(
        f"Запустив переаналіз: {label}. Напишу, коли буде готово."
    )
    asyncio.create_task(
        _reanalyze_features_task(
            bot=bot,
            chat_id=callback.message.chat.id,
            telegram_user_id=callback.from_user.id,
            settings=settings,
            sessionmaker=sessionmaker,
            ai_service=ai_service,
            scope=scope,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
        )
    )


@router.callback_query(F.data.startswith("memory:rebuild:"))
async def rebuild_memory_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai_service: AIService,
    bot: Bot,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    action = (callback.data or "").rsplit(":", maxsplit=1)[-1]
    if action == "cancel":
        await callback.answer("Скасовано")
        await _edit_or_answer_menu(
            callback,
            "Ок, пам’ять не перебудовую.",
            memory_menu_keyboard(embeddings_enabled=_embeddings_ready(settings)),
        )
        return
    if not _embeddings_ready(settings):
        await callback.answer("Embeddings не активні", show_alert=True)
        return
    try:
        limit = max(1, min(int(action), 1000))
    except ValueError:
        await callback.answer("Не можу прочитати ліміт", show_alert=True)
        return

    await callback.answer("Запускаю перебудову")
    await _clear_inline_keyboard(callback)
    await callback.message.answer(
        f"Запустив перебудову пам’яті для останніх {limit} записів. Напишу, коли буде готово."
    )
    asyncio.create_task(
        _rebuild_memory_task(
            bot=bot,
            chat_id=callback.message.chat.id,
            telegram_user_id=callback.from_user.id,
            settings=settings,
            sessionmaker=sessionmaker,
            ai_service=ai_service,
            limit=limit,
        )
    )


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
        if action.startswith("settings:section:"):
            notice = "Налаштування."
        elif action == "settings:toggle:pause":
            await repo.set_user_active(session, user_id=user.id, is_active=True)
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_snapshot_pause(user_settings, not snapshots_paused(user_settings))},
            )
            notice = "Автоматичні зрізи оновлено."
        elif action == "settings:toggle:context_quiet":
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_context_quiet(user_settings, not context_quiet_enabled(user_settings))},
            )
            notice = "Контекстну тишу оновлено."
        elif action == "settings:pause":
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
        elif action == "settings:toggle:adaptive_observation":
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={
                    "settings_json": settings_json_with_adaptive_observation(
                        user_settings,
                        not adaptive_observation_enabled(user_settings),
                    )
                },
            )
            notice = "Адаптивну частоту увімкнено." if adaptive_observation_enabled(user_settings) else "Адаптивну частоту вимкнено."
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
        keyboard = _settings_keyboard_for_action(action, user_settings=user_settings)
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


@router.callback_query(F.data.startswith("correction:start"))
async def correction_start_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    target_entry_id = _correction_target_from_callback(callback.data or "")
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        if pending_post_entry_followup(user_settings) is not None:
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_pending_post_entry_followup(user_settings, None)},
            )
        next_settings_json = settings_json_with_pending_input(user_settings, "correction")
        if target_entry_id:
            next_settings_json["pending_correction_entry_id"] = str(target_entry_id)
        else:
            next_settings_json.pop("pending_correction_entry_id", None)
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": next_settings_json},
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
    bot: Bot,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    interaction_service: InteractionService,
    memory_service: MemoryService,
    summary_service: SummaryService,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    action = (callback.data or "").split(":", maxsplit=1)[1]
    if action not in VOICE_TRANSCRIPTION_ACTIONS:
        await callback.answer("Не впізнав дію", show_alert=True)
        return
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
                    summary_service=summary_service,
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
            reply_markup=_inline_reply_keyboard(reply.keyboard, options=reply.keyboard_options) or fallback_markup,
        )
    if should_embed and entry_id and user_id:
        asyncio.create_task(_embed_entry_task(settings, sessionmaker, memory_service, entry_id, user_id))
        asyncio.create_task(
            _maybe_offer_life_context_task(
                bot=bot,
                settings=settings,
                sessionmaker=sessionmaker,
                ai_service=interaction_service.ai,
                user_id=user_id,
                chat_id=callback.message.chat.id,
            )
        )


@router.callback_query(F.data.startswith("manual:"))
async def manual_entry_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    interaction_service: InteractionService,
    memory_service: MemoryService,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    action = (callback.data or "").split(":", maxsplit=1)[1]
    if action not in MANUAL_ENTRY_ACTIONS:
        await callback.answer("Не впізнав дію", show_alert=True)
        return
    entry_id: UUID | None = None
    should_embed = False
    replies: list[BotReply] = []
    user_id: UUID | None = None

    await _clear_inline_keyboard(callback)
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_id = user.id
        user_settings = await repo.get_user_settings(session, user.id)
        pending = pending_manual_entry(user_settings)
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_without_pending_manual_entry(user_settings)},
        )
        if pending is None:
            replies = [BotReply("Не бачу непідтвердженого тексту.")]
        elif action == "ignore":
            replies = [BotReply("Ок, не записую це.")]
        else:
            text = str(pending.get("text") or "").strip()
            if not text:
                replies = [BotReply("Текст порожній, нічого не записую.")]
            else:
                async with _typing(callback.message):
                    result = await interaction_service.handle_text_entry(
                        session,
                        user=user,
                        text=text,
                        telegram_message_id=_pending_int(pending.get("telegram_message_id")),
                        reply_to_message_id=_pending_int(pending.get("reply_to_message_id")),
                        source="manual_confirmed",
                    )
                entry_id = result.entry_id
                should_embed = result.should_embed_entry
                replies = result.replies

    await callback.answer()
    for reply in replies:
        await callback.message.answer(reply.text, reply_markup=_inline_reply_keyboard(reply.keyboard, options=reply.keyboard_options))
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
    await callback.answer("Генерую підсумок")
    parts = (callback.data or "").split(":")
    period = parts[1] if len(parts) > 1 else "week"
    choice = parts[2] if len(parts) > 2 else "current"
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        async with _typing(callback.message):
            if period == "month" and choice == "previous":
                summary = await summary_service.generate_previous_month_summary(session, user=user)
            elif period == "month":
                summary = await summary_service.generate_current_month_summary(session, user=user)
            elif choice == "previous":
                summary = await summary_service.generate_previous_week_summary(session, user=user)
            else:
                summary = await summary_service.generate_current_week_summary(session, user=user)
        text = format_period_summary(summary)
        summary_id = str(summary.id)
    await _answer_long_text(callback.message, text, reply_markup=period_detail_keyboard(summary_id=summary_id))
    await _mark_summary_delivered(sessionmaker, summary_id=summary.id)


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
    await callback.answer()
    summary_id, section = parsed
    chart = None
    emotion_chart = None
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
            elif section == "emotions":
                text = format_period_emotions_view(summary)
            elif section == "patterns":
                text = format_period_patterns_view(summary)
            elif section == "turning_points":
                text = format_period_turning_points_view(summary)
            elif section == "chart":
                text = "Динаміка за період."
                chart = await build_period_metrics_chart_png(session, user=user, summary=summary)
                emotion_chart = await build_period_emotion_chart_png(session, user=user, summary=summary)
                if chart is None and emotion_chart is None:
                    text = "Для графіків за цей період поки замало метрик."
            elif section == "days":
                text = await format_period_days_view(session, user=user, summary=summary)
            else:
                text = format_period_summary(summary)
    await _answer_long_text(callback.message, text, reply_markup=reply_markup)
    if chart is not None:
        await callback.message.answer_photo(
            BufferedInputFile(chart, filename="period-metrics.png"),
        )
    if emotion_chart is not None:
        await callback.message.answer_photo(
            BufferedInputFile(emotion_chart, filename="period-emotions.png"),
        )


@router.message(F.document)
async def memory_graph_import_document_handler(
    message: Message,
    bot: Bot,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed(message, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        waiting_for_import = pending_input(user_settings) == "memory_graph_import"
    if not waiting_for_import:
        await message.answer("Файл не використовую як щоденниковий запис. Імпорт графа можна почати через «Пам’ять → Імпорт графа».")
        return
    document = message.document
    if document is None or (document.file_size or 0) > 2_000_000:
        await message.answer("Надішли JSON графа розміром до 2 МБ.")
        return
    buffer = BytesIO()
    try:
        await bot.download(document.file_id, destination=buffer)
        payload, preview = parse_memory_graph_import(buffer.getvalue())
    except ValueError as error:
        await message.answer(f"Не підготував імпорт: {error}")
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        if pending_input(user_settings) != "memory_graph_import":
            await message.answer("Імпорт уже скасовано або замінено іншою дією. Почни його ще раз через меню пам’яті.")
            return
        next_json = settings_json_with_pending_memory_graph_import(user_settings, payload)
        await repo.update_user_settings(session, user_id=user.id, values={"settings_json": next_json})
    skipped = f"\nПропущено некоректних елементів: {preview.skipped}." if preview.skipped else ""
    await message.answer(
        "Підготував імпорт графа:\n"
        f"- вузлів: {preview.nodes}\n"
        f"- зв’язків: {preview.edges}\n"
        f"- доказів: {preview.evidence}{skipped}\n\n"
        "Після підтвердження поточний граф буде замінено. Сирі записи, фотографії, підсумки та AI-аналізи не зміняться.",
        reply_markup=memory_graph_import_confirmation_keyboard(),
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
    replace_existing_embedding = False
    voice_note: VoiceNoteTranscription | None = None
    direct_response_sent = False
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_message_user(session, message, settings)
        user_id = user.id
        user_settings = await repo.get_user_settings(session, user.id)
        pending_kind = pending_input(user_settings)
        pending_deferred = pending_clarification(user_settings)
        if pending_kind == "sleep_reflection" and await repo.get_open_snapshot(session, user_id=user.id):
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_without_pending_input(user_settings)},
            )
            pending_kind = None
        manual_pending = pending_manual_entry(user_settings)
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
                                target_pending_kind=_voice_target_pending_kind(
                                    pending_kind or ("clarification_queue" if pending_deferred else None)
                                ),
                            ),
                        )
                    },
                )
                target_hint = (
                    "\n\nПісля підтвердження використаю це як відповідь на питання про живий контекст."
                    if _voice_target_pending_kind(pending_kind) == "life_context_free_answer"
                    else "\n\nПісля підтвердження використаю це як твою оцінку дня і закрию день."
                    if _voice_target_pending_kind(pending_kind) == "sleep_reflection"
                    else ""
                )
                replies = [
                    BotReply(
                        "Перевір транскрипцію голосового:\n"
                        f"{_voice_transcription_preview(text)}\n\n"
                        "Якщо все правильно, підтвердь. Якщо ні — натисни «Виправити текст» або просто надішли правильний текст наступним повідомленням."
                        f"{target_hint}",
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
        pending_deferred = pending_clarification(user_settings)
        manual_pending = pending_manual_entry(user_settings)
        if replies:
            pass
        elif pending_kind == "day_date" and not message.photo:
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_without_pending_input(user_settings)},
            )
            today = await current_journal_date(session, user=user, user_settings=user_settings)
            target_date = _parse_day_query(text, user.timezone, today=today)
            if target_date is None:
                await message.answer(
                    "Не впізнав дату. Напиши, наприклад: 2026-06-30, 30.06.2026, сьогодні або вчора.",
                    reply_markup=day_menu_keyboard(),
                )
            else:
                day_text, day_id = await _format_day_for_date(session, user=user, target_date=target_date)
                await _answer_long_text(
                    message,
                    day_text,
                    reply_markup=day_detail_keyboard(day_id=day_id) if day_id else day_menu_keyboard(),
                )
            direct_response_sent = True
        elif pending_kind == "memory_search" and not message.photo:
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_without_pending_input(user_settings)},
            )
            if not _embeddings_ready(settings):
                await message.answer(
                    "Пошук у пам’яті зараз не активний: потрібні EMBEDDINGS_ENABLED=true і EMBEDDING_API_KEY.",
                    reply_markup=memory_menu_keyboard(embeddings_enabled=False),
                )
            else:
                async with _typing(message, bot):
                    memory_text = await _format_similar_memory_query(
                        session,
                        user=user,
                        memory_service=memory_service,
                        query=text,
                    )
                await _answer_long_text(
                    message,
                    memory_text,
                    reply_markup=memory_menu_keyboard(embeddings_enabled=True),
                )
            direct_response_sent = True
        elif pending_kind == "visual_report_range" and not message.photo:
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_without_pending_input(user_settings)},
            )
            today = await current_journal_date(session, user=user, user_settings=user_settings)
            parsed = _parse_date_range_query(text, user.timezone, today=today)
            if parsed is None:
                await message.answer(_visual_report_range_hint(), reply_markup=data_menu_keyboard())
            else:
                start_date, end_date = parsed
                async with _typing(message, bot):
                    try:
                        pdf = await build_visual_report_pdf(
                            session,
                            user=user,
                            start_date=start_date,
                            end_date=end_date,
                        )
                    except ValueError as error:
                        await message.answer(str(error), reply_markup=data_menu_keyboard())
                    else:
                        filename = f"visual-report-{start_date.isoformat()}-{end_date.isoformat()}.pdf"
                        await message.answer_document(
                            BufferedInputFile(pdf, filename=filename),
                            caption=f"Візуальний звіт: {start_date.isoformat()} - {end_date.isoformat()}",
                        )
            direct_response_sent = True
        elif pending_kind == "reanalysis_range" and not message.photo:
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_without_pending_input(user_settings)},
            )
            today = await current_journal_date(session, user=user, user_settings=user_settings)
            parsed = _parse_date_range_query(text, user.timezone, today=today)
            if parsed is None:
                await message.answer(_reanalysis_range_hint(), reply_markup=data_menu_keyboard())
            else:
                start_date, end_date = parsed
                count = len(
                    await repo.list_entries_for_journal_dates(
                        session,
                        user_id=user.id,
                        start_date=start_date,
                        end_date=end_date,
                    )
                )
                if count == 0:
                    await message.answer("У цьому діапазоні ще немає записів.", reply_markup=reanalysis_scope_keyboard())
                else:
                    action = f"range:{start_date.strftime('%Y%m%d')}:{end_date.strftime('%Y%m%d')}"
                    await message.answer(
                        (
                            f"Вибрано журнальні дні {start_date.isoformat()} - {end_date.isoformat()}.\n"
                            f"Записів для переаналізу: {count}.\n\n"
                            "Після завершення покажу, як змінилося покриття графіків."
                        ),
                        reply_markup=reanalysis_confirmation_keyboard(action=action, selected=count),
                    )
            direct_response_sent = True
        elif pending_deferred and not message.photo:
            target_entry_id = _uuid_or_none(str(pending_deferred.get("entry_id") or ""))
            queue = clarification_queue(user_settings)
            related_ids = _clarification_related_ids(pending_deferred)
            next_queue = [
                {**item, "status": "answered", "answered_at": utc_now().isoformat()}
                if str(item.get("id")) in related_ids
                else item
                for item in queue
            ]
            next_json = settings_json_with_clarification_queue(user_settings, next_queue)
            next_json = settings_json_with_pending_clarification(_settings_view(next_json), None)
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": next_json},
            )
            async with _typing(message, bot):
                result = await interaction_service.record_correction(
                    session,
                    user=user,
                    correction_text=text,
                    telegram_message_id=message.message_id,
                    reply_to_message_id=(
                        message.reply_to_message.message_id if message.reply_to_message else None
                    ),
                    target_entry_id=target_entry_id,
                    clarification_context=pending_deferred,
                )
            entry_id = result.entry_id
            should_embed = result.should_embed_entry
            replies = result.replies
            replace_existing_embedding = True
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
                    summary_service=summary_service,
                )
            entry_id = result.entry_id
            should_embed = result.should_embed_entry
            replies = result.replies
            replace_existing_embedding = pending_kind == "correction"
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
        elif manual_pending and not message.photo:
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={
                    "settings_json": settings_json_with_pending_manual_entry(
                        user_settings,
                        _pending_manual_entry_payload(message, text),
                    )
                },
            )
            replies = [
                BotReply(
                    "Оновив непідтверджений текст.\n\n"
                    f"{_manual_entry_confirmation_text(text)}",
                    keyboard="manual_entry_confirm",
                )
            ]
        else:
            open_snapshot = await repo.get_open_snapshot(session, user_id=user.id)
            if open_snapshot is None and not message.photo:
                await repo.update_user_settings(
                    session,
                    user_id=user.id,
                    values={
                        "settings_json": settings_json_with_pending_manual_entry(
                            user_settings,
                            _pending_manual_entry_payload(message, text),
                        )
                    },
                )
                replies = [
                    BotReply(
                        _manual_entry_confirmation_text(text),
                        keyboard="manual_entry_confirm",
                    )
                ]
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

    if direct_response_sent:
        return
    for reply in replies:
        await message.answer(
            reply.text,
            reply_markup=_inline_reply_keyboard(reply.keyboard, options=reply.keyboard_options),
        )
    if should_embed and entry_id and user_id:
        asyncio.create_task(
            _embed_entry_task(
                settings,
                sessionmaker,
                memory_service,
                entry_id,
                user_id,
                replace_existing=replace_existing_embedding,
            )
        )
        asyncio.create_task(
            _maybe_offer_wake_time_task(
                bot=bot,
                settings=settings,
                sessionmaker=sessionmaker,
                entry_id=entry_id,
                user_id=user_id,
                chat_id=message.chat.id,
            )
        )
        asyncio.create_task(
            _maybe_offer_life_context_task(
                bot=bot,
                settings=settings,
                sessionmaker=sessionmaker,
                ai_service=interaction_service.ai,
                user_id=user_id,
                chat_id=message.chat.id,
            )
        )
        asyncio.create_task(
            _maybe_offer_quiet_pause_task(
                bot=bot,
                settings=settings,
                sessionmaker=sessionmaker,
                ai_service=interaction_service.ai,
                entry_id=entry_id,
                user_id=user_id,
                chat_id=message.chat.id,
            )
        )
        asyncio.create_task(
            _maybe_offer_planned_event_task(
                bot=bot,
                settings=settings,
                sessionmaker=sessionmaker,
                entry_id=entry_id,
                user_id=user_id,
                chat_id=message.chat.id,
            )
        )


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


@router.callback_query(F.data.startswith("clarification:skip:"))
async def deferred_clarification_skip_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    item_id = (callback.data or "").rsplit(":", maxsplit=1)[-1]
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        pending = pending_clarification(user_settings)
        if not pending or str(pending.get("id")) != item_id:
            await callback.answer("Це уточнення вже неактуальне", show_alert=True)
            await _clear_inline_keyboard(callback)
            return
        queue = clarification_queue(user_settings)
        related_ids = _clarification_related_ids(pending)
        next_queue = [
            {**item, "status": "skipped", "skipped_at": utc_now().isoformat()}
            if str(item.get("id")) in related_ids
            else item
            for item in queue
        ]
        next_json = settings_json_with_clarification_queue(user_settings, next_queue)
        next_json = settings_json_with_pending_clarification(
            _settings_view(next_json),
            None,
        )
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": next_json},
        )
    await callback.answer("Пропущено")
    await _clear_inline_keyboard(callback)
    await callback.message.answer("Ок, не уточнюю. Це не стало окремим записом.")


@router.callback_query(F.data.startswith("clarification:option:"))
async def deferred_clarification_option_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    interaction_service: InteractionService,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    parts = (callback.data or "").split(":")
    if len(parts) != 4:
        await callback.answer("Не впізнав варіант", show_alert=True)
        return
    item_id = parts[2]
    try:
        option_index = int(parts[3])
    except ValueError:
        await callback.answer("Не впізнав варіант", show_alert=True)
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        pending = pending_clarification(user_settings)
        options = list((pending or {}).get("options") or [])
        target_entry_id = _uuid_or_none(str((pending or {}).get("entry_id") or ""))
        if not pending or str(pending.get("id")) != item_id or target_entry_id is None or not 0 <= option_index < len(options):
            await callback.answer("Це уточнення вже неактуальне", show_alert=True)
            await _clear_inline_keyboard(callback)
            return
        answer = " ".join(str(options[option_index]).split())
        queue = clarification_queue(user_settings)
        related_ids = _clarification_related_ids(pending)
        now = utc_now().isoformat()
        next_queue = [
            {**item, "status": "answered", "answered_at": now, "answer": answer, "answer_source": "option"}
            if str(item.get("id")) in related_ids
            else item
            for item in queue
        ]
        next_json = settings_json_with_clarification_queue(user_settings, next_queue)
        next_json = settings_json_with_pending_clarification(_settings_view(next_json), None)
        await repo.update_user_settings(session, user_id=user.id, values={"settings_json": next_json})
        async with _typing(callback.message):
            result = await interaction_service.record_correction(
                session,
                user=user,
                correction_text=answer,
                telegram_message_id=None,
                reply_to_message_id=None,
                target_entry_id=target_entry_id,
                clarification_context=pending,
            )
    await callback.answer("Записую")
    await _clear_inline_keyboard(callback)
    for reply in result.replies:
        await callback.message.answer(reply.text, reply_markup=_inline_reply_keyboard(reply.keyboard, options=reply.keyboard_options))


@router.callback_query(F.data.startswith("clarification_queue:"))
async def clarification_queue_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    action = callback.data or ""
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_settings = await repo.get_user_settings(session, user.id)

        if action == "clarification_queue:open":
            text = _format_clarification_queue_status(user_settings)
            keyboard = _clarification_queue_menu_keyboard(user_settings)
            await callback.answer()
            await _edit_or_answer_menu(callback, text, keyboard)
            return

        if action == "clarification_queue:next":
            if pending_clarification(user_settings) is not None:
                await callback.answer("Уже є активне уточнення", show_alert=True)
                return
            if pending_input(user_settings) is not None:
                await callback.answer("Спочатку заверши поточний ввід", show_alert=True)
                return
            item = next((item for item in clarification_queue(user_settings) if item.get("status") == "queued"), None)
            if item is None:
                await callback.answer("У черзі немає нових уточнень", show_alert=True)
                text = _format_clarification_queue_status(user_settings)
                keyboard = _clarification_queue_menu_keyboard(user_settings)
                await _edit_or_answer_menu(callback, text, keyboard)
                return
            item = {**item, "status": "active", "delivered_at": utc_now().isoformat(), "delivery_source": "manual"}
            next_queue = [item if other.get("id") == item.get("id") else other for other in clarification_queue(user_settings)]
            next_json = settings_json_with_clarification_queue(user_settings, next_queue)
            next_json = settings_json_with_pending_clarification(_settings_view(next_json), item)
            await repo.update_user_settings(session, user_id=user.id, values={"settings_json": next_json})
            await callback.answer("Поставив уточнення")
            await _clear_inline_keyboard(callback)
            await callback.message.answer(
                str(item.get("question") or "Є одне необов’язкове уточнення до попереднього запису."),
                reply_markup=deferred_clarification_keyboard(item_id=str(item["id"]), options=item.get("options") or []),
            )
            return

        if action == "clarification_queue:skip_all":
            await callback.answer()
            await _edit_or_answer_menu(
                callback,
                "Пропустити всі відкладені уточнення? Це не створить записів і не змінить старі висновки.",
                clarifications_skip_all_confirmation_keyboard(),
            )
            return

        if action == "clarification_queue:skip_all_confirm":
            now_text = utc_now().isoformat()
            next_queue = [
                {**item, "status": "skipped", "skipped_at": now_text, "skip_source": "manual_bulk"}
                if item.get("status") in {"queued", "active"}
                else item
                for item in clarification_queue(user_settings)
            ]
            next_json = settings_json_with_clarification_queue(user_settings, next_queue)
            next_json = settings_json_with_pending_clarification(_settings_view(next_json), None)
            updated = await repo.update_user_settings(session, user_id=user.id, values={"settings_json": next_json})
            await callback.answer("Пропущено")
            await _edit_or_answer_menu(
                callback,
                _format_clarification_queue_status(updated),
                _clarification_queue_menu_keyboard(updated),
            )
            return

    await callback.answer("Не впізнав дію", show_alert=True)


@router.callback_query(F.data.startswith("quiet:"))
async def quiet_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    action = callback.data or ""
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        if action == "quiet:menu":
            text = _quiet_status_text(user_settings, timezone=user.timezone)
            keyboard = quiet_menu_keyboard(active=quiet_is_active(user_settings))
        elif action == "quiet:cancel":
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_quiet_until(user_settings, None)},
            )
            text = "Ок, тиха пауза скасована."
            keyboard = quiet_menu_keyboard(active=False)
        elif action == "quiet:custom":
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_pending_input(user_settings, "quiet_until")},
            )
            await callback.answer()
            await _clear_inline_keyboard(callback)
            await callback.message.answer(
                "До якого часу не турбувати? Можна написати `18:30`, `до 20:00` або `на 3 години`.",
                reply_markup=main_reply_keyboard("Наприклад: до 18:30"),
            )
            return
        elif action == "quiet:offer:no":
            text = "Ок, не ставлю паузу."
            keyboard = None
        elif action.startswith("quiet:set:"):
            preset = action.rsplit(":", maxsplit=1)[1]
            until = _quiet_until_for_preset(preset, timezone=user.timezone)
            user_settings = await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_quiet_until(user_settings, until)},
            )
            text = _quiet_set_text(user_settings, timezone=user.timezone)
            keyboard = quiet_menu_keyboard(active=True)
        else:
            text = "Не впізнав дію паузи."
            keyboard = quiet_menu_keyboard(active=quiet_is_active(user_settings))
    await callback.answer()
    await _edit_or_answer_menu(callback, text, keyboard)


@router.callback_query(F.data.startswith("planned_event:"))
async def planned_event_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    action = callback.data or ""
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        pending = pending_planned_event(user_settings)
        if pending is None:
            text = "Цю пропозицію події вже закрито."
            keyboard = None
        elif action == "planned_event:confirm":
            event, events = confirm_pending_planned_event(user_settings)
            next_json = settings_json_with_planned_events(user_settings, events)
            next_json.pop("pending_planned_event", None)
            await repo.update_user_settings(session, user_id=user.id, values={"settings_json": next_json})
            text = f"Ок, запам'ятав як майбутню подію: {planned_event_text(event or pending)}."
            keyboard = None
        elif action == "planned_event:clarify":
            next_json = settings_json_with_pending_input(user_settings, "planned_event_clarify")
            await repo.update_user_settings(session, user_id=user.id, values={"settings_json": next_json})
            text = "Напиши, як це краще запам'ятати: що це за подія і, якщо знаєш, коли приблизно."
            keyboard = None
        elif action == "planned_event:ignore":
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_pending_planned_event(user_settings, None)},
            )
            text = "Ок, ігнорую цю пропозицію."
            keyboard = None
        else:
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_pending_planned_event(user_settings, None)},
            )
            text = "Ок, не зберігаю це як майбутню подію."
            keyboard = None
    await callback.answer()
    await _clear_inline_keyboard(callback)
    await callback.message.answer(text, reply_markup=keyboard)


@router.callback_query(F.data == "wake_time:skip")
async def wake_time_skip_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        record = skipped_wake_time_record(timezone=user.timezone, now=utc_now())
        next_json = settings_json_with_wake_time_records(
            user_settings,
            append_wake_time_record(user_settings, record),
        )
        next_json = settings_json_without_pending_input(_settings_view(next_json))
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": next_json},
        )
    await callback.answer("Ок")
    await _clear_inline_keyboard(callback)
    await callback.message.answer("Ок, не уточнюю час пробудження.")


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
            reply_markup=_inline_reply_keyboard(reply.keyboard, options=reply.keyboard_options),
        )
    if should_embed and entry_id and user_id:
        asyncio.create_task(_embed_entry_task(settings, sessionmaker, memory_service, entry_id, user_id))


@router.callback_query((F.data == "sleep:confirm") | (F.data == "day:sleep"))
async def sleep_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    await callback.answer()
    await _clear_inline_keyboard(callback)
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_with_pending_input(user_settings, "sleep_reflection")},
        )
    await callback.message.answer(_sleep_reflection_prompt(), reply_markup=sleep_reflection_keyboard())


@router.callback_query(F.data.startswith("sleep:reflect:"))
async def sleep_reflection_callback_handler(
    callback: CallbackQuery,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    summary_service: SummaryService,
) -> None:
    if not await _allowed_callback(callback, settings):
        return
    action = (callback.data or "").rsplit(":", maxsplit=1)[-1]
    if action == "custom":
        await callback.answer()
        await _clear_inline_keyboard(callback)
        await callback.message.answer("Напиши одним повідомленням, як би ти сам оцінив цей день.")
        return

    reflection = _sleep_reflection_text(action)
    if reflection is None:
        await callback.answer("Не впізнав оцінку", show_alert=True)
        return

    await callback.answer("Закриваю день")
    await _clear_inline_keyboard(callback)
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        user_settings = await repo.get_user_settings(session, user.id)
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_without_pending_input(user_settings)},
        )
        async with _typing(callback.message):
            summary = await summary_service.close_today_with_summary(
                session,
                user=user,
                day_reflection=reflection,
                day_reflection_kind=action,
            )
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
    await callback.answer("Генерую підсумок")
    async with sessionmaker() as session, session.begin():
        user = await _get_or_create_callback_user(session, callback, settings)
        async with _typing(callback.message):
            summary = await summary_service.generate_today_summary(session, user=user)
    await callback.message.answer(summary.short_text, reply_markup=summary_detail_keyboard(summary_id=str(summary.id)))


@router.callback_query(F.data.startswith("archive:export"))
async def export_callback_handler(callback: CallbackQuery, settings: Settings) -> None:
    if not await _allowed_callback(callback, settings):
        return
    await callback.answer("Готую експорт")
    export_format, extension, prefix, caption = _archive_export_options(callback.data or "")
    output = Path("./data") / f"{prefix}-{callback.from_user.id}.{extension}"
    await export_user_archive(settings, callback.from_user.id, output, format=export_format)
    await callback.message.answer_document(FSInputFile(output), caption=caption)


@router.callback_query()
async def unknown_callback_handler(callback: CallbackQuery, settings: Settings) -> None:
    if not await _allowed_callback(callback, settings):
        return
    logger.warning("Unhandled callback data", extra={"callback_data": callback.data})
    await callback.answer("Ця кнопка вже неактуальна", show_alert=True)
    await _clear_inline_keyboard(callback)


async def _handle_pending_input(
    *,
    session: AsyncSession,
    user,
    user_settings: UserSettings,
    pending_kind: str,
    text: str,
    message: Message,
    interaction_service: InteractionService,
    summary_service: SummaryService,
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
    if pending_kind == "quiet_until":
        until = _parse_quiet_until_text(text, timezone=user.timezone)
        if until is None:
            return InteractionResult(
                replies=[BotReply("Не впізнав час. Напиши, наприклад: `18:30`, `до 20:00` або `на 3 години`.")],
                snapshot_closed=True,
            )
        next_settings_json = settings_json_with_quiet_until(user_settings, until)
        temp_settings = await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_without_pending_input(_settings_view(next_settings_json))},
        )
        return InteractionResult(
            replies=[BotReply(_quiet_set_text(temp_settings, timezone=user.timezone))],
            snapshot_closed=True,
        )
    if pending_kind == "life_context_free_answer":
        return await _handle_life_context_free_answer(
            session,
            user=user,
            user_settings=user_settings,
            text=text,
            ai_service=interaction_service.ai,
        )
    if pending_kind == "sleep_reflection":
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_without_pending_input(user_settings)},
        )
        summary = await summary_service.close_today_with_summary(
            session,
            user=user,
            day_reflection=text,
            day_reflection_kind="free",
        )
        return InteractionResult(
            replies=[BotReply(summary.short_text, keyboard=f"summary_detail:{summary.id}")],
            snapshot_closed=True,
        )
    if pending_kind == "planned_event_clarify":
        candidate = detect_planned_event_candidate(text, timezone=user.timezone) or {
            **(pending_planned_event(user_settings) or {}),
            "source_text": " ".join(text.split())[:240],
        }
        if not candidate.get("title"):
            candidate["title"] = "подія"
        next_json = settings_json_with_pending_planned_event(user_settings, candidate)
        next_json = settings_json_without_pending_input(_settings_view(next_json))
        await repo.update_user_settings(session, user_id=user.id, values={"settings_json": next_json})
        return InteractionResult(
            replies=[
                BotReply(
                    f"Уточнив кандидат події: {planned_event_text(candidate)}. Запам'ятати?",
                    keyboard="planned_event_offer",
                )
            ],
            snapshot_closed=True,
        )
    if pending_kind == "wake_time":
        record = parse_wake_time_text(text, timezone=user.timezone, now=utc_now())
        next_json = settings_json_with_wake_time_records(
            user_settings,
            append_wake_time_record(user_settings, record),
        )
        next_json = settings_json_without_pending_input(_settings_view(next_json))
        await repo.update_user_settings(session, user_id=user.id, values={"settings_json": next_json})
        if record.get("estimated_woke_at"):
            local_time = datetime.fromisoformat(record["estimated_woke_at"]).astimezone(zoneinfo(user.timezone))
            reply = f"Записав приблизний час пробудження: {local_time.strftime('%H:%M')}."
        else:
            reply = "Записав приблизний час пробудження словами."
        return InteractionResult(replies=[BotReply(reply)], snapshot_closed=True)
    if pending_kind == "correction":
        target_entry_id = _uuid_or_none(pending_correction_entry_id(user_settings))
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
            target_entry_id=target_entry_id,
        )
    if pending_kind in {"voice_transcript", "voice_transcript_fix"}:
        return await _confirm_pending_voice_transcript(
            session=session,
            user=user,
            user_settings=user_settings,
            interaction_service=interaction_service,
            summary_service=summary_service,
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
    summary_service: SummaryService | None = None,
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

    if pending.get("target_pending_kind") == "clarification_queue":
        deferred = pending_clarification(user_settings)
        target_entry_id = _uuid_or_none(str((deferred or {}).get("entry_id") or ""))
        if deferred is None or target_entry_id is None:
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_without_pending_voice_transcript(user_settings)},
            )
            return InteractionResult(replies=[BotReply("Це уточнення вже неактуальне.")], snapshot_closed=True)
        queue = clarification_queue(user_settings)
        related_ids = _clarification_related_ids(deferred)
        next_queue = [
            {**item, "status": "answered", "answered_at": utc_now().isoformat()}
            if str(item.get("id")) in related_ids
            else item
            for item in queue
        ]
        next_json = settings_json_with_clarification_queue(
            user_settings,
            next_queue,
        )
        next_json = settings_json_without_pending_voice_transcript(_settings_view(next_json))
        next_json = settings_json_with_pending_clarification(_settings_view(next_json), None)
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": next_json},
        )
        return await interaction_service.record_correction(
            session,
            user=user,
            correction_text=text,
            telegram_message_id=_pending_int(pending.get("telegram_message_id")) or fallback_message_id,
            reply_to_message_id=(
                _pending_int(pending.get("reply_to_message_id")) or fallback_reply_to_message_id
            ),
            target_entry_id=target_entry_id,
            clarification_context=deferred,
        )
    if pending.get("target_pending_kind") == "life_context_free_answer":
        return await _handle_life_context_free_answer(
            session,
            user=user,
            user_settings=user_settings,
            text=text,
            ai_service=interaction_service.ai,
            clear_pending_voice=True,
        )
    if pending.get("target_pending_kind") == "sleep_reflection":
        if summary_service is None:
            return InteractionResult(
                replies=[BotReply("Не можу закрити день із голосового без сервісу підсумків.")],
                snapshot_closed=True,
            )
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_without_pending_voice_transcript(user_settings)},
        )
        summary = await summary_service.close_today_with_summary(
            session,
            user=user,
            day_reflection=text,
            day_reflection_kind="voice",
        )
        return InteractionResult(
            replies=[BotReply(summary.short_text, keyboard=f"summary_detail:{summary.id}")],
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


async def _handle_life_context_free_answer(
    session: AsyncSession,
    *,
    user,
    user_settings: UserSettings,
    text: str,
    ai_service: AIService,
    clear_pending_voice: bool = False,
) -> InteractionResult:
    settings_json = (
        settings_json_without_pending_voice_transcript(user_settings)
        if clear_pending_voice
        else settings_json_without_pending_input(user_settings)
    )
    user_settings = await repo.update_user_settings(
        session,
        user_id=user.id,
        values={"settings_json": settings_json},
    )
    answer_text, next_review = await answer_life_context_candidate(
        session,
        user=user,
        user_settings=user_settings,
        answer=text,
        answer_kind="free",
        ai_service=ai_service,
    )
    replies = [BotReply(answer_text)]
    if next_review:
        next_candidate = current_life_context_candidate(next_review) or {}
        keyboard = (
            "life_context_current"
            if next_candidate.get("question_type") == "confirm"
            else "life_context_open"
        )
        replies.append(
            BotReply(
                format_life_context_question(next_review),
                keyboard=keyboard,
            )
        )
    return InteractionResult(replies=replies, snapshot_closed=True)


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
    target_pending_kind: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
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
    if target_pending_kind:
        payload["target_pending_kind"] = target_pending_kind
    return payload


def _voice_target_pending_kind(pending_kind: str | None) -> str | None:
    return pending_kind if pending_kind in {"life_context_free_answer", "sleep_reflection", "clarification_queue"} else None


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
    *,
    replace_existing: bool = False,
) -> None:
    try:
        async with sessionmaker() as session, session.begin():
            entry = await repo.get_entry(session, entry_id=entry_id)
            if entry is None:
                return
            await memory_service.embed_entry(
                session,
                entry=entry,
                user_id=user_id,
                replace_existing=replace_existing,
            )
    except Exception:
        logger.exception("Background embedding task failed", extra={"entry_id": str(entry_id)})


async def _mark_summary_delivered(
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    summary_id: UUID,
) -> None:
    async with sessionmaker() as session, session.begin():
        await repo.mark_summary_delivered(session, summary_id=summary_id, delivered_at=utc_now())


async def _maybe_offer_life_context_task(
    *,
    bot: Bot,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai_service: AIService,
    user_id: UUID,
    chat_id: int,
) -> None:
    try:
        async with sessionmaker() as session, session.begin():
            user = await session.get(User, user_id)
            if user is None:
                return
            if settings.telegram_allowed_user_ids and user.telegram_user_id not in settings.telegram_allowed_user_ids:
                return
            user_settings = await repo.get_user_settings(session, user.id)
            result = await maybe_start_auto_life_context_review(
                session,
                user=user,
                user_settings=user_settings,
                ai_service=ai_service,
            )
            if result is None:
                return
            lead_text, _review = result
        await bot.send_message(
            chat_id=chat_id,
            text=f"{lead_text}\n\nМожемо швидко перевірити кілька питань. Якщо не зараз — просто відклади.",
            reply_markup=life_context_offer_keyboard(),
        )
    except Exception:
        logger.exception("Background life context offer task failed", extra={"user_id": str(user_id)})


async def _maybe_offer_wake_time_task(
    *,
    bot: Bot,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    entry_id: UUID,
    user_id: UUID,
    chat_id: int,
) -> None:
    try:
        async with sessionmaker() as session, session.begin():
            user = await session.get(User, user_id)
            entry = await repo.get_entry(session, entry_id=entry_id)
            if user is None or entry is None or entry.day_id is None:
                return
            if settings.telegram_allowed_user_ids and user.telegram_user_id not in settings.telegram_allowed_user_ids:
                return
            day = await repo.get_day(session, day_id=entry.day_id)
            if day is None:
                return
            user_settings = await repo.get_user_settings(session, user.id)
            if pending_input(user_settings):
                return
            day_entries = await repo.list_day_entries(session, day_id=day.id)
            if not should_offer_wake_time_question(
                entries=day_entries,
                current_text=entry.raw_text or "",
                user_settings=user_settings,
                local_date=day.local_date.isoformat(),
            ):
                return
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_pending_input(user_settings, "wake_time")},
            )
        await bot.send_message(
            chat_id=chat_id,
            text=(
                "Якщо хочеш, уточни ще одне: коли приблизно прокинувся?\n"
                "Можна дуже приблизно: `о 9`, `десь о 9:30`, `пів години тому`."
            ),
            reply_markup=wake_time_keyboard(),
        )
    except Exception:
        logger.exception("Background wake time offer task failed", extra={"user_id": str(user_id)})


async def _maybe_offer_quiet_pause_task(
    *,
    bot: Bot,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai_service: AIService,
    entry_id: UUID,
    user_id: UUID,
    chat_id: int,
) -> None:
    try:
        async with sessionmaker() as session, session.begin():
            user = await session.get(User, user_id)
            entry = await repo.get_entry(session, entry_id=entry_id)
            if user is None or entry is None:
                return
            if settings.telegram_allowed_user_ids and user.telegram_user_id not in settings.telegram_allowed_user_ids:
                return
            user_settings = await repo.get_user_settings(session, user.id)
            if not context_quiet_enabled(user_settings) or quiet_is_active(user_settings):
                return
            now = datetime.now(UTC)
            last_check = context_quiet_last_check_at(user_settings)
            if last_check and now - last_check < timedelta(minutes=90):
                return
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_context_quiet_last_check(user_settings, now)},
            )
            recent_entries = await repo.get_recent_entries(session, user_id=user.id, limit=3)
            suggestion, _run_id = await ai_service.suggest_quiet_pause(
                session,
                user_id=user.id,
                context={
                    "latest_entry": {
                        "raw_text": entry.raw_text,
                        "source": entry.source,
                        "created_at": entry.created_at.isoformat() if entry.created_at else None,
                    },
                    "recent_entries": [
                        {
                            "raw_text": item.raw_text,
                            "source": item.source,
                            "created_at": item.created_at.isoformat() if item.created_at else None,
                        }
                        for item in recent_entries
                    ],
                    "life_context": [
                        {
                            "label": item.get("label"),
                            "value": item.get("answer") or item.get("value") or item.get("hypothesis"),
                        }
                        for item in life_context_items(user_settings)[-8:]
                        if isinstance(item, dict)
                    ],
                },
            )
            if not suggestion.should_offer or suggestion.confidence < 0.65:
                return
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_context_quiet_last_offer(user_settings, now)},
            )
            text = suggestion.message or "Схоже, зараз може бути незручно відповідати. Поставити тиху паузу?"
        await bot.send_message(chat_id=chat_id, text=text, reply_markup=quiet_offer_keyboard())
    except Exception:
        logger.exception("Background quiet pause offer task failed", extra={"user_id": str(user_id)})


async def _maybe_offer_planned_event_task(
    *,
    bot: Bot,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    entry_id: UUID,
    user_id: UUID,
    chat_id: int,
) -> None:
    try:
        async with sessionmaker() as session, session.begin():
            user = await session.get(User, user_id)
            entry = await repo.get_entry(session, entry_id=entry_id)
            if user is None or entry is None:
                return
            if settings.telegram_allowed_user_ids and user.telegram_user_id not in settings.telegram_allowed_user_ids:
                return
            user_settings = await repo.get_user_settings(session, user.id)
            if pending_planned_event(user_settings) is not None:
                return
            candidate = detect_planned_event_candidate(entry.raw_text or "", timezone=user.timezone)
            if candidate is None:
                return
            existing_titles = {
                str(event.get("title") or "").strip().lower()
                for event in planned_event_context(user_settings)
            }
            if str(candidate.get("title") or "").strip().lower() in existing_titles:
                return
            await repo.update_user_settings(
                session,
                user_id=user.id,
                values={"settings_json": settings_json_with_pending_planned_event(user_settings, candidate)},
            )
            text = (
                "Схоже, попереду є подія.\n\n"
                f"{planned_event_text(candidate)}\n\n"
                "Запам'ятати це, щоб не питати про неї невчасно?"
            )
        await bot.send_message(chat_id=chat_id, text=text, reply_markup=planned_event_offer_keyboard())
    except Exception:
        logger.exception("Background planned event offer task failed", extra={"user_id": str(user_id)})


async def _reanalyze_features_task(
    *,
    bot: Bot,
    chat_id: int,
    telegram_user_id: int,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai_service: AIService,
    scope: str,
    limit: int | None,
    start_date: date | None,
    end_date: date | None,
) -> None:
    try:
        result = await guided_reanalyze_entry_features(
            settings=settings,
            ai_service=ai_service,
            sessionmaker=sessionmaker,
            telegram_user_id=telegram_user_id,
            scope=scope,  # type: ignore[arg-type]
            limit=limit,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        logger.exception(
            "Background feature reanalysis task failed",
            extra={"telegram_user_id": telegram_user_id, "scope": scope, "limit": limit},
        )
        await bot.send_message(
            chat_id,
            "Переаналіз не завершився через помилку.\n\n"
            f"Технічно: {_truncate_text(str(exc), 600)}",
        )
        return

    await bot.send_message(
        chat_id,
        "Переаналіз AI завершено.\n"
        f"Вибрано записів: {result.selected}\n"
        f"Оброблено: {result.processed}\n"
        f"Пропущено зниклих: {result.skipped_missing}\n\n"
        "Що змінилося:\n"
        f"- графік настрою: {result.before.mood_points} -> {result.after.mood_points}\n"
        f"- графік енергії: {result.before.energy_points} -> {result.after.energy_points}\n"
        f"- спостережені емоційні моменти: {result.before.observed_emotion_points} -> {result.after.observed_emotion_points}\n"
        f"- оновлено трактувань: {result.changed}\n\n"
        "Тепер метрики й графіки для цих записів будуть будуватися на новішому аналізі.",
    )


async def _rebuild_memory_task(
    *,
    bot: Bot,
    chat_id: int,
    telegram_user_id: int,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai_service: AIService,
    limit: int,
) -> None:
    try:
        processed = await backfill_entry_embeddings(
            settings=settings,
            ai_service=ai_service,
            sessionmaker=sessionmaker,
            telegram_user_id=telegram_user_id,
            limit=limit,
            force=True,
        )
    except Exception as exc:
        logger.exception(
            "Background memory rebuild task failed",
            extra={"telegram_user_id": telegram_user_id, "limit": limit},
        )
        await bot.send_message(
            chat_id,
            "Перебудова пам’яті не завершилася через помилку.\n\n"
            f"Технічно: {_truncate_text(str(exc), 600)}",
        )
        return

    await bot.send_message(
        chat_id,
        "Перебудова пам’яті завершена.\n"
        f"Оброблено записів: {processed}\n\n"
        "Для цих записів оновлено contextual memory capsules, embeddings і graph-зв’язки.",
    )


def _format_memory_maintenance_result(result: MemoryGraphMaintenanceResult) -> str:
    return (
        "Обслуговування графа завершено.\n\n"
        f"Перевірено вузлів: {result.nodes_checked}\n"
        f"Можливих пар-дублікатів: {result.duplicate_pairs_found}\n"
        f"Вузлів із позначками `possible_duplicates`: {result.nodes_marked_candidate}\n\n"
        "Послаблення старої пам’яті:\n"
        f"- оновлено вузлів: {result.decay.nodes_updated}\n"
        f"- оновлено зв’язків: {result.decay.edges_updated}\n"
        f"- позначено `stale` вузлів: {result.decay.nodes_staled}\n"
        f"- позначено `stale` зв’язків: {result.decay.edges_staled}\n\n"
        "Дублікатів тут не зливаю і не видаляю: локальний прохід лише позначає кандидатів, "
        "щоб наступна AI-ревізія або підтвердження могли вирішити обережніше."
    )


def _format_memory_ai_review_result(
    result: MemoryGraphAIReviewResult,
    *,
    maintenance: MemoryGraphMaintenanceResult,
) -> str:
    if result.pairs_selected == 0:
        return (
            "AI-ревізія графа не знайшла пар для перевірки.\n\n"
            f"Перед цим локально перевірено вузлів: {maintenance.nodes_checked}. "
            f"Можливих пар-дублікатів: {maintenance.duplicate_pairs_found}."
        )
    return (
        "AI-ревізія графа завершена.\n\n"
        f"Вибрано пар: {result.pairs_selected}\n"
        f"Отримано рішень: {result.decisions_received}\n"
        f"Додано альтернативних назв: {result.aliases_added}\n"
        f"Позначено слабких дублікатів як `stale`: {result.nodes_staled_as_duplicate}\n"
        f"Позначено як різні речі: {result.pairs_marked_separate}\n"
        f"Залишено для підтвердження: {result.pairs_needing_confirmation}\n\n"
        "Це обережний прохід: невпевнені або підтверджені пари не зливаються автоматично."
    )


def _format_clarification_queue_status(user_settings: UserSettings) -> str:
    queue = clarification_queue(user_settings)
    pending = pending_clarification(user_settings)
    counts = _clarification_queue_counts(queue)
    lines = [
        "Відкладені уточнення.",
        "",
        f"Очікують: {counts['queued']}",
        f"Активні: {counts['active']}",
        f"Відповіді отримано: {counts['answered']}",
        f"Пропущено: {counts['skipped']}",
    ]
    if pending:
        lines.extend(["", "Зараз активне:", _clarification_item_preview(pending)])
    else:
        next_item = next((item for item in queue if item.get("status") == "queued"), None)
        if next_item:
            lines.extend(["", "Наступне:", _clarification_item_preview(next_item)])
        else:
            lines.extend(["", "Немає уточнень, які чекають відповіді."])
    lines.extend(["", "Пропуск не створює окремий запис і не змушує AI щось домислювати."])
    return "\n".join(lines)


def _clarification_queue_menu_keyboard(user_settings: UserSettings):
    queue = clarification_queue(user_settings)
    pending = pending_clarification(user_settings)
    return clarifications_menu_keyboard(
        has_queued=any(item.get("status") == "queued" for item in queue),
        has_pending=pending is not None,
        has_clearable=any(item.get("status") in {"queued", "active"} for item in queue) or pending is not None,
    )


def _clarification_queue_counts(queue: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"queued": 0, "active": 0, "answered": 0, "skipped": 0}
    for item in queue:
        status = str(item.get("status") or "queued")
        if status in counts:
            counts[status] += 1
    return counts


def _clarification_item_preview(item: dict[str, Any]) -> str:
    question = " ".join(str(item.get("question") or "Уточнення без тексту").split())
    reason = " ".join(str(item.get("reason") or "").split())
    preview = _truncate_text(question, 220)
    if reason:
        preview += f"\nПричина: {_truncate_text(reason, 160)}"
    return preview


def _clarification_related_ids(item: dict[str, Any]) -> set[str]:
    ids = {str(item.get("id") or "")}
    ids.update(str(value) for value in item.get("grouped_item_ids") or [] if value)
    return {value for value in ids if value}


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


async def _clear_pending_post_entry_followup(session, *, user, entry_id) -> None:
    user_settings = await repo.get_user_settings(session, user.id)
    pending = pending_post_entry_followup(user_settings)
    if pending is None:
        return
    if str(pending.get("entry_id") or "") != str(entry_id):
        return
    await repo.update_user_settings(
        session,
        user_id=user.id,
        values={"settings_json": settings_json_with_pending_post_entry_followup(user_settings, None)},
    )


async def _set_pending_post_entry_followup_from_replies(
    session,
    *,
    user,
    entry_id,
    replies: list[BotReply],
) -> None:
    keyboard = replies[-1].keyboard if replies else None
    if keyboard and keyboard.startswith("metric_score:"):
        kind = "metric"
    elif keyboard and keyboard.startswith("emotion_calibration:"):
        kind = "emotion"
    else:
        return
    user_settings = await repo.get_user_settings(session, user.id)
    await repo.update_user_settings(
        session,
        user_id=user.id,
        values={
            "settings_json": settings_json_with_pending_post_entry_followup(
                user_settings,
                {"entry_id": str(entry_id), "kind": kind, "created_at": utc_now().isoformat()},
            )
        },
    )


def _inline_reply_keyboard(kind: str | None, *, options: Sequence[str] = ()):
    if kind == "snapshot_control":
        return snapshot_clarification_keyboard()
    if kind == "correction":
        return correction_keyboard()
    if kind and kind.startswith("correction:"):
        entry_id = kind.split(":", maxsplit=1)[1]
        return correction_keyboard(entry_id=entry_id)
    if kind and kind.startswith("interpretation:"):
        entry_id = kind.split(":", maxsplit=1)[1]
        return interpretation_keyboard(entry_id=entry_id)
    if kind == "voice_transcript":
        return voice_transcription_keyboard()
    if kind == "manual_entry_confirm":
        return manual_entry_confirmation_keyboard()
    if kind == "planned_event_offer":
        return planned_event_offer_keyboard()
    if kind == "wake_time":
        return wake_time_keyboard()
    if kind and kind.startswith("metric_score:"):
        _, entry_id, metric = kind.split(":", maxsplit=2)
        return metric_score_keyboard(entry_id=entry_id, metric=metric)
    if kind and kind.startswith("metric_score_with_correction:"):
        _, entry_id, metric = kind.split(":", maxsplit=2)
        return metric_score_keyboard(entry_id=entry_id, metric=metric, include_correction=True)
    if kind and kind.startswith("emotion_calibration:"):
        entry_id = kind.split(":", maxsplit=1)[1]
        return emotion_calibration_keyboard(entry_id=entry_id)
    if kind and kind.startswith("clarification:"):
        item_id = kind.split(":", maxsplit=1)[1]
        return deferred_clarification_keyboard(item_id=item_id, options=options)
    if kind and kind.startswith("emotion_calibration_with_correction:"):
        entry_id = kind.split(":", maxsplit=1)[1]
        return emotion_calibration_keyboard(entry_id=entry_id, include_correction=True)
    if kind == "sleep_confirm":
        return sleep_confirmation_keyboard()
    if kind and kind.startswith("summary_detail:"):
        summary_id = kind.split(":", maxsplit=1)[1]
        return summary_detail_keyboard(summary_id=summary_id)
    if kind == "life_context_continue":
        return life_context_continue_keyboard()
    if kind == "life_context_current":
        return life_context_current_question_keyboard()
    if kind == "life_context_open":
        return life_context_open_question_keyboard()
    return None


def _sleep_reflection_prompt() -> str:
    return (
        "Перед тим як закрити день: як би ти сам оцінив цей день?\n\n"
        "Можна натиснути варіант або написати своїми словами."
    )


def _sleep_reflection_text(action: str) -> str | None:
    return {
        "hard": "День був важкий.",
        "mixed": "День був змішаний.",
        "okay": "День був нормальний.",
        "good": "День був добрий.",
        "skip": "",
    }.get(action)


def _correction_target_from_callback(callback_data: str) -> UUID | None:
    parts = callback_data.split(":")
    if len(parts) != 3:
        return None
    return _uuid_or_none(parts[2])


def _parse_metric_callback(data: str) -> tuple[UUID, str, str] | None:
    parts = data.split(":")
    if len(parts) != 4 or parts[0] != "metric":
        return None
    entry_id = _uuid_or_none(parts[1])
    metric = parts[2]
    score_text = parts[3]
    if entry_id is None or metric not in {"mood", "energy"}:
        return None
    return entry_id, metric, score_text


def _parse_metric_start_callback(data: str) -> tuple[UUID, str] | None:
    parts = data.split(":")
    if len(parts) != 4 or parts[0] != "metric" or parts[1] != "start":
        return None
    entry_id = _uuid_or_none(parts[2])
    metric = parts[3]
    if entry_id is None or metric not in {"mood", "energy"}:
        return None
    return entry_id, metric


def _metric_prompt_text(metric: str) -> str:
    if metric == "energy":
        return "Скільки було енергії приблизно від 0 до 10?"
    return "Як би ти оцінив настрій приблизно від 0 до 10?"


def _parse_emotion_callback(data: str) -> tuple[str, UUID, list[str]] | None:
    parts = data.split(":")
    if len(parts) < 3 or parts[0] != "emotion":
        return None
    action = parts[1]
    if action not in {"t", "save", "custom", "skip"}:
        return None
    entry_id = _uuid_or_none(parts[2])
    if entry_id is None:
        return None
    selected = _emotion_labels_from_indexes(parts[3] if len(parts) >= 4 else "")
    action_name = {"t": "toggle"}.get(action, action)
    return action_name, entry_id, selected


def _parse_emotion_intensity_callback(data: str) -> tuple[UUID, list[str], str] | None:
    parts = data.split(":")
    if len(parts) != 5 or parts[0] != "emotion" or parts[1] not in {"i", "intensity"}:
        return None
    entry_id = _uuid_or_none(parts[2])
    if entry_id is None:
        return None
    selected = _emotion_labels_from_indexes(parts[3])
    intensity_level = parts[4]
    if intensity_level not in EMOTION_INTENSITY_LEVELS:
        return None
    return entry_id, selected, intensity_level


def _parse_per_emotion_intensity_callback(
    data: str,
) -> tuple[str, UUID, list[str], list[str], str, int, str] | None:
    parts = data.split(":")
    if len(parts) not in {7, 8} or parts[:2] not in (["e", "d"], ["e", "x"], ["e", "n"], ["e", "s"]):
        return None
    action_code = parts[1]
    entry_id = _uuid_or_none(parts[2])
    emotions = _emotion_labels_from_indexes(parts[3])
    level_codes = parts[4]
    time_scope = {"c": "current", "n": "mentioned_not_felt"}.get(parts[5])
    try:
        position = int(parts[6])
    except ValueError:
        return None
    code_to_level = {"t": "trace", "m": "mild", "d": "moderate", "s": "strong", "o": "overwhelming", "u": "unclear"}
    intensity_levels = [code_to_level.get(code) for code in level_codes]
    value = code_to_level.get(parts[7]) if len(parts) == 8 else ""
    action = {"s": "set", "n": "navigate", "x": "scope", "d": "done"}.get(action_code)
    if (
        entry_id is None
        or not emotions
        or len(emotions) > 8
        or len(intensity_levels) != len(emotions)
        or any(level is None for level in intensity_levels)
        or time_scope is None
        or not 0 <= position < len(emotions)
        or action is None
        or (action == "set" and value is None)
        or (action == "navigate" and len(parts) != 8)
        or (action in {"done", "scope"} and len(parts) != 7)
    ):
        return None
    if action == "navigate":
        try:
            position = int(parts[7])
        except ValueError:
            return None
        if not 0 <= position < len(emotions):
            return None
        value = ""
    return action, entry_id, emotions, [str(level) for level in intensity_levels], time_scope, position, value or ""


def _emotion_intensity_prompt(
    emotions: list[str],
    intensity_levels: list[str],
    position: int,
    time_scope: str,
) -> str:
    current = emotions[position]
    progress = f"{position + 1}/{len(emotions)}"
    if time_scope == "mentioned_not_felt":
        scope_note = "\n\nПозначено: це були згадані або описані емоції, не поточний стан цього моменту."
    else:
        scope_note = ""
    known = [
        f"{emotion} — {_emotion_intensity_level_label(level)}"
        for emotion, level in zip(emotions, intensity_levels, strict=True)
        if level != "unclear"
    ]
    saved = f"\nВже вибрано: {'; '.join(known)}." if known else ""
    return (
        f"Сила емоції {progress}: {current.capitalize()}. Наскільки вона була виражена?"
        f"{saved}{scope_note}\n\nЯкщо силу важко оцінити, можна просто натиснути «Зберегти»."
    )


def _emotion_intensity_level_label(level: str) -> str:
    return {
        "trace": "ледь фоном",
        "mild": "слабко",
        "moderate": "помірно",
        "strong": "сильно",
        "overwhelming": "дуже сильно",
        "unclear": "неясно",
    }.get(level, "неясно")


def _emotion_labels_from_indexes(value: str) -> list[str]:
    labels: list[str] = []
    if value.startswith("h"):
        try:
            mask = int(value[1:] or "0", 16)
        except ValueError:
            return []
        for index, label in enumerate(EMOTION_CALIBRATION_OPTIONS):
            if mask & (1 << index):
                labels.append(label)
        return labels
    for chunk in value.split(","):
        if not chunk:
            continue
        try:
            index = int(chunk)
        except ValueError:
            continue
        if 0 <= index < len(EMOTION_CALIBRATION_OPTIONS):
            labels.append(EMOTION_CALIBRATION_OPTIONS[index])
    return labels


def _life_context_answer_kind(callback_data: str) -> str:
    parts = callback_data.split(":")
    if len(parts) >= 4 and parts[2] == "option":
        return "option"
    return parts[-1] if parts else ""


def _life_context_answer_from_callback(callback_data: str, candidate: dict) -> str:
    parts = callback_data.split(":")
    if len(parts) >= 4 and parts[2] == "option":
        try:
            index = int(parts[3])
        except ValueError:
            return ""
        options = [str(option) for option in candidate.get("options") or []]
        if 0 <= index < len(options):
            return options[index]
        return ""
    action = parts[-1] if parts else ""
    return {
        "yes": "Так, це правильно.",
        "no": "Ні, це не варто так запам’ятовувати.",
        "skip": "",
        "stop": "",
    }.get(action, "")


def _without_pending(settings_json: dict) -> dict:
    updated = dict(settings_json)
    updated.pop("pending_input", None)
    updated.pop("pending_correction_entry_id", None)
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


def _pending_manual_entry_payload(message: Message, text: str) -> dict[str, object]:
    return {
        "text": text,
        "telegram_message_id": message.message_id,
        "reply_to_message_id": message.reply_to_message.message_id if message.reply_to_message else None,
    }


def _manual_entry_confirmation_text(text: str) -> str:
    return (
        "Ти написав це поза відкритим зрізом. Що зробити?\n\n"
        f"{_voice_transcription_preview(text, limit=500)}"
    )


def _truncate_setting(value: str, limit: int = 180) -> str:
    return _truncate_text(value, limit)


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _is_sleep_marker_text(text: str) -> bool:
    normalized = " ".join(text.lower().split()).strip(" \t\r\n.,!?…")
    return normalized == SLEEP_MARKER_TEXT


def _format_entry_management_view(entries: list[Entry], *, timezone: str) -> str:
    if not entries:
        return "У цьому дні немає записів для керування."
    lines = [
        "Керування записами",
        "",
        "Обери запис нижче, якщо треба видалити його з щоденника.",
        "",
    ]
    for index, entry in enumerate(entries, start=1):
        lines.append(f"{index}. {_format_entry_delete_preview(entry, timezone=timezone, limit=120)}")
    return "\n".join(lines)


def _entry_management_buttons(entries: list[Entry], *, timezone: str) -> list[tuple[str, str]]:
    return [
        (str(entry.id), f"{index}. {_format_entry_delete_preview(entry, timezone=timezone, limit=42)}")
        for index, entry in enumerate(entries, start=1)
    ]


def _format_entry_delete_preview(entry: Entry, *, timezone: str, limit: int = 500) -> str:
    timestamp = entry.local_timestamp or entry.created_at
    time_text = "??:??"
    if timestamp is not None:
        time_text = timestamp.astimezone(zoneinfo(timezone)).strftime("%H:%M")
    text = " ".join((entry.raw_text or "[без тексту]").split())
    return f"{time_text} - {_truncate_text(text, limit)}"


async def _format_day_detail_section(
    session: AsyncSession, *, user: User, day: Day, section: str
) -> tuple[str, bytes | None, bytes | None, bytes | None, list[PhotoMoment]]:
    chart = None
    emotion_chart = None
    spectrum_chart = None
    moments: list[PhotoMoment] = []
    day_title = day.local_date.isoformat()
    if section == "raw":
        text = await format_raw_entries_for_day(session, user=user, day=day, title=day_title)
    elif section == "timeline":
        text = await format_day_view(session, user=user, day=day, limit=30)
    elif section == "metrics":
        text = await format_metrics_for_day(session, user=user, day=day, title=day_title)
        chart = await build_metrics_chart_png_for_day(session, user=user, day=day)
        emotion_chart = await build_emotion_timeline_png_for_day(session, user=user, day=day)
        spectrum_chart = await build_affect_spectrum_png_for_day(session, user=user, day=day)
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
    return text, chart, emotion_chart, spectrum_chart, moments


def _turning_point_labels(points, *, timezone: str) -> list[str]:
    return [
        f"{index}. {_entry_time_label(point.entry, timezone)} — {point.title}"
        for index, point in enumerate(points, start=1)
    ]


def _entry_time_label(entry: Entry, timezone: str) -> str:
    timestamp = entry.local_timestamp or entry.created_at
    if timestamp is None:
        return "час неясний"
    return timestamp.astimezone(zoneinfo(timezone)).strftime("%H:%M")


async def _format_day_for_date(
    session: AsyncSession, *, user: User, target_date: date
) -> tuple[str, str | None]:
    day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=target_date)
    if day is None:
        return f"За {target_date.isoformat()} ще немає збереженого дня.", None
    text = await format_day_view(session, user=user, day=day, limit=30)
    return text, str(day.id)


async def _format_similar_memory_query(
    session: AsyncSession,
    *,
    user: User,
    memory_service: MemoryService,
    query: str,
) -> str:
    records = list(await memory_service.similar_entries(session, user_id=user.id, query_text=query, limit=6))
    entry_ids = [record.target_id for record in records if record.target_type == "entry"]
    entries = list(await repo.list_entries_by_ids(session, entry_ids=entry_ids))
    analyses = list(await repo.list_analyses_for_targets(session, target_type="entry", target_ids=entry_ids))
    return format_similar_entries(
        records,
        query=query,
        entries=entries,
        analyses=analyses,
        timezone=user.timezone,
    )


def _embeddings_ready(settings: Settings) -> bool:
    return bool(settings.embeddings_enabled and settings.embedding_api_key)


async def _edit_or_answer_menu(callback: CallbackQuery, text: str, reply_markup) -> None:
    try:
        await callback.message.edit_text(text, reply_markup=reply_markup)
    except Exception:
        logger.debug("Could not edit menu message", exc_info=True)
        await callback.message.answer(text, reply_markup=reply_markup)


def _parse_day_query(query: str, timezone: str, *, today: date | None = None) -> date | None:
    normalized = " ".join(query.strip().lower().split())
    if not normalized:
        return None
    base_date = today or datetime.now(tz=zoneinfo(timezone)).date()
    if normalized in {"сьогодні", "today"}:
        return base_date
    if normalized in {"вчора", "yesterday"}:
        return base_date - timedelta(days=1)
    try:
        return date.fromisoformat(normalized)
    except ValueError:
        pass
    try:
        return datetime.strptime(normalized, "%d.%m.%Y").date()
    except ValueError:
        return None


def _parse_date_range_query(
    query: str,
    timezone: str,
    *,
    today: date | None = None,
) -> tuple[date, date] | None:
    normalized = " ".join(query.replace("—", " ").replace("–", " ").replace(",", " ").split())
    if not normalized:
        return None
    parts = normalized.split()
    if len(parts) == 1:
        single = _parse_day_query(parts[0], timezone, today=today)
        return (single, single) if single is not None else None
    if len(parts) != 2:
        return None
    start = _parse_day_query(parts[0], timezone, today=today)
    end = _parse_day_query(parts[1], timezone, today=today)
    if start is None or end is None:
        return None
    if start > end:
        start, end = end, start
    return start, end


def _visual_report_range_hint() -> str:
    return (
        "Напиши діапазон для PDF-звіту двома датами.\n\n"
        "Наприклад: `2026-07-01 2026-07-09`.\n"
        "Можна також: `вчора сьогодні` або одну дату для звіту за один день.\n"
        f"Поки максимум: {MAX_VISUAL_REPORT_DAYS} днів."
    )


def _reanalysis_range_hint() -> str:
    return (
        "Напиши журнальний діапазон для переаналізу двома датами.\n\n"
        "Наприклад: `2026-07-01 2026-07-09`.\n"
        "Можна також: `вчора сьогодні` або одну дату."
    )


def _parse_reanalysis_action(action: str) -> tuple[str, int | None, date | None, date | None] | None:
    if action == "all":
        return "all", None, None, None
    parts = action.split(":")
    if len(parts) == 2 and parts[0] == "recent":
        try:
            return "recent", max(1, min(int(parts[1]), FEATURE_REANALYSIS_LIMIT)), None, None
        except ValueError:
            return None
    if len(parts) == 3 and parts[0] == "range":
        try:
            start_date = datetime.strptime(parts[1], "%Y%m%d").date()
            end_date = datetime.strptime(parts[2], "%Y%m%d").date()
        except ValueError:
            return None
        return "range", None, min(start_date, end_date), max(start_date, end_date)
    # Compatibility with buttons sent before the guided flow existed.
    try:
        return "recent", max(1, min(int(action), FEATURE_REANALYSIS_LIMIT)), None, None
    except ValueError:
        return None


def _reanalysis_scope_label(*, scope: str, limit: int | None, start_date: date | None, end_date: date | None) -> str:
    if scope == "recent":
        return f"останні {limit or 0} записів"
    if scope == "range" and start_date and end_date:
        return f"журнальні дні {start_date.isoformat()} - {end_date.isoformat()}"
    return "увесь архів"


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


def _quiet_until_for_preset(preset: str, *, timezone: str) -> datetime:
    now_local = datetime.now(UTC).astimezone(zoneinfo(timezone))
    if preset == "1h":
        return (now_local + timedelta(hours=1)).astimezone(UTC)
    if preset == "2h":
        return (now_local + timedelta(hours=2)).astimezone(UTC)
    if preset == "evening":
        target = now_local.replace(hour=20, minute=0, second=0, microsecond=0)
        if target <= now_local:
            target = now_local.replace(hour=23, minute=30, second=0, microsecond=0)
        if target <= now_local:
            target = now_local + timedelta(hours=2)
        return target.astimezone(UTC)
    if preset == "tomorrow":
        target = (now_local + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        return target.astimezone(UTC)
    return (now_local + timedelta(hours=1)).astimezone(UTC)


def _parse_quiet_until_text(text: str, *, timezone: str) -> datetime | None:
    normalized = " ".join(text.lower().replace(",", ".").split())
    now_local = datetime.now(UTC).astimezone(zoneinfo(timezone))
    duration_match = re.search(r"(?:на\s*)?(\d+(?:\.\d+)?)\s*(год|годин|години|h)", normalized)
    if duration_match:
        hours = float(duration_match.group(1))
        if 0 < hours <= 24:
            return (now_local + timedelta(hours=hours)).astimezone(UTC)
    minutes_match = re.search(r"(?:на\s*)?(\d+)\s*(хв|хвилин|m)", normalized)
    if minutes_match:
        minutes = int(minutes_match.group(1))
        if 0 < minutes <= 24 * 60:
            return (now_local + timedelta(minutes=minutes)).astimezone(UTC)
    time_match = re.search(r"(\d{1,2})[:.](\d{2})", normalized)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            target = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now_local:
                target += timedelta(days=1)
            return target.astimezone(UTC)
    return None


def _quiet_status_text(user_settings: UserSettings, *, timezone: str) -> str:
    until = quiet_until(user_settings)
    if until and quiet_is_active(user_settings):
        local_until = until.astimezone(zoneinfo(timezone))
        return f"Тиха пауза активна до {local_until:%H:%M}."
    return "Тиха пауза зараз не активна."


def _quiet_set_text(user_settings: UserSettings, *, timezone: str) -> str:
    until = quiet_until(user_settings)
    if until is None:
        return "Ок, тиха пауза встановлена."
    local_until = until.astimezone(zoneinfo(timezone))
    return f"Ок, не турбуватиму до {local_until:%H:%M}."


def _settings_view(settings_json: dict):
    class SettingsView:
        pass

    view = SettingsView()
    view.settings_json = settings_json
    return view


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
            "/visual_report 2026-07-01 2026-07-09 - PDF-звіт за період",
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


def _format_semantic_memory_influences(snapshots, prompts, *, timezone: str) -> str:
    if not snapshots:
        return "Поки немає автоматичних питань для перевірки впливу пам’яті."
    prompt_by_snapshot = {}
    for prompt in prompts:
        if prompt.prompt_kind == "initial":
            prompt_by_snapshot[str(prompt.snapshot_id)] = prompt.text
    lines = ["Пам’ять у питаннях", ""]
    used_count = 0
    for snapshot in snapshots:
        insight = (snapshot.context_json or {}).get("semantic_memory_insight") or {}
        if not isinstance(insight, dict) or not insight.get("used"):
            continue
        hypothesis = " ".join(str(insight.get("hypothesis") or "").split())
        if not hypothesis:
            continue
        used_count += 1
        prompted_at = snapshot.prompted_at.astimezone(zoneinfo(timezone)) if snapshot.prompted_at else None
        time_text = prompted_at.strftime("%d.%m %H:%M") if prompted_at else "час невідомий"
        evidence_count = len(insight.get("evidence_entry_ids") or [])
        question = " ".join(str(prompt_by_snapshot.get(str(snapshot.id)) or "").split())
        lines.extend(
            [
                f"{time_text} — {hypothesis}",
                f"Підстави: {evidence_count} схожих попередніх записів.",
                f"Питання: {question or 'не збережено'}",
                "",
            ]
        )
    if used_count == 0:
        lines.append(
            "У переглянутих питаннях семантична пам’ять не змінила формулювання. "
            "Це нормально: вона не має підтягуватися лише через поверхневу схожість."
        )
    return "\n".join(lines).rstrip()


def _format_settings_text(*, user_settings: UserSettings, snapshots_are_paused: bool) -> str:
    status = "на паузі" if snapshots_are_paused else "увімкнені"
    body = "так" if user_settings.ask_body_signals else "ні"
    photo = "так" if user_settings.photo_prompts_enabled else "ні"
    context_quiet = "так" if context_quiet_enabled(user_settings) else "ні"
    adaptive_observation = "так" if adaptive_observation_enabled(user_settings) else "ні"
    quiet_status = _quiet_status_text(user_settings, timezone=getattr(user_settings, "timezone", "Europe/Kyiv"))
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
            f"Адаптивна частота: {adaptive_observation}",
            f"Нагадування після: {user_settings.reminder_delay_minutes} хв",
            f"Питати про тіло: {body}",
            f"Фото-підказки: {photo}",
            f"Контекстна тиша: {context_quiet}",
            quiet_status,
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


def _settings_keyboard_for_action(action: str, *, user_settings: UserSettings):
    if action == "settings:section:rhythm" or action.startswith(("settings:freq:", "settings:reminder:")) or action == "settings:toggle:adaptive_observation":
        return settings_rhythm_keyboard(user_settings=user_settings)
    if action in {"settings:pause", "settings:resume", "settings:toggle:pause"}:
        return settings_rhythm_keyboard(user_settings=user_settings)
    if action == "settings:section:style" or action.startswith(("settings:tone:", "settings:humanity:")):
        return settings_style_keyboard(user_settings=user_settings)
    if action in {"settings:custom_style", "settings:profile_context"}:
        return settings_style_keyboard(user_settings=user_settings)
    if action == "settings:section:capture" or action in {"settings:toggle:body", "settings:toggle:photo"}:
        return settings_capture_keyboard(user_settings=user_settings)
    if action == "settings:toggle:context_quiet":
        return settings_keyboard(user_settings=user_settings)
    return settings_keyboard(user_settings=user_settings)


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
