from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from hashlib import blake2b
from typing import Any

from aiogram import Bot
from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.ai.service import AIService
from mental_state_bot.bot.keyboards import missed_prompt_keyboard, snapshot_initial_keyboard
from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import Day, Snapshot, User, UserSettings
from mental_state_bot.services.analysis_backfill import ENTRY_FEATURES_TASK
from mental_state_bot.services.interaction_gate import serialized_user_interaction
from mental_state_bot.services.journal_day import current_journal_date
from mental_state_bot.services.memory_graph import relevant_memory_context_for_text
from mental_state_bot.services.planned_events import planned_event_context
from mental_state_bot.services.preferences import (
    adaptive_observation_enabled,
    custom_interaction_style,
    life_context_items,
    pending_clarification,
    pending_memory_graph_confirmation,
    post_entry_followup_is_active,
    quiet_is_active,
    snapshots_paused,
    user_profile_context,
)
from mental_state_bot.services.semantic_context import semantic_memory_context
from mental_state_bot.time_utils import parse_hhmm, utc_now, zoneinfo

logger = logging.getLogger(__name__)

_OPEN_SNAPSHOT_REMINDERS = [
    "Той зріз ще відкритий. Можеш відповісти зараз, відкласти або коротко позначити пропуск.",
    "Ще лишився незавершений зріз. Якщо зараз не до нього, можна відкласти або вказати причину.",
    "Є відкритий зріз без відповіді. Можна закрити його короткою відповіддю або просто відкласти.",
]

_OPEN_CLARIFICATION_REMINDERS = [
    "Те уточнення ще відкрите. Можеш відповісти одним реченням або відкласти.",
    "Ще лишилося відкрите уточнення. Якщо зараз не хочеться розбиратися, можна відкласти.",
    "Є незавершене уточнення до попереднього запису. Можеш відповісти коротко або позначити пропуск.",
]


@serialized_user_interaction
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
    if (
        pending_clarification(user_settings) is not None
        or pending_memory_graph_confirmation(user_settings) is not None
        or post_entry_followup_is_active(user_settings)
    ):
        return False
    if not _is_active_time(now, user.timezone, user_settings):
        return False

    target_date = await current_journal_date(session, user=user, user_settings=user_settings, now=now)
    today = await repo.get_day_by_date(
        session,
        user_id=user.id,
        local_date_value=target_date,
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
    last_snapshot_activity = _last_snapshot_activity_at(last_snapshot)
    if _activity_is_too_recent(
        now,
        last_snapshot_activity,
        threshold_minutes=user_settings.min_interval_minutes,
    ):
        return False

    latest_entry = await repo.get_latest_observation_entry(session, user_id=user.id)
    threshold, scheduling_context = await _next_snapshot_interval(
        session,
        user_settings=user_settings,
        entry=latest_entry,
    )
    if _activity_is_too_recent(now, last_snapshot_activity, threshold_minutes=threshold):
        return False

    last_entry_activity = _last_entry_activity_at(latest_entry)
    if _activity_is_too_recent(now, last_entry_activity, threshold_minutes=threshold):
        return False

    logger.info(
        "Scheduling snapshot",
        extra={"user_id": str(user.id), "interval_minutes": threshold, "scheduling": scheduling_context},
    )

    return await _send_snapshot_prompt_unlocked(
        session,
        bot=bot,
        ai_service=ai_service,
        user=user,
        user_settings=user_settings,
        intent="state_and_activity",
        scheduled_for=now,
        photo_prompt_chance=settings.photo_prompt_chance,
        scheduling_context=scheduling_context,
    )


@serialized_user_interaction
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
    scheduling_context: dict[str, Any] | None = None,
) -> bool:
    return await _send_snapshot_prompt_unlocked(
        session,
        bot=bot,
        ai_service=ai_service,
        user=user,
        user_settings=user_settings,
        intent=intent,
        scheduled_for=scheduled_for,
        photo_prompt_chance=photo_prompt_chance,
        scheduling_context=scheduling_context,
    )


async def _send_snapshot_prompt_unlocked(
    session: AsyncSession,
    *,
    bot: Bot,
    ai_service: AIService,
    user: User,
    user_settings: UserSettings,
    intent: str,
    scheduled_for: datetime | None,
    photo_prompt_chance: float = 0.18,
    scheduling_context: dict[str, Any] | None = None,
) -> bool:
    now = utc_now()
    if post_entry_followup_is_active(user_settings):
        return False
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
        local_date_value=await current_journal_date(session, user=user, user_settings=user_settings, now=now),
        started_at=now,
    )
    recent_entries = await repo.get_recent_entries(session, user_id=user.id, limit=4)
    day_entries = await repo.list_day_entries(session, day_id=day.id)
    rhythm_context = await daily_rhythm_context(session, user=user, current_day=day)
    context = snapshot_question_context(
        recent_entries=recent_entries,
        day_entries=day_entries,
        user_settings=user_settings,
        trigger="manual" if scheduled_for is None else "scheduled",
        photo_prompt_opportunity=random.random() < photo_prompt_chance,
        day_phase=_question_day_phase(
            trigger="manual" if scheduled_for is None else "scheduled",
            day_entries=day_entries,
            daily_rhythm=rhythm_context,
        ),
        daily_rhythm=rhythm_context,
    )
    if scheduling_context:
        context["scheduling"] = scheduling_context
    ai_settings = getattr(ai_service, "settings", None)
    context["semantic_memory"] = (
        await semantic_memory_context(
            session,
            settings=ai_settings,
            ai_service=ai_service,
            user=user,
            query_text=_question_query_text(context),
            task_name="snapshot_question_semantic_context",
            limit=4,
            exclude_entry_ids={str(entry.id) for entry in day_entries},
        )
        if ai_settings is not None
        else []
    )
    try:
        context["relevant_memory_graph"] = await relevant_memory_context_for_text(
            session,
            user_id=user.id,
            text=_question_query_text(context),
            limit=8,
            task_name="snapshot_question_memory_graph",
        )
    except Exception:
        context["relevant_memory_graph"] = {"nodes": [], "edges": [], "matched": []}
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
    memory_insight = _verified_semantic_memory_insight(
        question.semantic_memory_insight.model_dump(),
        context.get("semantic_memory") or [],
    )
    if memory_insight is not None:
        snapshot.context_json = {
            **snapshot.context_json,
            "semantic_memory_insight": memory_insight,
        }
        await session.flush()
    question_text = _ensure_photo_prompt_if_requested(question.question, context)
    message = await bot.send_message(
        chat_id=user.chat_id,
        text=question_text,
        reply_markup=snapshot_initial_keyboard(),
    )
    await repo.mark_snapshot_prompted(session, snapshot_id=snapshot.id, prompted_at=now)
    await repo.add_prompt(
        session,
        snapshot_id=snapshot.id,
        prompt_kind="initial",
        text=question_text,
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
    day_phase: str | None = None,
    daily_rhythm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timezone_name = getattr(user_settings, "timezone", "Europe/Kyiv")
    now = utc_now().astimezone(zoneinfo(timezone_name))
    photo_prompt_style = "Не проси фото."
    if user_settings.photo_prompts_enabled:
        photo_prompt_style = (
            "Цього разу явно додай необов'язкову можливість відповісти фото або показати момент. "
            "Сформулюй природно, коротко, без вимоги фото і без окремого великого діалогу."
            if photo_prompt_opportunity
            else "Цього разу краще не просити фото, якщо тільки контекст явно не просить візуального моменту."
        )
    return {
        "recent_entries": [
            _entry_context(entry)
            for entry in recent_entries[-4:]
        ],
        "day_context": _day_context(day_entries or [], limit=16),
        "style": {
            "tone": user_settings.tone,
            "humanity_level": user_settings.humanity_level,
            "custom_interaction_style": custom_interaction_style(user_settings),
            "life_context": life_context_items(user_settings)[-20:],
        },
        "user_profile_context": user_profile_context(user_settings),
        "day_phase": day_phase
        or _question_day_phase(
            trigger=trigger,
            day_entries=day_entries or [],
            daily_rhythm=daily_rhythm,
        ),
        "current_local_datetime": now.isoformat(),
        "current_local_date": now.date().isoformat(),
        "current_local_time": now.strftime("%H:%M"),
        "current_day_part": _day_part(now),
        "daily_rhythm": daily_rhythm or _fallback_daily_rhythm(now),
        "planned_events": planned_event_context(user_settings, now=now),
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


async def daily_rhythm_context(
    session: AsyncSession,
    *,
    user: User,
    current_day: Day,
    lookback_days: int = 7,
) -> dict[str, Any]:
    start_date = current_day.local_date - timedelta(days=lookback_days)
    end_date = current_day.local_date - timedelta(days=1)
    if end_date < start_date:
        return _fallback_daily_rhythm(datetime.now(tz=zoneinfo(user.timezone)))
    days = await repo.list_days_between(
        session,
        user_id=user.id,
        start_date=start_date,
        end_date=end_date,
    )
    first_times: list[int] = []
    for day in days:
        entries = await repo.list_day_entries(session, day_id=day.id)
        first_entry = _first_meaningful_day_entry(entries)
        if first_entry is None or first_entry.local_timestamp is None:
            continue
        local_timestamp = first_entry.local_timestamp.astimezone(zoneinfo(user.timezone))
        first_times.append(local_timestamp.hour * 60 + local_timestamp.minute)
    local_now = utc_now().astimezone(zoneinfo(user.timezone))
    if not first_times:
        return _fallback_daily_rhythm(local_now)
    average_minutes = int(round(sum(first_times) / len(first_times)))
    return {
        "average_first_entry_time_last_7_days": _minutes_to_hhmm(average_minutes),
        "average_first_entry_label": "орієнтовний час старту дня, не точний час пробудження",
        "days_sampled": len(first_times),
        "current_day_part": _day_part(local_now),
        "minutes_since_average_first_entry": _minutes_since_anchor(
            local_now.hour * 60 + local_now.minute,
            average_minutes,
        ),
        "source": "first_meaningful_entry_after_journal_day_boundary",
    }


def _ensure_photo_prompt_if_requested(question_text: str, context: dict[str, Any]) -> str:
    preferences = context.get("question_preferences") or {}
    if not preferences.get("photo_prompts_enabled") or not preferences.get("photo_prompt_opportunity"):
        return question_text
    if _mentions_photo(question_text):
        return question_text
    return f"{question_text}\n\nМожна відповісти текстом або показати цей момент фото."


def _mentions_photo(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ("фото", "знім", "світлин", "покажи"))


def _question_day_phase(*, trigger: str, day_entries, daily_rhythm: dict[str, Any] | None = None) -> str:
    if trigger == "scheduled" and not list(day_entries or []) and _looks_like_morning_start(daily_rhythm):
        return "morning_start"
    return "regular"


def _looks_like_morning_start(daily_rhythm: dict[str, Any] | None) -> bool:
    if not daily_rhythm:
        return True
    day_part = daily_rhythm.get("current_day_part")
    minutes = daily_rhythm.get("minutes_since_average_first_entry")
    if isinstance(minutes, int):
        return -90 <= minutes <= 180
    return day_part in {None, "morning"}


def _first_meaningful_day_entry(entries) -> Any | None:
    skipped_sources = {
        "sleep_marker",
        "day_reflection",
        "correction",
        "profile_context_update",
        "missed_reason",
    }
    for entry in entries:
        source = str(getattr(entry, "source", "") or "")
        if source in skipped_sources or source.startswith("button_"):
            continue
        if not str(getattr(entry, "raw_text", "") or "").strip():
            continue
        return entry
    return None


def _day_part(local_time: datetime) -> str:
    hour = local_time.hour
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 17:
        return "daytime"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def _fallback_daily_rhythm(local_time: datetime) -> dict[str, Any]:
    return {
        "average_first_entry_time_last_7_days": None,
        "average_first_entry_label": None,
        "days_sampled": 0,
        "current_day_part": _day_part(local_time),
        "minutes_since_average_first_entry": None,
        "source": "insufficient_history",
    }


def _minutes_to_hhmm(minutes: int) -> str:
    minutes %= 24 * 60
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _minutes_since_anchor(current_minutes: int, anchor_minutes: int) -> int:
    delta = current_minutes - anchor_minutes
    if delta < -12 * 60:
        delta += 24 * 60
    if delta > 12 * 60:
        delta -= 24 * 60
    return delta


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
    if context.get("day_phase") == "morning_start":
        return "ранковий зріз сон пробудження виспався настрій сили початок дня"
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
        text = _snapshot_reminder_text(prompts)
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


def _snapshot_reminder_text(prompts: list[Any]) -> str:
    latest_kind = _latest_non_reminder_prompt_kind(prompts)
    variants = (
        _OPEN_CLARIFICATION_REMINDERS
        if latest_kind == "clarification"
        else _OPEN_SNAPSHOT_REMINDERS
    )
    return random.choice(variants)


def _latest_non_reminder_prompt_kind(prompts: list[Any]) -> str | None:
    for prompt in reversed(prompts):
        prompt_kind = getattr(prompt, "prompt_kind", None)
        if prompt_kind and prompt_kind != "reminder":
            return str(prompt_kind)
    return None


def _last_snapshot_activity_at(snapshot: Snapshot | None) -> datetime | None:
    if snapshot is None:
        return None
    return snapshot.closed_at or snapshot.prompted_at


def _last_entry_activity_at(entry: Any | None) -> datetime | None:
    if entry is None:
        return None
    return getattr(entry, "created_at", None)


def _activity_is_too_recent(
    now: datetime,
    activity_at: datetime | None,
    *,
    threshold_minutes: int,
) -> bool:
    if activity_at is None:
        return False
    return (now - activity_at).total_seconds() / 60 < threshold_minutes


async def _next_snapshot_interval(
    session: AsyncSession,
    *,
    user_settings: UserSettings,
    entry: Any | None,
) -> tuple[int, dict[str, Any]]:
    minimum = int(user_settings.min_interval_minutes)
    maximum = int(user_settings.max_interval_minutes)
    baseline = random.randint(minimum, maximum)
    default_context = {"mode": "settings_random", "window_minutes": [minimum, maximum], "selected_minutes": baseline}
    if entry is None or getattr(entry, "id", None) is None or not adaptive_observation_enabled(user_settings):
        return baseline, default_context

    analyses = await repo.list_analyses_for_targets(session, target_type="entry", target_ids=[entry.id])
    latest = next(
        (
            analysis.result
            for analysis in reversed(list(analyses))
            if analysis.task_name == ENTRY_FEATURES_TASK and isinstance(analysis.result, dict)
        ),
        None,
    )
    cadence = latest.get("observation_cadence") if isinstance(latest, dict) else None
    if not isinstance(cadence, dict):
        return baseline, default_context
    try:
        confidence = float(cadence.get("confidence") or 0.0)
        suggested_minimum = int(cadence["next_checkin_min_minutes"])
        suggested_maximum = int(cadence["next_checkin_max_minutes"])
    except (KeyError, TypeError, ValueError):
        return baseline, default_context
    if confidence < 0.45:
        return baseline, default_context

    bounded_minimum = min(max(suggested_minimum, minimum), maximum)
    bounded_maximum = min(max(suggested_maximum, minimum), maximum)
    window_minimum, window_maximum = min(bounded_minimum, bounded_maximum), max(bounded_minimum, bounded_maximum)
    selected = _stable_interval_for_entry(entry.id, window_minimum, window_maximum)
    return selected, {
        "mode": "adaptive",
        "window_minutes": [window_minimum, window_maximum],
        "selected_minutes": selected,
        "volatility": cadence.get("volatility"),
        "change_likelihood": cadence.get("change_likelihood"),
        "eventfulness": cadence.get("eventfulness"),
        "confidence": confidence,
        "reason": cadence.get("reason"),
    }


def _stable_interval_for_entry(entry_id: object, minimum: int, maximum: int) -> int:
    if minimum >= maximum:
        return minimum
    digest = blake2b(str(entry_id).encode("utf-8"), digest_size=4).digest()
    return minimum + int.from_bytes(digest, "big") % (maximum - minimum + 1)


def _verified_semantic_memory_insight(insight: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not insight.get("used"):
        return None
    allowed_ids = {str(record.get("target_id") or "") for record in records if record.get("target_id")}
    evidence_ids = [
        str(entry_id)
        for entry_id in insight.get("evidence_entry_ids") or []
        if str(entry_id) in allowed_ids
    ]
    hypothesis = " ".join(str(insight.get("hypothesis") or "").split())
    if not hypothesis or not evidence_ids:
        return None
    return {
        "used": True,
        "hypothesis": hypothesis[:400],
        "evidence_entry_ids": evidence_ids[:4],
        "confidence": float(insight.get("confidence") or 0.0),
    }


def _is_active_time(now_utc: datetime, timezone: str, user_settings: UserSettings) -> bool:
    if snapshots_paused(user_settings):
        return False
    if quiet_is_active(user_settings, now_utc):
        return False
    local = now_utc.astimezone(zoneinfo(timezone))
    start = parse_hhmm(user_settings.active_start)
    end = parse_hhmm(user_settings.active_end)
    current = local.time()
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end
