from __future__ import annotations

import calendar
import logging
import uuid
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.ai.schemas import EntryFeatures
from mental_state_bot.ai.service import AIService
from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import Day, EmbeddingRecord, Entry, Summary, User
from mental_state_bot.services.analysis_backfill import (
    ENTRY_FEATURES_SCHEMA_VERSION,
    ENTRY_FEATURES_TASK,
    postprocess_entry_features,
)
from mental_state_bot.services.journal_day import current_journal_date
from mental_state_bot.services.period_analysis import build_period_analysis, compare_period_analyses
from mental_state_bot.services.preferences import (
    clarification_queue,
    custom_interaction_style,
    settings_json_with_clarification_queue,
    user_profile_context,
)
from mental_state_bot.services.semantic_context import verified_semantic_memory_insight
from mental_state_bot.time_utils import local_now, parse_hhmm, utc_now, zoneinfo

logger = logging.getLogger(__name__)

AUTO_MORNING_BOUNDARY_KIND = "auto_morning"
UNCERTAIN_DAY_BOUNDARY_QUALITY = "day_boundary_uncertain"


class SummaryService:
    def __init__(self, settings: Settings, ai_service: AIService) -> None:
        self.settings = settings
        self.ai = ai_service

    async def generate_today_summary(self, session: AsyncSession, *, user: User) -> Summary:
        user_settings = await repo.get_user_settings(session, user.id)
        target_date = await current_journal_date(session, user=user, user_settings=user_settings)
        day = await repo.get_or_create_day(
            session,
            user_id=user.id,
            local_date_value=target_date,
            started_at=utc_now(),
        )
        return await self.generate_day_summary(session, user=user, day=day)

    async def close_today_with_summary(
        self,
        session: AsyncSession,
        *,
        user: User,
        day_reflection: str | None = None,
        day_reflection_kind: str | None = None,
    ) -> Summary:
        user_settings = await repo.get_user_settings(session, user.id)
        sleep_time = local_now(user.timezone)
        target_date = await current_journal_date(
            session,
            user=user,
            user_settings=user_settings,
            now=sleep_time,
        )
        day = await repo.get_or_create_day(
            session,
            user_id=user.id,
            local_date_value=target_date,
            started_at=utc_now(),
        )
        normalized_reflection = " ".join((day_reflection or "").split())
        if normalized_reflection:
            await repo.add_entry(
                session,
                user_id=user.id,
                day_id=day.id,
                snapshot_id=None,
                source="day_reflection",
                raw_text=f"Оцінка дня: {normalized_reflection}",
                telegram_message_id=None,
                reply_to_message_id=None,
                local_timestamp=sleep_time,
                meta={
                    "day_reflection": normalized_reflection,
                    "day_reflection_kind": day_reflection_kind or "free",
                    "sleep_marker_target_date": target_date.isoformat(),
                },
            )
        await repo.add_entry(
            session,
            user_id=user.id,
            day_id=day.id,
            snapshot_id=None,
            source="sleep_marker",
            raw_text="лягаю спати",
            telegram_message_id=None,
            reply_to_message_id=None,
            local_timestamp=sleep_time,
            meta={
                "boundary_kind": "sleep_marker",
                "sleep_marker_target_date": target_date.isoformat(),
            },
        )
        return await self.generate_day_summary(session, user=user, day=day, close_day=True)

    async def generate_yesterday_summary_if_needed(
        self, session: AsyncSession, *, user: User
    ) -> Summary | None:
        user_settings = await repo.get_user_settings(session, user.id)
        today = await current_journal_date(session, user=user, user_settings=user_settings)
        yesterday = today - timedelta(days=1)
        day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=yesterday)
        if day is None:
            return None
        entries = await repo.list_day_entries(session, day_id=day.id)
        if not entries:
            return None
        if day.ended_at is None:
            await _close_day_in_session(
                session,
                day=day,
                ended_at=auto_morning_boundary_end(day.local_date, user.timezone),
                boundary_kind=AUTO_MORNING_BOUNDARY_KIND,
                data_quality=UNCERTAIN_DAY_BOUNDARY_QUALITY,
            )
        if await repo.summary_exists(session, user_id=user.id, day_id=day.id, period_type="daily"):
            return None
        return await self.generate_day_summary(session, user=user, day=day)

    async def generate_day_summary(
        self, session: AsyncSession, *, user: User, day: Day, close_day: bool = False
    ) -> Summary:
        if close_day:
            await _close_day_in_session(
                session,
                day=day,
                ended_at=utc_now(),
                boundary_kind="sleep_marker",
            )
        user_settings = await repo.get_user_settings(session, user.id)
        period_start, period_end = _day_period_bounds(
            day,
            user.timezone,
            active_start=user_settings.active_start,
        )
        entries = await repo.list_day_entries(session, day_id=day.id)
        missed_prompts = await repo.list_missed_prompts_between(
            session,
            user_id=user.id,
            start=period_start,
            end=period_end,
        )
        analyses = await repo.list_analyses_for_targets(
            session,
            target_type="entry",
            target_ids=[entry.id for entry in entries],
        )
        analyses_by_entry = {}
        for analysis in analyses:
            analyses_by_entry.setdefault(str(analysis.target_id), []).append(
                {
                    "task": analysis.task_name,
                    "result": analysis.result,
                    "confidence": float(analysis.confidence) if isinstance(analysis.confidence, Decimal) else None,
                    "uncertainty_notes": analysis.uncertainty_notes,
                }
            )
        snapshot_conversations = await _snapshot_conversations_from_entries(session, entries=entries)
        context = {
            "date": day.local_date.isoformat(),
            "user_profile_context": user_profile_context(user_settings),
            "style": {
                "custom_interaction_style": custom_interaction_style(user_settings),
            },
            "entries": [
                {
                    "id": str(entry.id),
                    "created_at": entry.created_at.isoformat() if entry.created_at else None,
                    "local_timestamp": entry.local_timestamp.isoformat() if entry.local_timestamp else None,
                    "source": entry.source,
                    "snapshot_id": str(entry.snapshot_id) if entry.snapshot_id else None,
                    "raw_text": entry.raw_text,
                    "analyses": analyses_by_entry.get(str(entry.id), []),
                }
                for entry in entries
            ],
            "snapshot_conversations": snapshot_conversations,
            "entry_count": len(entries),
            "missed_prompts": [
                {
                    "missed_at": missed.missed_at.isoformat(),
                    "status": missed.status,
                    "reason_text": missed.reason_text,
                    "reminder_sent_at": missed.reminder_sent_at.isoformat()
                    if missed.reminder_sent_at
                    else None,
                    "resolved_at": missed.resolved_at.isoformat() if missed.resolved_at else None,
                }
                for missed in missed_prompts
            ],
            "day_boundary": {
                "started_at": day.started_at.isoformat() if day.started_at else None,
                "ended_at": day.ended_at.isoformat() if day.ended_at else None,
                "boundary_kind": day.boundary_kind,
                "data_quality": day.data_quality,
            },
        }
        context["semantic_memory"] = await self._semantic_memory_context(
            session,
            user=user,
            query=_entries_query_text(entries, label=f"daily summary {day.local_date.isoformat()}"),
            task_name="daily_summary_semantic_context",
            limit=8,
            exclude_entry_ids={str(entry.id) for entry in entries},
        )
        evening_review, evening_review_run_id = await self.ai.review_evening_day(
            session, user_id=user.id, context=context
        )
        patched = await _apply_evening_review_patches(
            session,
            settings=self.settings,
            user=user,
            entries=entries,
            analyses=analyses,
            patches=evening_review.patches,
            model_run_id=evening_review_run_id,
        )
        if patched:
            context["evening_review_applied_patches"] = sorted(patched)
            _replace_context_entry_analyses(context, patched)
        queued_questions = await _queue_evening_review_questions(
            session,
            user=user,
            user_settings=user_settings,
            entries=entries,
            review=evening_review,
            model_run_id=evening_review_run_id,
        )
        if queued_questions:
            context["evening_review_queued_questions"] = queued_questions
        context["evening_review"] = evening_review.model_dump()
        await repo.add_ai_analysis(
            session,
            user_id=user.id,
            target_type="day",
            target_id=day.id,
            task_name="review_evening_day",
            schema_version="evening_review.v1",
            provider=self.settings.ai_provider,
            model=self.settings.ai_heavy_model,
            result=evening_review.model_dump(),
            confidence=None,
            uncertainty_notes=evening_review.uncertain_items,
            model_run_id=evening_review_run_id,
        )
        daily, model_run_id = await self.ai.generate_daily_summary(
            session,
            user_id=user.id,
            context=context,
        )
        details = daily.model_dump()
        raw_insight = getattr(daily, "semantic_memory_insight", None)
        insight = verified_semantic_memory_insight(
            raw_insight.model_dump() if raw_insight is not None else {}, context["semantic_memory"]
        )
        if insight is None:
            details.pop("semantic_memory_insight", None)
        else:
            details["semantic_memory_insight"] = insight
        details["journal_date"] = day.local_date.isoformat()
        details["journal_active_start"] = user_settings.active_start
        summary = await repo.upsert_summary(
            session,
            user_id=user.id,
            day_id=day.id,
            period_type="daily",
            period_start=period_start,
            period_end=period_end,
            short_text=daily.short_text,
            details=details,
            model_run_id=model_run_id,
        )
        return summary

    async def generate_current_week_summary(self, session: AsyncSession, *, user: User) -> Summary:
        user_settings = await repo.get_user_settings(session, user.id)
        today = await current_journal_date(session, user=user, user_settings=user_settings)
        start_date, end_date = current_week_dates(today)
        return await self.generate_period_summary(
            session,
            user=user,
            period_type="weekly",
            start_date=start_date,
            end_date=end_date,
        )

    async def generate_current_month_summary(self, session: AsyncSession, *, user: User) -> Summary:
        user_settings = await repo.get_user_settings(session, user.id)
        today = await current_journal_date(session, user=user, user_settings=user_settings)
        start_date, end_date = current_month_dates(today)
        return await self.generate_period_summary(
            session,
            user=user,
            period_type="monthly",
            start_date=start_date,
            end_date=end_date,
        )

    async def generate_previous_week_summary(self, session: AsyncSession, *, user: User) -> Summary:
        user_settings = await repo.get_user_settings(session, user.id)
        today = await current_journal_date(session, user=user, user_settings=user_settings)
        start_date, end_date = previous_week_dates(today)
        return await self.generate_period_summary(
            session,
            user=user,
            period_type="weekly",
            start_date=start_date,
            end_date=end_date,
        )

    async def generate_previous_month_summary(self, session: AsyncSession, *, user: User) -> Summary:
        user_settings = await repo.get_user_settings(session, user.id)
        today = await current_journal_date(session, user=user, user_settings=user_settings)
        start_date, end_date = previous_month_dates(today)
        return await self.generate_period_summary(
            session,
            user=user,
            period_type="monthly",
            start_date=start_date,
            end_date=end_date,
        )

    async def generate_previous_week_summary_if_needed(
        self, session: AsyncSession, *, user: User
    ) -> Summary | None:
        user_settings = await repo.get_user_settings(session, user.id)
        today = await current_journal_date(session, user=user, user_settings=user_settings)
        start_date, end_date = previous_week_dates(today)
        period_start, period_end = _date_period_bounds(
            start_date,
            end_date,
            user.timezone,
            active_start=user_settings.active_start,
        )
        existing = await repo.get_period_summary(
            session,
            user_id=user.id,
            period_type="weekly",
            period_start=period_start,
            period_end=period_end,
        )
        if existing is not None and existing.delivered_at is None:
            return existing
        if existing is not None:
            return None
        entries = await repo.list_entries_between(
            session, user_id=user.id, start=period_start, end=period_end
        )
        if not entries:
            return None
        return await self.generate_previous_week_summary(session, user=user)

    async def generate_previous_month_summary_if_needed(
        self, session: AsyncSession, *, user: User
    ) -> Summary | None:
        user_settings = await repo.get_user_settings(session, user.id)
        today = await current_journal_date(session, user=user, user_settings=user_settings)
        start_date, end_date = previous_month_dates(today)
        period_start, period_end = _date_period_bounds(
            start_date,
            end_date,
            user.timezone,
            active_start=user_settings.active_start,
        )
        existing = await repo.get_period_summary(
            session,
            user_id=user.id,
            period_type="monthly",
            period_start=period_start,
            period_end=period_end,
        )
        if existing is not None and existing.delivered_at is None:
            return existing
        if existing is not None:
            return None
        entries = await repo.list_entries_between(
            session, user_id=user.id, start=period_start, end=period_end
        )
        if not entries:
            return None
        return await self.generate_previous_month_summary(session, user=user)

    async def generate_period_summary(
        self,
        session: AsyncSession,
        *,
        user: User,
        period_type: str,
        start_date: date,
        end_date: date,
    ) -> Summary:
        user_settings = await repo.get_user_settings(session, user.id)
        period_start, period_end = _date_period_bounds(
            start_date,
            end_date,
            user.timezone,
            active_start=user_settings.active_start,
        )
        entries = await repo.list_entries_between(
            session, user_id=user.id, start=period_start, end=period_end
        )
        daily_summaries = await repo.list_summaries_between(
            session,
            user_id=user.id,
            period_type="daily",
            start=period_start,
            end=period_end,
        )
        previous_start_date, previous_end_date = previous_comparable_period_dates(
            start_date=start_date,
            end_date=end_date,
            period_type=period_type,
        )
        previous_start, previous_end = _date_period_bounds(
            previous_start_date,
            previous_end_date,
            user.timezone,
            active_start=user_settings.active_start,
        )
        previous_summaries = await repo.list_summaries_between(
            session,
            user_id=user.id,
            period_type="daily",
            start=previous_start,
            end=previous_end,
        )
        deterministic_analysis = await build_period_analysis(
            session,
            user=user,
            start_date=start_date,
            end_date=end_date,
            period_start=period_start,
            period_end=period_end,
        )
        previous_deterministic_analysis = await build_period_analysis(
            session,
            user=user,
            start_date=previous_start_date,
            end_date=previous_end_date,
            period_start=previous_start,
            period_end=previous_end,
        )
        analyses = await repo.list_analyses_for_targets(
            session,
            target_type="entry",
            target_ids=[entry.id for entry in entries],
        )
        analyses_by_entry = {}
        for analysis in analyses:
            if analysis.task_name != "extract_entry_features":
                continue
            analyses_by_entry[str(analysis.target_id)] = analysis.result
        selected_entries = entries[-120:]
        snapshot_conversations = await _snapshot_conversations_from_entries(
            session,
            entries=selected_entries,
        )

        context = {
            "period_type": period_type,
            "journal_start_date": start_date.isoformat(),
            "journal_end_date": end_date.isoformat(),
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "user_profile_context": user_profile_context(user_settings),
            "style": {
                "custom_interaction_style": custom_interaction_style(user_settings),
            },
            "entry_count": len(entries),
            "deterministic_period_analysis": deterministic_analysis,
            "previous_period_comparison": compare_period_analyses(
                deterministic_analysis,
                previous_deterministic_analysis,
            ),
            "daily_summaries": [
                {
                    "period_start": summary.period_start.isoformat(),
                    "short_text": summary.short_text,
                    "details": summary.details,
                }
                for summary in daily_summaries
            ],
            "previous_daily_summaries": [
                {
                    "period_start": summary.period_start.isoformat(),
                    "short_text": summary.short_text,
                    "details": summary.details,
                }
                for summary in previous_summaries
            ],
            "selected_entries": [
                {
                    "id": str(entry.id),
                    "created_at": entry.created_at.isoformat() if entry.created_at else None,
                    "local_timestamp": entry.local_timestamp.isoformat() if entry.local_timestamp else None,
                    "snapshot_id": str(entry.snapshot_id) if entry.snapshot_id else None,
                    "source": entry.source,
                    "raw_text": entry.raw_text,
                    "features": analyses_by_entry.get(str(entry.id), {}),
                }
                for entry in selected_entries
            ],
            "snapshot_conversations": snapshot_conversations,
        }
        context["semantic_memory"] = await self._semantic_memory_context(
            session,
            user=user,
            query=_entries_query_text(
                entries[-80:],
                label=f"{period_type} summary {start_date.isoformat()} to {end_date.isoformat()}",
            ),
            task_name=f"{period_type}_summary_semantic_context",
            limit=12,
            exclude_entry_ids={str(entry.id) for entry in entries},
        )
        period_summary, model_run_id = await self.ai.generate_period_summary(
            session,
            user_id=user.id,
            period_type=period_type,
            context=context,
        )
        details = period_summary.model_dump()
        raw_insight = getattr(period_summary, "semantic_memory_insight", None)
        insight = verified_semantic_memory_insight(
            raw_insight.model_dump() if raw_insight is not None else {}, context["semantic_memory"]
        )
        if insight is None:
            details.pop("semantic_memory_insight", None)
        else:
            details["semantic_memory_insight"] = insight
        details["journal_start_date"] = start_date.isoformat()
        details["journal_end_date"] = end_date.isoformat()
        details["deterministic_period_analysis"] = deterministic_analysis
        details["previous_period_comparison"] = context["previous_period_comparison"]
        return await repo.upsert_summary(
            session,
            user_id=user.id,
            day_id=None,
            period_type=period_type,
            period_start=period_start,
            period_end=period_end,
            short_text=period_summary.short_text,
            details=details,
            model_run_id=model_run_id,
        )

    async def _semantic_memory_context(
        self,
        session: AsyncSession,
        *,
        user: User,
        query: str,
        task_name: str,
        limit: int,
        exclude_entry_ids: set[str],
    ) -> list[dict[str, Any]]:
        if not self.settings.embeddings_enabled or not self.settings.embedding_api_key:
            return []
        if not query.strip():
            return []
        try:
            result = await self.ai.create_embedding(query)
            embedding = result.data["embedding"] if result.data else []
            if len(embedding) != self.settings.embedding_dimensions:
                logger.warning(
                    "Summary semantic context skipped because embedding dimensions mismatch",
                    extra={"expected": self.settings.embedding_dimensions, "actual": len(embedding)},
                )
                return []
            records = await repo.find_similar_embeddings(
                session,
                user_id=user.id,
                embedding=embedding,
                limit=limit,
            )
            filtered_records = [
                record
                for record in records
                if not (record.target_type == "entry" and str(record.target_id) in exclude_entry_ids)
            ]
            await repo.add_retrieval_log(
                session,
                user_id=user.id,
                task_name=task_name,
                query_text=query,
                provider=result.provider,
                model=result.model,
                retrieved=[
                    {
                        "target_type": record.target_type,
                        "target_id": str(record.target_id),
                        "source_hash": record.source_hash,
                    }
                    for record in filtered_records
                ],
            )
            return _semantic_records_context(filtered_records)
        except Exception as exc:
            logger.warning(
                "Summary semantic context retrieval failed",
                extra={"user_id": str(user.id), "task_name": task_name, "error": str(exc)},
            )
            return []


async def _snapshot_conversations_from_entries(
    session: AsyncSession, *, entries: Sequence[Entry]
) -> dict[str, dict[str, Any]]:
    snapshot_ids = sorted({entry.snapshot_id for entry in entries if entry.snapshot_id}, key=str)
    if not snapshot_ids:
        return {}
    prompts = await repo.list_prompts_for_snapshots(session, snapshot_ids=snapshot_ids)
    prompts_by_snapshot: dict[str, list[dict[str, Any]]] = {str(snapshot_id): [] for snapshot_id in snapshot_ids}
    entries_by_snapshot: dict[str, list[dict[str, Any]]] = {str(snapshot_id): [] for snapshot_id in snapshot_ids}

    for prompt in prompts:
        prompts_by_snapshot.setdefault(str(prompt.snapshot_id), []).append(
            {
                "role": "bot",
                "kind": prompt.prompt_kind,
                "text": prompt.text,
                "sent_at": prompt.sent_at.isoformat() if prompt.sent_at else None,
            }
        )
    for entry in entries:
        if not entry.snapshot_id:
            continue
        entries_by_snapshot.setdefault(str(entry.snapshot_id), []).append(
            {
                "role": "user",
                "entry_id": str(entry.id),
                "created_at": entry.created_at.isoformat() if entry.created_at else None,
                "local_timestamp": entry.local_timestamp.isoformat() if entry.local_timestamp else None,
                "source": entry.source,
                "raw_text": entry.raw_text,
            }
        )

    conversations: dict[str, dict[str, Any]] = {}
    for snapshot_id in {**prompts_by_snapshot, **entries_by_snapshot}:
        prompt_context = prompts_by_snapshot.get(snapshot_id, [])
        entry_context = entries_by_snapshot.get(snapshot_id, [])
        transcript = sorted(
            [*prompt_context, *entry_context],
            key=lambda item: item.get("sent_at") or item.get("local_timestamp") or item.get("created_at") or "",
        )
        conversations[snapshot_id] = {
            "latest_prompt": prompt_context[-1]["text"] if prompt_context else None,
            "transcript": transcript,
            "prompts": prompt_context,
            "entries": entry_context,
        }
    return conversations


def _day_period_bounds(day: Day, timezone: str, *, active_start: str = "00:00") -> tuple[datetime, datetime]:
    tz = zoneinfo(timezone)
    start_time = parse_hhmm(active_start)
    start = datetime.combine(day.local_date, start_time, tzinfo=tz)
    end = start + timedelta(days=1) - timedelta(microseconds=1)
    return start, end


async def _close_day_in_session(
    session: AsyncSession,
    *,
    day: Day,
    ended_at: datetime,
    boundary_kind: str,
    data_quality: str | None = None,
) -> None:
    await repo.close_day(
        session,
        day_id=day.id,
        ended_at=ended_at,
        boundary_kind=boundary_kind,
        data_quality=data_quality,
    )
    day.ended_at = ended_at
    day.boundary_kind = boundary_kind
    if data_quality is not None:
        day.data_quality = data_quality


def auto_morning_boundary_end(local_date_value: date, timezone: str) -> datetime:
    tz = zoneinfo(timezone)
    next_midnight = datetime.combine(local_date_value + timedelta(days=1), time.min, tzinfo=tz)
    return next_midnight.astimezone(zoneinfo("UTC"))


def sleep_marker_target_date(sleep_time: datetime, *, active_start: str) -> date:
    target_date = sleep_time.date()
    if sleep_time.time() < parse_hhmm(active_start):
        return target_date - timedelta(days=1)
    return target_date


def _date_period_bounds(
    start_date: date,
    end_date: date,
    timezone: str,
    *,
    active_start: str = "00:00",
) -> tuple[datetime, datetime]:
    tz = zoneinfo(timezone)
    start_time = parse_hhmm(active_start)
    start = datetime.combine(start_date, start_time, tzinfo=tz)
    end = datetime.combine(end_date + timedelta(days=1), start_time, tzinfo=tz) - timedelta(microseconds=1)
    return start, end


def current_week_dates(today: date) -> tuple[date, date]:
    start = today - timedelta(days=today.weekday())
    return start, start + timedelta(days=6)


def previous_week_dates(today: date) -> tuple[date, date]:
    current_start, _ = current_week_dates(today)
    start = current_start - timedelta(days=7)
    return start, start + timedelta(days=6)


def current_month_dates(today: date) -> tuple[date, date]:
    end_day = calendar.monthrange(today.year, today.month)[1]
    return today.replace(day=1), today.replace(day=end_day)


def previous_month_dates(today: date) -> tuple[date, date]:
    first_this_month = today.replace(day=1)
    last_previous_month = first_this_month - timedelta(days=1)
    return current_month_dates(last_previous_month)


def previous_comparable_period_dates(
    *, start_date: date, end_date: date, period_type: str
) -> tuple[date, date]:
    if period_type == "monthly":
        return previous_month_dates(start_date)
    else:
        span = end_date - start_date
        previous_end = start_date - timedelta(days=1)
        previous_start = previous_end - span
        return previous_start, previous_end


def _entries_query_text(entries: Sequence[Entry], *, label: str) -> str:
    lines = [label]
    for entry in entries[-40:]:
        raw_text = " ".join((entry.raw_text or "").split())
        if not raw_text:
            continue
        timestamp = entry.local_timestamp or entry.created_at
        time_text = timestamp.isoformat() if timestamp else "unknown-time"
        lines.append(f"- {time_text} [{entry.source}] {_truncate_text(raw_text, 220)}")
    return "\n".join(lines)


def _semantic_records_context(records: list[EmbeddingRecord]) -> list[dict[str, Any]]:
    return [
        {
            "target_type": record.target_type,
            "target_id": str(record.target_id),
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "source_hash": record.source_hash,
            "source_text": _truncate_text(record.source_text, 700),
        }
        for record in records
    ]


def _truncate_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


async def _apply_evening_review_patches(session, *, settings, user, entries, analyses, patches, model_run_id) -> dict[str, dict[str, Any]]:
    entries_by_id = {str(entry.id): entry for entry in entries}
    latest = {str(item.target_id): item.result for item in analyses if item.task_name == ENTRY_FEATURES_TASK and isinstance(item.result, dict)}
    allowed = {"emotions", "affective_states", "emotion_labels", "mentioned_but_not_felt", "emotion_observation", "emotion_transition", "emotion_transition_confidence", "emotion_needs_clarification", "uncertainty_notes"}
    applied: dict[str, dict[str, Any]] = {}
    for patch in patches:
        entry, base = entries_by_id.get(patch.entry_id), latest.get(patch.entry_id)
        updates = {key: value for key, value in patch.patch.items() if key in allowed}
        if entry is None or base is None or patch.confidence < 0.8 or not patch.evidence or not updates:
            continue
        features = postprocess_entry_features(EntryFeatures.model_validate({**base, **updates}), entry.raw_text or "")
        await repo.add_ai_analysis(session, user_id=user.id, target_type="entry", target_id=entry.id, task_name=ENTRY_FEATURES_TASK, schema_version=ENTRY_FEATURES_SCHEMA_VERSION, provider=settings.ai_provider, model=settings.ai_heavy_model, result=features.model_dump(), confidence=Decimal(str(features.confidence)), uncertainty_notes=[*features.uncertainty_notes, "evening_reviewer_patch"], model_run_id=model_run_id)
        applied[patch.entry_id] = features.model_dump()
    return applied


def _replace_context_entry_analyses(context: dict[str, Any], patched: dict[str, dict[str, Any]]) -> None:
    for item in context.get("entries", []):
        entry_id = str(item.get("id") or "")
        result = patched.get(entry_id)
        if result is None:
            continue
        analyses = item.get("analyses") or []
        item["analyses"] = [
            analysis for analysis in analyses if analysis.get("task") != ENTRY_FEATURES_TASK
        ] + [{"task": ENTRY_FEATURES_TASK, "result": result, "confidence": result.get("confidence")}]


async def _queue_evening_review_questions(
    session,
    *,
    user,
    user_settings,
    entries,
    review,
    model_run_id,
) -> list[str]:
    entries_by_id = {str(entry.id): entry for entry in entries}
    queue = clarification_queue(user_settings)
    queued_ids: list[str] = []
    for candidate in review.question_candidates:
        entry = entries_by_id.get(candidate.entry_id)
        question = " ".join(candidate.question.split())
        if (
            entry is None
            or candidate.confidence < 0.8
            or not candidate.evidence
            or not question
            or any(
                item.get("entry_id") == candidate.entry_id
                and item.get("status") in {"queued", "active"}
                for item in queue
            )
        ):
            continue
        queue.append(
            {
                "id": str(uuid.uuid4()),
                "entry_id": candidate.entry_id,
                "question": question[:600],
                "options": [
                    " ".join(str(option).split())[:80]
                    for option in candidate.options[:4]
                    if str(option).strip()
                ],
                "reason": "evening_reviewer",
                "expected_gain": candidate.expected_gain,
                "evidence": candidate.evidence[:600],
                "model_run_id": str(model_run_id) if model_run_id else None,
                "status": "queued",
                "created_at": utc_now().isoformat(),
                "source": "evening_reviewer",
            }
        )
        queued_ids.append(candidate.entry_id)
    if queued_ids:
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_with_clarification_queue(user_settings, queue)},
        )
    return queued_ids
