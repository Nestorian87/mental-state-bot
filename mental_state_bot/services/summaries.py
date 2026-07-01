from __future__ import annotations

import calendar
import logging
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.ai.service import AIService
from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import Day, EmbeddingRecord, Entry, Summary, User
from mental_state_bot.services.preferences import custom_interaction_style, user_profile_context
from mental_state_bot.time_utils import local_date, local_now, utc_now, zoneinfo

logger = logging.getLogger(__name__)

AUTO_MORNING_BOUNDARY_KIND = "auto_morning"
UNCERTAIN_DAY_BOUNDARY_QUALITY = "day_boundary_uncertain"


class SummaryService:
    def __init__(self, settings: Settings, ai_service: AIService) -> None:
        self.settings = settings
        self.ai = ai_service

    async def generate_today_summary(self, session: AsyncSession, *, user: User) -> Summary:
        day = await repo.get_or_create_day(
            session,
            user_id=user.id,
            local_date_value=local_date(user.timezone),
            started_at=utc_now(),
        )
        return await self.generate_day_summary(session, user=user, day=day)

    async def close_today_with_summary(self, session: AsyncSession, *, user: User) -> Summary:
        day = await repo.get_or_create_day(
            session,
            user_id=user.id,
            local_date_value=local_date(user.timezone),
            started_at=utc_now(),
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
            local_timestamp=local_now(user.timezone),
            meta={"boundary_kind": "sleep_marker"},
        )
        return await self.generate_day_summary(session, user=user, day=day, close_day=True)

    async def generate_yesterday_summary_if_needed(
        self, session: AsyncSession, *, user: User
    ) -> Summary | None:
        yesterday = local_date(user.timezone) - timedelta(days=1)
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
        period_start, period_end = _day_period_bounds(day, user.timezone)
        user_settings = await repo.get_user_settings(session, user.id)
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
        daily, model_run_id = await self.ai.generate_daily_summary(
            session,
            user_id=user.id,
            context=context,
        )
        summary = await repo.upsert_summary(
            session,
            user_id=user.id,
            day_id=day.id,
            period_type="daily",
            period_start=period_start,
            period_end=period_end,
            short_text=daily.short_text,
            details=daily.model_dump(),
            model_run_id=model_run_id,
        )
        return summary

    async def generate_current_week_summary(self, session: AsyncSession, *, user: User) -> Summary:
        start_date, end_date = current_week_dates(local_date(user.timezone))
        return await self.generate_period_summary(
            session,
            user=user,
            period_type="weekly",
            start_date=start_date,
            end_date=end_date,
        )

    async def generate_current_month_summary(self, session: AsyncSession, *, user: User) -> Summary:
        start_date, end_date = current_month_dates(local_date(user.timezone))
        return await self.generate_period_summary(
            session,
            user=user,
            period_type="monthly",
            start_date=start_date,
            end_date=end_date,
        )

    async def generate_previous_week_summary(self, session: AsyncSession, *, user: User) -> Summary:
        start_date, end_date = previous_week_dates(local_date(user.timezone))
        return await self.generate_period_summary(
            session,
            user=user,
            period_type="weekly",
            start_date=start_date,
            end_date=end_date,
        )

    async def generate_previous_month_summary(self, session: AsyncSession, *, user: User) -> Summary:
        start_date, end_date = previous_month_dates(local_date(user.timezone))
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
        start_date, end_date = previous_week_dates(local_date(user.timezone))
        period_start, period_end = _date_period_bounds(start_date, end_date, user.timezone)
        if await repo.period_summary_exists(
            session,
            user_id=user.id,
            period_type="weekly",
            period_start=period_start,
            period_end=period_end,
        ):
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
        start_date, end_date = previous_month_dates(local_date(user.timezone))
        period_start, period_end = _date_period_bounds(start_date, end_date, user.timezone)
        if await repo.period_summary_exists(
            session,
            user_id=user.id,
            period_type="monthly",
            period_start=period_start,
            period_end=period_end,
        ):
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
        period_start, period_end = _date_period_bounds(start_date, end_date, user.timezone)
        user_settings = await repo.get_user_settings(session, user.id)
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
        previous_start, previous_end = previous_comparable_period(
            start_date=start_date,
            end_date=end_date,
            period_type=period_type,
            timezone=user.timezone,
        )
        previous_summaries = await repo.list_summaries_between(
            session,
            user_id=user.id,
            period_type="daily",
            start=previous_start,
            end=previous_end,
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
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "user_profile_context": user_profile_context(user_settings),
            "style": {
                "custom_interaction_style": custom_interaction_style(user_settings),
            },
            "entry_count": len(entries),
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
        return await repo.upsert_summary(
            session,
            user_id=user.id,
            day_id=None,
            period_type=period_type,
            period_start=period_start,
            period_end=period_end,
            short_text=period_summary.short_text,
            details=period_summary.model_dump(),
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


def _day_period_bounds(day: Day, timezone: str) -> tuple[datetime, datetime]:
    tz = zoneinfo(timezone)
    start = datetime.combine(day.local_date, time.min, tzinfo=tz)
    end = datetime.combine(day.local_date, time.max, tzinfo=tz)
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


def _date_period_bounds(start_date: date, end_date: date, timezone: str) -> tuple[datetime, datetime]:
    tz = zoneinfo(timezone)
    start = datetime.combine(start_date, time.min, tzinfo=tz)
    end = datetime.combine(end_date, time.max, tzinfo=tz)
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


def previous_comparable_period(
    *, start_date: date, end_date: date, period_type: str, timezone: str
) -> tuple[datetime, datetime]:
    if period_type == "monthly":
        previous_start, previous_end = previous_month_dates(start_date)
    else:
        span = end_date - start_date
        previous_end = start_date - timedelta(days=1)
        previous_start = previous_end - span
    return _date_period_bounds(previous_start, previous_end, timezone)


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
