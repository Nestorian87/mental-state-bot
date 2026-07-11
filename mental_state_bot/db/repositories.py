from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, delete, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.db.models import (
    AIAnalysis,
    Day,
    EmbeddingRecord,
    Entry,
    ExportJob,
    Media,
    MemoryEdge,
    MemoryEvidence,
    MemoryNode,
    MissedPrompt,
    ModelRun,
    RetrievalLog,
    Snapshot,
    SnapshotPrompt,
    Summary,
    User,
    UserSettings,
)
from mental_state_bot.time_utils import utc_now


async def get_or_create_user(
    session: AsyncSession,
    *,
    telegram_user_id: int,
    chat_id: int,
    username: str | None,
    first_name: str | None,
    timezone: str,
) -> User:
    result = await session.execute(select(User).where(User.telegram_user_id == telegram_user_id))
    user = result.scalar_one_or_none()
    now = utc_now()
    if user is None:
        user = User(
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            username=username,
            first_name=first_name,
            timezone=timezone,
            last_seen_at=now,
        )
        session.add(user)
        await session.flush()
        session.add(UserSettings(user_id=user.id))
        await session.flush()
        return user

    user.chat_id = chat_id
    user.username = username
    user.first_name = first_name
    user.last_seen_at = now
    await session.flush()
    return user


async def get_user_by_telegram_id(session: AsyncSession, telegram_user_id: int) -> User | None:
    result = await session.execute(select(User).where(User.telegram_user_id == telegram_user_id))
    return result.scalar_one_or_none()


async def list_active_users(session: AsyncSession) -> Sequence[User]:
    result = await session.execute(select(User).where(User.is_active.is_(True)))
    return result.scalars().all()


async def set_user_active(session: AsyncSession, *, user_id: uuid.UUID, is_active: bool) -> None:
    await session.execute(update(User).where(User.id == user_id).values(is_active=is_active))


async def get_user_settings(session: AsyncSession, user_id: uuid.UUID) -> UserSettings:
    result = await session.execute(select(UserSettings).where(UserSettings.user_id == user_id))
    settings = result.scalar_one_or_none()
    if settings is None:
        settings = UserSettings(user_id=user_id)
        session.add(settings)
        await session.flush()
    return settings


async def update_user_settings(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    values: dict[str, Any],
) -> UserSettings:
    settings = await get_user_settings(session, user_id)
    for key, value in values.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    await session.flush()
    return settings


async def get_or_create_day(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    local_date_value: date,
    started_at: datetime | None = None,
    boundary_kind: str = "calendar",
) -> Day:
    result = await session.execute(
        select(Day).where(and_(Day.user_id == user_id, Day.local_date == local_date_value))
    )
    day = result.scalar_one_or_none()
    if day is None:
        day = Day(
            user_id=user_id,
            local_date=local_date_value,
            started_at=started_at,
            boundary_kind=boundary_kind,
        )
        session.add(day)
        await session.flush()
    return day


async def get_day_by_date(
    session: AsyncSession, *, user_id: uuid.UUID, local_date_value: date
) -> Day | None:
    result = await session.execute(
        select(Day).where(and_(Day.user_id == user_id, Day.local_date == local_date_value))
    )
    return result.scalar_one_or_none()


async def get_day(session: AsyncSession, *, day_id: uuid.UUID) -> Day | None:
    return await session.get(Day, day_id)


async def list_days_between(
    session: AsyncSession, *, user_id: uuid.UUID, start_date: date, end_date: date
) -> Sequence[Day]:
    result = await session.execute(
        select(Day)
        .where(Day.user_id == user_id, Day.local_date >= start_date, Day.local_date <= end_date)
        .order_by(Day.local_date)
    )
    return result.scalars().all()


async def close_day(
    session: AsyncSession,
    *,
    day_id: uuid.UUID,
    ended_at: datetime,
    boundary_kind: str,
    data_quality: str | None = None,
) -> None:
    values: dict[str, Any] = {"ended_at": ended_at, "boundary_kind": boundary_kind}
    if data_quality is not None:
        values["data_quality"] = data_quality
    await session.execute(
        update(Day).where(Day.id == day_id).values(**values)
    )


async def reopen_day(session: AsyncSession, *, day_id: uuid.UUID) -> None:
    await session.execute(
        update(Day)
        .where(Day.id == day_id)
        .values(ended_at=None, boundary_kind="calendar", data_quality=None)
    )


async def create_snapshot(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    day_id: uuid.UUID | None,
    intent: str,
    scheduled_for: datetime | None,
    context: dict[str, Any] | None = None,
) -> Snapshot:
    snapshot = Snapshot(
        user_id=user_id,
        day_id=day_id,
        intent=intent,
        scheduled_for=scheduled_for,
        context_json=context or {},
    )
    session.add(snapshot)
    await session.flush()
    return snapshot


async def mark_snapshot_prompted(session: AsyncSession, *, snapshot_id: uuid.UUID, prompted_at: datetime) -> None:
    await session.execute(
        update(Snapshot).where(Snapshot.id == snapshot_id).values(status="prompted", prompted_at=prompted_at)
    )


async def mark_snapshot_in_progress(session: AsyncSession, *, snapshot_id: uuid.UUID) -> None:
    await session.execute(update(Snapshot).where(Snapshot.id == snapshot_id).values(status="in_progress"))


async def close_snapshot(session: AsyncSession, *, snapshot_id: uuid.UUID, status: str = "closed") -> None:
    await session.execute(
        update(Snapshot).where(Snapshot.id == snapshot_id).values(status=status, closed_at=utc_now())
    )


async def increment_clarification_count(session: AsyncSession, *, snapshot_id: uuid.UUID) -> None:
    snapshot = await session.get(Snapshot, snapshot_id)
    if snapshot is not None:
        snapshot.clarification_count += 1
        await session.flush()


async def get_open_snapshot(session: AsyncSession, *, user_id: uuid.UUID) -> Snapshot | None:
    result = await session.execute(
        select(Snapshot)
        .where(Snapshot.user_id == user_id, Snapshot.status.in_(["prompted", "in_progress"]))
        .order_by(desc(Snapshot.created_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_last_snapshot(session: AsyncSession, *, user_id: uuid.UUID) -> Snapshot | None:
    result = await session.execute(
        select(Snapshot)
        .where(Snapshot.user_id == user_id, Snapshot.prompted_at.is_not(None))
        .order_by(desc(Snapshot.prompted_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def list_recent_snapshots(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    limit: int = 12,
) -> Sequence[Snapshot]:
    result = await session.execute(
        select(Snapshot)
        .where(Snapshot.user_id == user_id, Snapshot.prompted_at.is_not(None))
        .order_by(desc(Snapshot.prompted_at))
        .limit(limit)
    )
    return result.scalars().all()


async def get_snapshot_prompts(
    session: AsyncSession, *, snapshot_id: uuid.UUID
) -> Sequence[SnapshotPrompt]:
    result = await session.execute(
        select(SnapshotPrompt)
        .where(SnapshotPrompt.snapshot_id == snapshot_id)
        .order_by(SnapshotPrompt.sent_at)
    )
    return result.scalars().all()


async def list_prompts_for_snapshots(
    session: AsyncSession, *, snapshot_ids: Sequence[uuid.UUID]
) -> Sequence[SnapshotPrompt]:
    if not snapshot_ids:
        return []
    result = await session.execute(
        select(SnapshotPrompt)
        .where(SnapshotPrompt.snapshot_id.in_(snapshot_ids))
        .order_by(SnapshotPrompt.sent_at)
    )
    return result.scalars().all()


async def add_prompt(
    session: AsyncSession,
    *,
    snapshot_id: uuid.UUID,
    prompt_kind: str,
    text: str,
    sent_at: datetime,
    telegram_message_id: int | None,
    model_run_id: uuid.UUID | None = None,
) -> SnapshotPrompt:
    prompt = SnapshotPrompt(
        snapshot_id=snapshot_id,
        prompt_kind=prompt_kind,
        text=text,
        sent_at=sent_at,
        telegram_message_id=telegram_message_id,
        model_run_id=model_run_id,
    )
    session.add(prompt)
    await session.flush()
    return prompt


async def add_entry(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    day_id: uuid.UUID | None,
    snapshot_id: uuid.UUID | None,
    source: str,
    raw_text: str | None,
    telegram_message_id: int | None,
    reply_to_message_id: int | None,
    local_timestamp: datetime | None,
    meta: dict[str, Any] | None = None,
) -> Entry:
    entry = Entry(
        user_id=user_id,
        day_id=day_id,
        snapshot_id=snapshot_id,
        source=source,
        raw_text=raw_text,
        normalized_text=raw_text.strip() if raw_text else None,
        telegram_message_id=telegram_message_id,
        reply_to_message_id=reply_to_message_id,
        local_timestamp=local_timestamp,
        meta=meta or {},
    )
    session.add(entry)
    await session.flush()
    return entry


async def get_entry(session: AsyncSession, *, entry_id: uuid.UUID) -> Entry | None:
    return await session.get(Entry, entry_id)


async def delete_entry_tree(session: AsyncSession, *, entry_id: uuid.UUID, user_id: uuid.UUID) -> Entry | None:
    entry = await session.get(Entry, entry_id)
    if entry is None or entry.user_id != user_id:
        return None
    await session.execute(
        delete(AIAnalysis).where(AIAnalysis.target_type == "entry", AIAnalysis.target_id == entry_id)
    )
    await session.execute(
        delete(EmbeddingRecord).where(EmbeddingRecord.target_type == "entry", EmbeddingRecord.target_id == entry_id)
    )
    await session.execute(delete(Media).where(Media.entry_id == entry_id))
    await session.delete(entry)
    await session.flush()
    return entry


async def list_entries_by_ids(session: AsyncSession, *, entry_ids: Sequence[uuid.UUID]) -> Sequence[Entry]:
    if not entry_ids:
        return []
    result = await session.execute(select(Entry).where(Entry.id.in_(entry_ids)))
    return result.scalars().all()


async def add_media(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    entry_id: uuid.UUID | None,
    media_type: str,
    telegram_file_id: str | None,
    telegram_file_unique_id: str | None,
    file_path: str | None,
    meta: dict[str, Any] | None = None,
) -> Media:
    media = Media(
        user_id=user_id,
        entry_id=entry_id,
        media_type=media_type,
        telegram_file_id=telegram_file_id,
        telegram_file_unique_id=telegram_file_unique_id,
        file_path=file_path,
        meta=meta or {},
    )
    session.add(media)
    await session.flush()
    return media


async def get_recent_entries(
    session: AsyncSession, *, user_id: uuid.UUID, limit: int = 8
) -> Sequence[Entry]:
    result = await session.execute(
        select(Entry).where(Entry.user_id == user_id).order_by(desc(Entry.created_at)).limit(limit)
    )
    return list(reversed(result.scalars().all()))


async def get_latest_observation_entry(session: AsyncSession, *, user_id: uuid.UUID) -> Entry | None:
    result = await session.execute(
        select(Entry)
        .where(
            Entry.user_id == user_id,
            Entry.source.notin_(("correction", "profile_context_update")),
        )
        .order_by(desc(Entry.created_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def list_day_entries(session: AsyncSession, *, day_id: uuid.UUID) -> Sequence[Entry]:
    result = await session.execute(select(Entry).where(Entry.day_id == day_id).order_by(Entry.created_at))
    return result.scalars().all()


async def list_snapshot_entries(session: AsyncSession, *, snapshot_id: uuid.UUID) -> Sequence[Entry]:
    result = await session.execute(
        select(Entry).where(Entry.snapshot_id == snapshot_id).order_by(Entry.created_at)
    )
    return result.scalars().all()


async def count_user_rows(session: AsyncSession, model, *, user_id: uuid.UUID) -> int:
    result = await session.execute(select(func.count(model.id)).where(model.user_id == user_id))
    return int(result.scalar_one() or 0)


async def list_entries_without_embedding(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    embedding_model: str,
    limit: int = 100,
) -> Sequence[Entry]:
    result = await session.execute(
        select(Entry)
        .outerjoin(
            EmbeddingRecord,
            and_(
                EmbeddingRecord.target_type == "entry",
                EmbeddingRecord.target_id == Entry.id,
                EmbeddingRecord.model == embedding_model,
            ),
        )
        .where(Entry.user_id == user_id, EmbeddingRecord.id.is_(None))
        .order_by(Entry.created_at)
        .limit(limit)
    )
    return result.scalars().all()


async def count_entries_without_embedding(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    embedding_model: str,
) -> int:
    result = await session.execute(
        select(func.count(Entry.id))
        .outerjoin(
            EmbeddingRecord,
            and_(
                EmbeddingRecord.target_type == "entry",
                EmbeddingRecord.target_id == Entry.id,
                EmbeddingRecord.model == embedding_model,
            ),
        )
        .where(Entry.user_id == user_id, EmbeddingRecord.id.is_(None))
    )
    return int(result.scalar_one() or 0)


async def list_entries_without_analysis(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    task_name: str,
    limit: int = 100,
) -> Sequence[Entry]:
    result = await session.execute(
        select(Entry)
        .outerjoin(
            AIAnalysis,
            and_(
                AIAnalysis.target_type == "entry",
                AIAnalysis.target_id == Entry.id,
                AIAnalysis.task_name == task_name,
            ),
        )
        .where(Entry.user_id == user_id, AIAnalysis.id.is_(None))
        .order_by(Entry.created_at)
        .limit(limit)
    )
    return result.scalars().all()


async def list_user_entries(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    limit: int | None = 100,
    descending: bool = False,
) -> Sequence[Entry]:
    order = Entry.created_at.desc() if descending else Entry.created_at
    result = await session.execute(select(Entry).where(Entry.user_id == user_id).order_by(order).limit(limit))
    return result.scalars().all()


async def list_entries_for_journal_dates(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    start_date: date,
    end_date: date,
) -> Sequence[Entry]:
    result = await session.execute(
        select(Entry)
        .join(Day, Entry.day_id == Day.id)
        .where(
            Entry.user_id == user_id,
            Day.local_date >= start_date,
            Day.local_date <= end_date,
        )
        .order_by(Entry.created_at)
    )
    return result.scalars().all()


async def count_entries_without_analysis(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    task_name: str,
) -> int:
    result = await session.execute(
        select(func.count(Entry.id))
        .outerjoin(
            AIAnalysis,
            and_(
                AIAnalysis.target_type == "entry",
                AIAnalysis.target_id == Entry.id,
                AIAnalysis.task_name == task_name,
            ),
        )
        .where(Entry.user_id == user_id, AIAnalysis.id.is_(None))
    )
    return int(result.scalar_one() or 0)


async def list_user_media(session: AsyncSession, *, user_id: uuid.UUID) -> Sequence[Media]:
    result = await session.execute(select(Media).where(Media.user_id == user_id).order_by(Media.created_at))
    return result.scalars().all()


async def list_day_media_with_entries(
    session: AsyncSession, *, day_id: uuid.UUID, media_type: str | None = None
) -> Sequence[tuple[Media, Entry]]:
    query = (
        select(Media, Entry)
        .join(Entry, Media.entry_id == Entry.id)
        .where(Entry.day_id == day_id)
        .order_by(Media.created_at)
    )
    if media_type:
        query = query.where(Media.media_type == media_type)
    result = await session.execute(query)
    return result.all()


async def list_entries_between(
    session: AsyncSession, *, user_id: uuid.UUID, start: datetime, end: datetime
) -> Sequence[Entry]:
    result = await session.execute(
        select(Entry)
        .where(and_(Entry.user_id == user_id, Entry.created_at >= start, Entry.created_at <= end))
        .order_by(Entry.created_at)
    )
    return result.scalars().all()


async def count_entries_for_day(session: AsyncSession, *, day_id: uuid.UUID) -> int:
    result = await session.execute(select(func.count(Entry.id)).where(Entry.day_id == day_id))
    return int(result.scalar_one())


async def add_missed_prompt(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    snapshot_id: uuid.UUID,
    prompt_id: uuid.UUID | None,
    missed_at: datetime,
) -> MissedPrompt:
    missed = MissedPrompt(
        user_id=user_id,
        snapshot_id=snapshot_id,
        prompt_id=prompt_id,
        missed_at=missed_at,
    )
    session.add(missed)
    await session.flush()
    return missed


async def get_latest_open_missed_prompt(
    session: AsyncSession, *, user_id: uuid.UUID
) -> MissedPrompt | None:
    result = await session.execute(
        select(MissedPrompt)
        .where(MissedPrompt.user_id == user_id, MissedPrompt.status == "open")
        .order_by(desc(MissedPrompt.missed_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def resolve_missed_prompt(
    session: AsyncSession,
    *,
    missed_prompt_id: uuid.UUID,
    reason_text: str,
    status: str = "explained",
) -> MissedPrompt | None:
    missed = await session.get(MissedPrompt, missed_prompt_id)
    if missed is None:
        return None
    missed.status = status
    missed.reason_text = reason_text
    missed.resolved_at = utc_now()
    await session.flush()
    return missed


async def list_missed_prompts_between(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    start: datetime,
    end: datetime,
) -> Sequence[MissedPrompt]:
    result = await session.execute(
        select(MissedPrompt)
        .where(
            MissedPrompt.user_id == user_id,
            MissedPrompt.missed_at >= start,
            MissedPrompt.missed_at <= end,
        )
        .order_by(MissedPrompt.missed_at)
    )
    return result.scalars().all()


async def list_prompts_due_for_miss_check(
    session: AsyncSession, *, older_than: datetime
) -> Sequence[SnapshotPrompt]:
    result = await session.execute(
        select(SnapshotPrompt)
        .join(Snapshot, Snapshot.id == SnapshotPrompt.snapshot_id)
        .where(
            Snapshot.status == "prompted",
            SnapshotPrompt.prompt_kind == "initial",
            SnapshotPrompt.sent_at <= older_than,
        )
        .order_by(SnapshotPrompt.sent_at)
    )
    return result.scalars().all()


async def create_model_run(
    session: AsyncSession,
    *,
    user_id: uuid.UUID | None,
    provider: str,
    model: str,
    task_name: str,
    status: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    total_tokens: int | None = None,
    estimated_cost_usd: Decimal | None = None,
    latency_ms: int | None = None,
    error_message: str | None = None,
    request_hash: str | None = None,
    meta: dict[str, Any] | None = None,
) -> ModelRun:
    run = ModelRun(
        user_id=user_id,
        provider=provider,
        model=model,
        task_name=task_name,
        status=status,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
        latency_ms=latency_ms,
        error_message=error_message,
        request_hash=request_hash,
        meta=meta or {},
    )
    session.add(run)
    await session.flush()
    return run


async def get_model_run(session: AsyncSession, *, model_run_id: uuid.UUID) -> ModelRun | None:
    return await session.get(ModelRun, model_run_id)


async def list_model_runs_since(
    session: AsyncSession, *, user_id: uuid.UUID, since: datetime
) -> Sequence[ModelRun]:
    result = await session.execute(
        select(ModelRun)
        .where(ModelRun.user_id == user_id, ModelRun.created_at >= since)
        .order_by(desc(ModelRun.created_at))
    )
    return result.scalars().all()


async def model_run_cost_totals(
    session: AsyncSession, *, user_id: uuid.UUID, since: datetime
) -> dict[str, Any]:
    result = await session.execute(
        select(
            func.count(ModelRun.id),
            func.coalesce(func.sum(ModelRun.estimated_cost_usd), 0),
            func.coalesce(func.sum(ModelRun.prompt_tokens), 0),
            func.coalesce(func.sum(ModelRun.completion_tokens), 0),
            func.coalesce(func.sum(ModelRun.reasoning_tokens), 0),
            func.coalesce(func.sum(ModelRun.total_tokens), 0),
        ).where(ModelRun.user_id == user_id, ModelRun.created_at >= since)
    )
    row = result.one()
    return {
        "runs": int(row[0] or 0),
        "estimated_cost_usd": row[1],
        "prompt_tokens": int(row[2] or 0),
        "completion_tokens": int(row[3] or 0),
        "reasoning_tokens": int(row[4] or 0),
        "total_tokens": int(row[5] or 0),
    }


async def summary_counts_by_type(session: AsyncSession, *, user_id: uuid.UUID) -> dict[str, int]:
    result = await session.execute(
        select(Summary.period_type, func.count(Summary.id))
        .where(Summary.user_id == user_id)
        .group_by(Summary.period_type)
    )
    return {str(period_type): int(count or 0) for period_type, count in result.all()}


async def snapshot_counts_by_status(session: AsyncSession, *, user_id: uuid.UUID) -> dict[str, int]:
    result = await session.execute(
        select(Snapshot.status, func.count(Snapshot.id))
        .where(Snapshot.user_id == user_id)
        .group_by(Snapshot.status)
    )
    return {str(status): int(count or 0) for status, count in result.all()}


async def missed_prompt_counts_by_status(session: AsyncSession, *, user_id: uuid.UUID) -> dict[str, int]:
    result = await session.execute(
        select(MissedPrompt.status, func.count(MissedPrompt.id))
        .where(MissedPrompt.user_id == user_id)
        .group_by(MissedPrompt.status)
    )
    return {str(status): int(count or 0) for status, count in result.all()}


async def add_ai_analysis(
    session: AsyncSession,
    *,
    user_id: uuid.UUID | None,
    target_type: str,
    target_id: uuid.UUID,
    task_name: str,
    schema_version: str,
    provider: str,
    model: str,
    result: dict[str, Any],
    confidence: Decimal | None,
    uncertainty_notes: list[str],
    model_run_id: uuid.UUID | None,
) -> AIAnalysis:
    analysis = AIAnalysis(
        user_id=user_id,
        target_type=target_type,
        target_id=target_id,
        task_name=task_name,
        schema_version=schema_version,
        provider=provider,
        model=model,
        result=result,
        confidence=confidence,
        uncertainty_notes=uncertainty_notes,
        model_run_id=model_run_id,
    )
    session.add(analysis)
    await session.flush()
    return analysis


async def list_analyses_for_targets(
    session: AsyncSession, *, target_type: str, target_ids: Sequence[uuid.UUID]
) -> Sequence[AIAnalysis]:
    if not target_ids:
        return []
    result = await session.execute(
        select(AIAnalysis)
        .where(AIAnalysis.target_type == target_type, AIAnalysis.target_id.in_(target_ids))
        .order_by(AIAnalysis.created_at)
    )
    return result.scalars().all()


async def add_summary(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    day_id: uuid.UUID | None,
    period_type: str,
    period_start: datetime,
    period_end: datetime,
    short_text: str,
    details: dict[str, Any],
    model_run_id: uuid.UUID | None,
) -> Summary:
    summary = Summary(
        user_id=user_id,
        day_id=day_id,
        period_type=period_type,
        period_start=period_start,
        period_end=period_end,
        short_text=short_text,
        details=details,
        model_run_id=model_run_id,
    )
    session.add(summary)
    await session.flush()
    return summary


async def upsert_summary(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    day_id: uuid.UUID | None,
    period_type: str,
    period_start: datetime,
    period_end: datetime,
    short_text: str,
    details: dict[str, Any],
    model_run_id: uuid.UUID | None,
) -> Summary:
    result = await session.execute(
        select(Summary)
        .where(
            Summary.user_id == user_id,
            Summary.period_type == period_type,
            Summary.period_start == period_start,
            Summary.period_end == period_end,
        )
        .limit(1)
    )
    summary = result.scalar_one_or_none()
    if summary is None:
        return await add_summary(
            session,
            user_id=user_id,
            day_id=day_id,
            period_type=period_type,
            period_start=period_start,
            period_end=period_end,
            short_text=short_text,
            details=details,
            model_run_id=model_run_id,
        )

    summary.day_id = day_id
    summary.short_text = short_text
    summary.details = details
    summary.model_run_id = model_run_id
    await session.flush()
    return summary


async def get_latest_summary(
    session: AsyncSession, *, user_id: uuid.UUID, period_type: str = "daily"
) -> Summary | None:
    result = await session.execute(
        select(Summary)
        .where(Summary.user_id == user_id, Summary.period_type == period_type)
        .order_by(desc(Summary.created_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_summary(session: AsyncSession, *, summary_id: uuid.UUID) -> Summary | None:
    return await session.get(Summary, summary_id)


async def get_period_summary(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    period_type: str,
    period_start: datetime,
    period_end: datetime,
) -> Summary | None:
    result = await session.execute(
        select(Summary)
        .where(
            Summary.user_id == user_id,
            Summary.period_type == period_type,
            Summary.period_start == period_start,
            Summary.period_end == period_end,
        )
        .limit(1)
    )
    return result.scalar_one_or_none()


async def mark_summary_delivered(session: AsyncSession, *, summary_id: uuid.UUID, delivered_at: datetime) -> None:
    await session.execute(update(Summary).where(Summary.id == summary_id).values(delivered_at=delivered_at))


async def get_day_summary(
    session: AsyncSession, *, user_id: uuid.UUID, day_id: uuid.UUID, period_type: str = "daily"
) -> Summary | None:
    result = await session.execute(
        select(Summary)
        .where(Summary.user_id == user_id, Summary.day_id == day_id, Summary.period_type == period_type)
        .order_by(desc(Summary.created_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def mark_day_summaries_stale(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    day_id: uuid.UUID,
    reason: str,
) -> int:
    result = await session.execute(
        select(Summary).where(Summary.user_id == user_id, Summary.day_id == day_id, Summary.period_type == "daily")
    )
    summaries = result.scalars().all()
    now = utc_now().isoformat()
    for summary in summaries:
        summary.details = {
            **(summary.details or {}),
            "stale": {
                "reason": reason,
                "marked_at": now,
            },
        }
    await session.flush()
    return len(summaries)


async def list_summaries_between(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    period_type: str,
    start: datetime,
    end: datetime,
) -> Sequence[Summary]:
    result = await session.execute(
        select(Summary)
        .where(
            Summary.user_id == user_id,
            Summary.period_type == period_type,
            Summary.period_start >= start,
            Summary.period_end <= end,
        )
        .order_by(Summary.period_start)
    )
    return result.scalars().all()


async def summary_exists(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    day_id: uuid.UUID | None,
    period_type: str,
) -> bool:
    result = await session.execute(
        select(Summary.id)
        .where(Summary.user_id == user_id, Summary.day_id == day_id, Summary.period_type == period_type)
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def period_summary_exists(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    period_type: str,
    period_start: datetime,
    period_end: datetime,
) -> bool:
    result = await session.execute(
        select(Summary.id)
        .where(
            Summary.user_id == user_id,
            Summary.period_type == period_type,
            Summary.period_start == period_start,
            Summary.period_end == period_end,
        )
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def add_embedding(
    session: AsyncSession,
    *,
    user_id: uuid.UUID | None,
    target_type: str,
    target_id: uuid.UUID,
    provider: str,
    model: str,
    dimensions: int,
    source_hash: str,
    source_text: str,
    embedding: list[float],
) -> EmbeddingRecord:
    record = EmbeddingRecord(
        user_id=user_id,
        target_type=target_type,
        target_id=target_id,
        provider=provider,
        model=model,
        dimensions=dimensions,
        source_hash=source_hash,
        source_text=source_text,
        embedding=embedding,
    )
    session.add(record)
    await session.flush()
    return record


async def delete_embeddings_for_target_model(
    session: AsyncSession,
    *,
    target_type: str,
    target_id: uuid.UUID,
    provider: str,
    model: str,
) -> None:
    await session.execute(
        delete(EmbeddingRecord).where(
            EmbeddingRecord.target_type == target_type,
            EmbeddingRecord.target_id == target_id,
            EmbeddingRecord.provider == provider,
            EmbeddingRecord.model == model,
        )
    )


async def find_similar_embeddings(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    embedding: list[float],
    limit: int = 8,
) -> Sequence[EmbeddingRecord]:
    result = await session.execute(
        select(EmbeddingRecord)
        .where(EmbeddingRecord.user_id == user_id)
        .order_by(EmbeddingRecord.embedding.cosine_distance(embedding))
        .limit(limit)
    )
    return result.scalars().all()


async def get_memory_nodes_by_normalized_labels(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    labels: Sequence[str],
) -> Sequence[MemoryNode]:
    if not labels:
        return []
    result = await session.execute(
        select(MemoryNode).where(MemoryNode.user_id == user_id, MemoryNode.normalized_label.in_(labels))
    )
    return result.scalars().all()


async def list_memory_nodes(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    limit: int = 500,
) -> Sequence[MemoryNode]:
    result = await session.execute(
        select(MemoryNode)
        .where(MemoryNode.user_id == user_id, MemoryNode.status.in_(["candidate", "hypothesis", "confirmed"]))
        .order_by(desc(MemoryNode.weight), desc(MemoryNode.confidence), MemoryNode.label)
        .limit(limit)
    )
    return result.scalars().all()


async def list_memory_nodes_for_export(session: AsyncSession, *, user_id: uuid.UUID) -> Sequence[MemoryNode]:
    result = await session.execute(
        select(MemoryNode).where(MemoryNode.user_id == user_id).order_by(MemoryNode.created_at)
    )
    return result.scalars().all()


async def list_memory_edges(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    limit: int = 1000,
) -> Sequence[MemoryEdge]:
    result = await session.execute(
        select(MemoryEdge)
        .where(MemoryEdge.user_id == user_id, MemoryEdge.status.in_(["candidate", "hypothesis", "confirmed"]))
        .order_by(desc(MemoryEdge.weight), desc(MemoryEdge.confidence), MemoryEdge.relation_label)
        .limit(limit)
    )
    return result.scalars().all()


async def list_memory_edges_for_export(session: AsyncSession, *, user_id: uuid.UUID) -> Sequence[MemoryEdge]:
    result = await session.execute(
        select(MemoryEdge).where(MemoryEdge.user_id == user_id).order_by(MemoryEdge.created_at)
    )
    return result.scalars().all()


async def list_memory_evidence_for_export(session: AsyncSession, *, user_id: uuid.UUID) -> Sequence[MemoryEvidence]:
    result = await session.execute(
        select(MemoryEvidence).where(MemoryEvidence.user_id == user_id).order_by(MemoryEvidence.created_at)
    )
    return result.scalars().all()


async def list_situation_nodes_for_entry_targets(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    entry_ids: Sequence[uuid.UUID],
) -> Sequence[tuple[uuid.UUID, MemoryNode]]:
    if not entry_ids:
        return []
    result = await session.execute(
        select(MemoryEvidence.target_id, MemoryNode)
        .join(MemoryNode, MemoryNode.id == MemoryEvidence.node_id)
        .where(
            MemoryEvidence.user_id == user_id,
            MemoryEvidence.target_type == "entry",
            MemoryEvidence.target_id.in_(entry_ids),
            MemoryNode.kind == "situation",
        )
        .order_by(desc(MemoryEvidence.created_at))
    )
    return result.all()


async def delete_memory_graph(session: AsyncSession, *, user_id: uuid.UUID) -> None:
    await session.execute(delete(MemoryEvidence).where(MemoryEvidence.user_id == user_id))
    await session.execute(delete(MemoryEdge).where(MemoryEdge.user_id == user_id))
    await session.execute(delete(MemoryNode).where(MemoryNode.user_id == user_id))


async def add_memory_node(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    label: str,
    normalized_label: str,
    kind: str,
    aliases: list[str],
    summary: str | None,
    confidence: Decimal | None,
    weight: Decimal | None,
    status: str,
    last_seen_at: datetime | None,
    meta: dict[str, Any] | None = None,
) -> MemoryNode:
    node = MemoryNode(
        user_id=user_id,
        label=label,
        normalized_label=normalized_label,
        kind=kind,
        aliases=aliases,
        summary=summary,
        confidence=confidence,
        weight=weight,
        status=status,
        last_seen_at=last_seen_at,
        meta=meta or {},
    )
    session.add(node)
    await session.flush()
    return node


async def get_memory_edge(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    source_node_id: uuid.UUID,
    target_node_id: uuid.UUID,
    relation_label: str,
) -> MemoryEdge | None:
    result = await session.execute(
        select(MemoryEdge).where(
            MemoryEdge.user_id == user_id,
            MemoryEdge.source_node_id == source_node_id,
            MemoryEdge.target_node_id == target_node_id,
            MemoryEdge.relation_label == relation_label,
        )
    )
    return result.scalar_one_or_none()


async def add_memory_edge(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    source_node_id: uuid.UUID,
    target_node_id: uuid.UUID,
    relation_label: str,
    summary: str | None,
    confidence: Decimal | None,
    weight: Decimal | None,
    status: str,
    evidence_count: int = 0,
    last_seen_at: datetime | None = None,
    meta: dict[str, Any] | None = None,
) -> MemoryEdge:
    edge = MemoryEdge(
        user_id=user_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        relation_label=relation_label,
        summary=summary,
        confidence=confidence,
        weight=weight,
        status=status,
        evidence_count=evidence_count,
        last_seen_at=last_seen_at,
        meta=meta or {},
    )
    session.add(edge)
    await session.flush()
    return edge


async def add_memory_evidence(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    node_id: uuid.UUID | None,
    edge_id: uuid.UUID | None,
    target_type: str,
    target_id: uuid.UUID,
    evidence_text: str,
    confidence: Decimal | None,
    meta: dict[str, Any] | None = None,
) -> MemoryEvidence:
    evidence = MemoryEvidence(
        user_id=user_id,
        node_id=node_id,
        edge_id=edge_id,
        target_type=target_type,
        target_id=target_id,
        evidence_text=evidence_text,
        confidence=confidence,
        meta=meta or {},
    )
    session.add(evidence)
    await session.flush()
    return evidence


async def list_memory_edges_for_nodes(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    node_ids: Sequence[uuid.UUID],
    limit: int = 40,
) -> Sequence[MemoryEdge]:
    if not node_ids:
        return []
    result = await session.execute(
        select(MemoryEdge)
        .where(
            MemoryEdge.user_id == user_id,
            MemoryEdge.status.in_(["candidate", "hypothesis", "confirmed"]),
            (MemoryEdge.source_node_id.in_(node_ids) | MemoryEdge.target_node_id.in_(node_ids)),
        )
        .order_by(desc(MemoryEdge.weight), desc(MemoryEdge.confidence), desc(MemoryEdge.last_seen_at))
        .limit(limit)
    )
    return result.scalars().all()


async def add_retrieval_log(
    session: AsyncSession,
    *,
    user_id: uuid.UUID | None,
    task_name: str,
    query_text: str,
    provider: str,
    model: str,
    retrieved: list[dict[str, Any]],
) -> RetrievalLog:
    log = RetrievalLog(
        user_id=user_id,
        task_name=task_name,
        query_text=query_text,
        provider=provider,
        model=model,
        retrieved=retrieved,
    )
    session.add(log)
    await session.flush()
    return log


async def add_export_job(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    status: str,
    format: str,
    file_path: str | None,
    meta: dict[str, Any] | None = None,
) -> ExportJob:
    job = ExportJob(user_id=user_id, status=status, format=format, file_path=file_path, meta=meta or {})
    session.add(job)
    await session.flush()
    return job


def reminder_cutoff(now: datetime, delay_minutes: int) -> datetime:
    return now - timedelta(minutes=delay_minutes)
