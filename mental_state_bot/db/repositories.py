from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.db.models import (
    AIAnalysis,
    Day,
    EmbeddingRecord,
    Entry,
    ExportJob,
    Media,
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


async def get_snapshot_prompts(
    session: AsyncSession, *, snapshot_id: uuid.UUID
) -> Sequence[SnapshotPrompt]:
    result = await session.execute(
        select(SnapshotPrompt)
        .where(SnapshotPrompt.snapshot_id == snapshot_id)
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


async def list_day_entries(session: AsyncSession, *, day_id: uuid.UUID) -> Sequence[Entry]:
    result = await session.execute(select(Entry).where(Entry.day_id == day_id).order_by(Entry.created_at))
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
    limit: int = 100,
) -> Sequence[Entry]:
    result = await session.execute(
        select(Entry).where(Entry.user_id == user_id).order_by(Entry.created_at).limit(limit)
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
