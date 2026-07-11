from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
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
    User,
)


@dataclass(frozen=True)
class ArchiveAudit:
    days: int
    entries: int
    snapshots: int
    prompts: int
    media: int
    missing_media_files: int
    missed_prompts: int
    ai_analyses: int
    feature_analyses_missing: int
    embeddings: int
    embeddings_missing: int | None
    summaries: dict[str, int]
    snapshot_statuses: dict[str, int]
    missed_prompt_statuses: dict[str, int]
    model_runs: int
    retrieval_logs: int
    exports: int


async def build_archive_audit(
    session: AsyncSession,
    *,
    settings: Settings,
    user: User,
) -> ArchiveAudit:
    media_items = list(await repo.list_user_media(session, user_id=user.id))
    entries = await repo.count_user_rows(session, Entry, user_id=user.id)
    embeddings_missing = None
    if settings.embeddings_enabled:
        embeddings_missing = await repo.count_entries_without_embedding(
            session,
            user_id=user.id,
            embedding_model=settings.embedding_model,
        )
    return ArchiveAudit(
        days=await repo.count_user_rows(session, Day, user_id=user.id),
        entries=entries,
        snapshots=await repo.count_user_rows(session, Snapshot, user_id=user.id),
        prompts=await _count_prompts_for_user(session, user_id=user.id),
        media=len(media_items),
        missing_media_files=count_missing_media_files(media_items),
        missed_prompts=await repo.count_user_rows(session, MissedPrompt, user_id=user.id),
        ai_analyses=await repo.count_user_rows(session, AIAnalysis, user_id=user.id),
        feature_analyses_missing=await repo.count_entries_without_analysis(
            session,
            user_id=user.id,
            task_name="extract_entry_features",
        ),
        embeddings=await repo.count_user_rows(session, EmbeddingRecord, user_id=user.id),
        embeddings_missing=embeddings_missing,
        summaries=await repo.summary_counts_by_type(session, user_id=user.id),
        snapshot_statuses=await repo.snapshot_counts_by_status(session, user_id=user.id),
        missed_prompt_statuses=await repo.missed_prompt_counts_by_status(session, user_id=user.id),
        model_runs=await repo.count_user_rows(session, ModelRun, user_id=user.id),
        retrieval_logs=await repo.count_user_rows(session, RetrievalLog, user_id=user.id),
        exports=await repo.count_user_rows(session, ExportJob, user_id=user.id),
    )


def format_archive_audit(audit: ArchiveAudit) -> str:
    lines = [
        "Стан архіву",
        "",
        "Дані:",
        f"- днів: {audit.days}",
        f"- записів: {audit.entries}",
        f"- snapshots: {audit.snapshots}",
        f"- prompts: {audit.prompts}",
        f"- media: {audit.media}",
        f"- media без локального файлу: {audit.missing_media_files}",
        f"- missed prompts: {audit.missed_prompts}",
        "",
        "AI покриття:",
        f"- analyses: {audit.ai_analyses}",
        f"- entries без extract_entry_features: {audit.feature_analyses_missing}",
        f"- embeddings: {audit.embeddings}",
        f"- entries без embedding: {_embedding_missing_text(audit.embeddings_missing)}",
        "",
        "Підсумки:",
        _format_counts(audit.summaries, empty="немає підсумків"),
        "",
        "Стани snapshot:",
        _format_counts(audit.snapshot_statuses, empty="немає snapshots"),
        "",
        "Стани missed prompts:",
        _format_counts(audit.missed_prompt_statuses, empty="немає missed prompts"),
        "",
        "Службове:",
        f"- model runs: {audit.model_runs}",
        f"- retrieval logs: {audit.retrieval_logs}",
        f"- exports: {audit.exports}",
    ]
    recommendations = archive_audit_recommendations(audit)
    if recommendations:
        lines.extend(["", "Що варто зробити:", *[f"- {item}" for item in recommendations]])
    return "\n".join(lines)


def archive_audit_recommendations(audit: ArchiveAudit) -> list[str]:
    recommendations: list[str] = []
    if audit.feature_analyses_missing:
        recommendations.append("запустити `mental-state-bot features-backfill <telegram-user-id>`")
    if audit.embeddings_missing:
        recommendations.append("запустити `mental-state-bot embed-backfill <telegram-user-id>`")
    if audit.missing_media_files:
        recommendations.append("перевірити media volume або зробити ZIP export, щоб побачити missing_media.json")
    if audit.entries and not audit.summaries.get("daily"):
        recommendations.append("згенерувати денний підсумок через `/summary` або дочекатися morning summary")
    return recommendations


def count_missing_media_files(media_items: list[Media]) -> int:
    missing = 0
    for item in media_items:
        if not item.file_path:
            missing += 1
            continue
        path = Path(item.file_path)
        if not path.exists() or not path.is_file():
            missing += 1
    return missing


async def _count_prompts_for_user(session: AsyncSession, *, user_id: UUID) -> int:
    snapshots = await repo.count_user_rows(session, Snapshot, user_id=user_id)
    if not snapshots:
        return 0

    result = await session.execute(
        select(func.count(SnapshotPrompt.id))
        .join(Snapshot, Snapshot.id == SnapshotPrompt.snapshot_id)
        .where(Snapshot.user_id == user_id)
    )
    return int(result.scalar_one() or 0)


def _embedding_missing_text(value: int | None) -> str:
    if value is None:
        return "не перевірялося, embeddings вимкнені"
    return str(value)


def _format_counts(counts: dict[str, int], *, empty: str) -> str:
    if not counts:
        return empty
    return "\n".join(
        f"- {key}: {value}" for key, value in sorted(counts.items(), key=lambda item: item[0])
    )
