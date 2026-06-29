from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.ai.schemas import EntryFeatures
from mental_state_bot.ai.service import AIService
from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import Entry

ENTRY_FEATURES_TASK = "extract_entry_features"
ENTRY_FEATURES_SCHEMA_VERSION = "entry_features.v1"


@dataclass(frozen=True)
class FeatureBackfillResult:
    selected: int
    processed: int
    skipped_missing: int = 0


async def backfill_entry_features(
    *,
    settings: Settings,
    ai_service: AIService,
    sessionmaker,
    telegram_user_id: int,
    limit: int,
) -> FeatureBackfillResult:
    async with sessionmaker() as session, session.begin():
        user = await repo.get_user_by_telegram_id(session, telegram_user_id)
        if user is None:
            raise ValueError(f"Unknown Telegram user id: {telegram_user_id}")
        entries = await repo.list_entries_without_analysis(
            session,
            user_id=user.id,
            task_name=ENTRY_FEATURES_TASK,
            limit=limit,
        )
        entry_ids = [entry.id for entry in entries]
        user_id = user.id

    processed = 0
    skipped_missing = 0
    for entry_id in entry_ids:
        async with sessionmaker() as session, session.begin():
            entry = await repo.get_entry(session, entry_id=entry_id)
            if entry is None:
                skipped_missing += 1
                continue
            await analyze_entry_features(
                session,
                settings=settings,
                ai_service=ai_service,
                user_id=user_id,
                entry=entry,
            )
            processed += 1
    return FeatureBackfillResult(
        selected=len(entry_ids),
        processed=processed,
        skipped_missing=skipped_missing,
    )


async def analyze_entry_features(
    session: AsyncSession,
    *,
    settings: Settings,
    ai_service: AIService,
    user_id: uuid.UUID,
    entry: Entry,
    extra_context: dict[str, Any] | None = None,
) -> EntryFeatures:
    features, model_run_id = await ai_service.extract_entry_features(
        session,
        user_id=user_id,
        context=entry_feature_context(entry, extra_context=extra_context),
    )
    await repo.add_ai_analysis(
        session,
        user_id=user_id,
        target_type="entry",
        target_id=entry.id,
        task_name=ENTRY_FEATURES_TASK,
        schema_version=ENTRY_FEATURES_SCHEMA_VERSION,
        provider=settings.ai_provider,
        model=settings.ai_live_model,
        result=features.model_dump(),
        confidence=Decimal(str(features.confidence)),
        uncertainty_notes=features.uncertainty_notes,
        model_run_id=model_run_id,
    )
    return features


def entry_feature_context(entry: Entry, *, extra_context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = {
        "raw_text": entry.raw_text or "",
        "source": entry.source,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "local_timestamp": entry.local_timestamp.isoformat() if entry.local_timestamp else None,
        "metadata": entry.meta or {},
        "backfill": True,
    }
    if extra_context:
        context.update(extra_context)
    return context
