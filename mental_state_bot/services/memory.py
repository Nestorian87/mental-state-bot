from __future__ import annotations

import hashlib
import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.ai.pricing import estimate_embedding_cost_usd
from mental_state_bot.ai.service import AIService
from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import Entry

logger = logging.getLogger(__name__)


class MemoryService:
    def __init__(self, settings: Settings, ai_service: AIService) -> None:
        self.settings = settings
        self.ai = ai_service

    async def embed_entry(
        self,
        session: AsyncSession,
        *,
        entry: Entry,
        user_id: UUID,
        replace_existing: bool = False,
    ) -> None:
        if not self.settings.embeddings_enabled:
            return
        if not self.settings.embedding_api_key:
            logger.info("Embedding skipped because EMBEDDING_API_KEY is not configured")
            return

        analyses = await repo.list_analyses_for_targets(session, target_type="entry", target_ids=[entry.id])
        features = {}
        micro_summary = ""
        for item in analyses:
            if item.task_name == "extract_entry_features":
                features = item.result or {}
            elif item.task_name == "generate_micro_summary" and isinstance(item.result, dict):
                micro_summary = item.result.get("text") or ""
        semantic_text, _ = await self.ai.generate_semantic_memory_text(
            session,
            user_id=user_id,
            context={
                "raw_text": entry.raw_text,
                "source": entry.source,
                "created_at": entry.created_at.isoformat() if entry.created_at else None,
                "features": features,
                "micro_summary": micro_summary,
            },
        )
        source_hash = hashlib.sha256(semantic_text.text.encode("utf-8")).hexdigest()
        try:
            result = await self.ai.create_embedding(semantic_text.text)
        except Exception as exc:
            logger.warning("Embedding creation failed", extra={"entry_id": str(entry.id), "error": str(exc)})
            await repo.create_model_run(
                session,
                user_id=user_id,
                provider=self.settings.embedding_provider,
                model=self.settings.embedding_model,
                task_name="embed_entry",
                status="failed",
                error_message=str(exc),
            )
            return

        embedding = result.data["embedding"] if result.data else []
        if len(embedding) != self.settings.embedding_dimensions:
            logger.warning(
                "Embedding dimensions mismatch",
                extra={"expected": self.settings.embedding_dimensions, "actual": len(embedding)},
            )
            await repo.create_model_run(
                session,
                user_id=user_id,
                provider=result.provider,
                model=result.model,
                task_name="embed_entry",
                status="failed_dimension_mismatch",
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                reasoning_tokens=result.usage.reasoning_tokens,
                total_tokens=result.usage.total_tokens,
                latency_ms=result.latency_ms,
                error_message=(
                    f"Embedding dimensions mismatch: expected "
                    f"{self.settings.embedding_dimensions}, got {len(embedding)}"
                ),
            )
            return
        model_run = await repo.create_model_run(
            session,
            user_id=user_id,
            provider=result.provider,
            model=result.model,
            task_name="embed_entry",
            status="success",
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            reasoning_tokens=result.usage.reasoning_tokens,
            total_tokens=result.usage.total_tokens,
            estimated_cost_usd=estimate_embedding_cost_usd(result.model, result.usage),
            latency_ms=result.latency_ms,
            meta={"target_type": "entry", "target_id": str(entry.id)},
        )
        if replace_existing:
            await repo.delete_embeddings_for_target_model(
                session,
                target_type="entry",
                target_id=entry.id,
                provider=result.provider,
                model=result.model,
            )
        await repo.add_embedding(
            session,
            user_id=user_id,
            target_type="entry",
            target_id=entry.id,
            provider=result.provider,
            model=result.model,
            dimensions=len(embedding),
            source_hash=source_hash,
            source_text=semantic_text.text,
            embedding=embedding,
        )
        model_run.meta = {**model_run.meta, "embedding_recorded": True}

    async def similar_entries(
        self, session: AsyncSession, *, user_id: UUID, query_text: str, limit: int = 8
    ):
        if not self.settings.embeddings_enabled or not self.settings.embedding_api_key:
            return []
        result = await self.ai.create_embedding(query_text)
        embedding = result.data["embedding"] if result.data else []
        records = await repo.find_similar_embeddings(
            session,
            user_id=user_id,
            embedding=embedding,
            limit=limit,
        )
        await repo.add_retrieval_log(
            session,
            user_id=user_id,
            task_name="similar_entries",
            query_text=query_text,
            provider=result.provider,
            model=result.model,
            retrieved=[
                {
                    "target_type": record.target_type,
                    "target_id": str(record.target_id),
                    "source_hash": record.source_hash,
                }
                for record in records
            ],
        )
        return records


async def backfill_entry_embeddings(
    *,
    settings: Settings,
    ai_service: AIService,
    sessionmaker,
    telegram_user_id: int,
    limit: int,
    force: bool = False,
) -> int:
    if not settings.embeddings_enabled:
        raise RuntimeError("Embeddings are disabled")
    if not settings.embedding_api_key:
        raise RuntimeError("EMBEDDING_API_KEY is not configured")

    memory_service = MemoryService(settings, ai_service)
    async with sessionmaker() as session, session.begin():
        user = await repo.get_user_by_telegram_id(session, telegram_user_id)
        if user is None:
            raise ValueError(f"Unknown Telegram user id: {telegram_user_id}")
        if force:
            entries = await repo.list_user_entries(session, user_id=user.id, limit=limit)
        else:
            entries = await repo.list_entries_without_embedding(
                session,
                user_id=user.id,
                embedding_model=settings.embedding_model,
                limit=limit,
            )
        entry_ids = [entry.id for entry in entries]
        user_id = user.id

    processed = 0
    for entry_id in entry_ids:
        async with sessionmaker() as session, session.begin():
            entry = await repo.get_entry(session, entry_id=entry_id)
            if entry is None:
                continue
            await memory_service.embed_entry(
                session,
                entry=entry,
                user_id=user_id,
                replace_existing=force,
            )
            processed += 1
    return processed
