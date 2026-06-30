from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import User

if TYPE_CHECKING:
    from mental_state_bot.ai.service import AIService

logger = logging.getLogger(__name__)


async def semantic_memory_context(
    session: AsyncSession,
    *,
    settings: Settings,
    ai_service: AIService,
    user: User,
    query_text: str,
    task_name: str,
    limit: int = 6,
    exclude_entry_ids: Iterable[str | UUID] = (),
) -> list[dict]:
    if not getattr(settings, "embeddings_enabled", False):
        return []
    if not getattr(settings, "embedding_api_key", ""):
        return []
    if not query_text.strip():
        return []
    excluded = {str(item) for item in exclude_entry_ids}
    try:
        result = await ai_service.create_embedding(query_text)
        embedding = result.data["embedding"] if result.data else []
        expected_dimensions = getattr(settings, "embedding_dimensions", len(embedding))
        if len(embedding) != expected_dimensions:
            logger.warning(
                "Semantic context skipped because embedding dimensions mismatch",
                extra={"expected": expected_dimensions, "actual": len(embedding)},
            )
            return []
        records = await repo.find_similar_embeddings(
            session,
            user_id=user.id,
            embedding=embedding,
            limit=limit,
        )
        filtered = [
            record
            for record in records
            if not (record.target_type == "entry" and str(record.target_id) in excluded)
        ]
        await repo.add_retrieval_log(
            session,
            user_id=user.id,
            task_name=task_name,
            query_text=query_text,
            provider=result.provider,
            model=result.model,
            retrieved=[
                {
                    "target_type": record.target_type,
                    "target_id": str(record.target_id),
                    "source_hash": record.source_hash,
                }
                for record in filtered
            ],
        )
        return [
            {
                "target_type": record.target_type,
                "target_id": str(record.target_id),
                "created_at": record.created_at.isoformat() if record.created_at else None,
                "source_text": _truncate(record.source_text, 700),
            }
            for record in filtered
        ]
    except Exception as exc:
        logger.warning(
            "Semantic context retrieval failed",
            extra={"user_id": str(user.id), "task_name": task_name, "error": str(exc)},
        )
        return []


def _truncate(text: str, limit: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
