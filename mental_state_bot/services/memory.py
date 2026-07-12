from __future__ import annotations

import hashlib
import logging
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.ai.pricing import estimate_embedding_cost_usd
from mental_state_bot.ai.schemas import MemoryGraphExtraction
from mental_state_bot.ai.service import AIService
from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import Entry
from mental_state_bot.services.memory_graph import (
    apply_memory_graph_extraction,
    mark_fresh_memory_graph_duplicate_candidates,
    relevant_memory_context_for_text,
    review_memory_graph_duplicates,
)
from mental_state_bot.services.preferences import life_context_items

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
        memory_context = await build_entry_memory_context(session, entry=entry, user_id=user_id)
        semantic_text, model_run_id = await self.ai.generate_semantic_memory_text(
            session,
            user_id=user_id,
            context=memory_context,
        )
        await self._update_memory_graph(
            session,
            user_id=user_id,
            entry=entry,
            memory_context=memory_context,
            semantic_text=semantic_text.text,
            extraction=MemoryGraphExtraction.model_validate(semantic_text.graph),
            model_run_id=model_run_id,
            review_fresh_candidates=not replace_existing,
        )
        if not self.settings.embeddings_enabled:
            return
        if not self.settings.embedding_api_key:
            logger.info("Embedding skipped because EMBEDDING_API_KEY is not configured")
            return

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

    async def _update_memory_graph(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        entry: Entry,
        memory_context: dict[str, Any],
        semantic_text: str,
        extraction: MemoryGraphExtraction,
        model_run_id: UUID | None,
        review_fresh_candidates: bool,
    ) -> None:
        try:
            result = await apply_memory_graph_extraction(
                session,
                user_id=user_id,
                entry=entry,
                extraction=extraction,
            )
            logger.debug(
                "Memory graph updated",
                extra={
                    "entry_id": str(entry.id),
                    "nodes_seen": result.nodes_seen,
                    "nodes_created": result.nodes_created,
                    "edges_seen": result.edges_seen,
                    "edges_created": result.edges_created,
                },
            )
            if review_fresh_candidates and result.touched_node_ids:
                fresh_pairs = await mark_fresh_memory_graph_duplicate_candidates(
                    session,
                    user_id=user_id,
                    touched_node_ids=set(result.touched_node_ids),
                )
                review = await review_memory_graph_duplicates(
                    session,
                    user_id=user_id,
                    ai_service=self.ai,
                    pair_limit=2,
                    only_node_ids=set(result.touched_node_ids),
                )
                if fresh_pairs and review.pairs_selected:
                    logger.debug(
                        "Fresh memory graph candidates reviewed",
                        extra={
                            "entry_id": str(entry.id),
                            "pairs_selected": review.pairs_selected,
                            "decisions_received": review.decisions_received,
                        },
                    )
            if model_run_id:
                run = await repo.get_model_run(session, model_run_id=model_run_id)
                if run is not None:
                    run.meta = {
                        **(run.meta or {}),
                        "memory_graph": {
                            "nodes_seen": result.nodes_seen,
                            "nodes_created": result.nodes_created,
                            "edges_seen": result.edges_seen,
                            "edges_created": result.edges_created,
                            "combined_with_memory_capsule": True,
                        },
                    }
        except Exception as exc:
            logger.warning(
                "Memory graph update failed",
                extra={"entry_id": str(entry.id), "error": str(exc)},
            )

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


async def build_entry_memory_context(
    session: AsyncSession,
    *,
    entry: Entry,
    user_id: UUID,
) -> dict[str, Any]:
    analyses = list(await repo.list_analyses_for_targets(session, target_type="entry", target_ids=[entry.id]))
    latest = _latest_analysis_results(analyses)
    day_entries = list(await repo.list_day_entries(session, day_id=entry.day_id)) if entry.day_id else []
    prompts = list(await repo.get_snapshot_prompts(session, snapshot_id=entry.snapshot_id)) if entry.snapshot_id else []
    user_settings = await repo.get_user_settings(session, user_id)
    try:
        graph_context = await relevant_memory_context_for_text(
            session,
            user_id=user_id,
            text=entry.raw_text or "",
        )
    except Exception as exc:
        logger.warning("Memory graph lookup failed", extra={"entry_id": str(entry.id), "error": str(exc)})
        graph_context = {"nodes": [], "edges": [], "matched": []}
    return {
        "memory_kind": "contextual_entry_capsule",
        "entry": _entry_payload(entry),
        "snapshot": _snapshot_payload(prompts),
        "local_day_window": _entry_window(day_entries, target_entry_id=entry.id, before=4, after=2),
        "features": latest.get("extract_entry_features") or {},
        "micro_summary": _micro_summary_text(latest.get("generate_micro_summary")),
        "corrections": _correction_payloads(analyses),
        "life_context": _life_context_payload(user_settings),
        "relevant_memory_graph": graph_context,
        "capsule_instructions": [
            "Summarize this entry as a contextual memory capsule, not as a user-facing message.",
            "Use raw text as source of truth, but include prompt/day/correction context when needed.",
            "Separate current state from story, dream, memory, topic, or metadata.",
            "Do not turn retrieved context into facts of this exact moment unless evidence supports it.",
            "Keep names, projects, places, and unresolved references if they help future lookup.",
        ],
    }


def _latest_analysis_results(analyses) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for item in analyses:
        if isinstance(getattr(item, "result", None), dict):
            latest[item.task_name] = item.result
    return latest


def _entry_payload(entry: Entry) -> dict[str, Any]:
    return {
        "id": str(entry.id),
        "source": entry.source,
        "raw_text": entry.raw_text,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "local_timestamp": entry.local_timestamp.isoformat() if entry.local_timestamp else None,
        "snapshot_id": str(entry.snapshot_id) if entry.snapshot_id else None,
        "day_id": str(entry.day_id) if entry.day_id else None,
        "metadata": entry.meta or {},
    }


def _snapshot_payload(prompts) -> dict[str, Any] | None:
    if not prompts:
        return None
    return {
        "latest_prompt": prompts[-1].text,
        "prompts": [
            {
                "kind": item.prompt_kind,
                "text": item.text,
                "sent_at": item.sent_at.isoformat() if item.sent_at else None,
            }
            for item in prompts[-4:]
        ],
    }


def _entry_window(
    entries: list[Entry],
    *,
    target_entry_id: UUID,
    before: int,
    after: int,
) -> dict[str, Any]:
    if not entries:
        return {"entries": []}
    index = next((position for position, item in enumerate(entries) if item.id == target_entry_id), None)
    if index is None:
        return {"entries": []}
    start = max(0, index - before)
    end = min(len(entries), index + after + 1)
    return {
        "position_in_day": index + 1,
        "entries_in_day": len(entries),
        "entries": [
            {
                **_entry_payload(item),
                "relation_to_target": "target"
                if item.id == target_entry_id
                else "before"
                if position < index
                else "after",
            }
            for position, item in enumerate(entries[start:end], start=start)
        ],
    }


def _micro_summary_text(result: dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return ""
    return str(result.get("text") or "").strip()


def _correction_payloads(analyses) -> list[dict[str, Any]]:
    corrections: list[dict[str, Any]] = []
    for item in analyses:
        if item.task_name != "apply_correction" or not isinstance(getattr(item, "result", None), dict):
            continue
        result = item.result
        text = str(result.get("correction_text") or "").strip()
        if not text:
            continue
        corrections.append(
            {
                "correction_text": text,
                "corrected_at": result.get("corrected_at"),
            }
        )
    return corrections[-5:]


def _life_context_payload(user_settings) -> list[dict[str, Any]]:
    items = life_context_items(user_settings)
    payload: list[dict[str, Any]] = []
    for item in items[-30:]:
        if not isinstance(item, dict):
            continue
        payload.append(
            {
                "category": item.get("category"),
                "label": item.get("label"),
                "value": item.get("value") or item.get("answer") or item.get("hypothesis"),
                "confidence": item.get("confidence"),
            }
        )
    return payload
