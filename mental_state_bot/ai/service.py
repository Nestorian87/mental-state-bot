from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any
from uuid import UUID

import httpx
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.ai.client import OpenAICompatibleClient
from mental_state_bot.ai.pricing import estimate_cost_usd, estimate_transcription_cost_usd
from mental_state_bot.ai.prompts import (
    CLARIFICATION_PROMPT,
    DAILY_SUMMARY_PROMPT,
    EXTRACTION_PROMPT,
    LIFE_CONTEXT_PROMPT,
    MICRO_SUMMARY_PROMPT,
    MONTHLY_SUMMARY_PROMPT,
    QUESTION_PROMPT,
    SEMANTIC_MEMORY_PROMPT,
    SYSTEM_STYLE,
    WEEKLY_SUMMARY_PROMPT,
)
from mental_state_bot.ai.schemas import (
    ClarificationResult,
    DailySummary,
    EntryFeatures,
    LifeContextExtraction,
    MicroSummary,
    ModelCallResult,
    PeriodSummary,
    QuestionResult,
    Route,
    SemanticMemoryText,
)
from mental_state_bot.config import Settings
from mental_state_bot.db import repositories as repo

logger = logging.getLogger(__name__)


class AIService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chat_client = OpenAICompatibleClient(
            provider=settings.ai_provider,
            base_url=settings.ai_base_url,
            api_key=settings.ai_api_key,
            timeout_seconds=settings.ai_timeout_seconds,
            provider_extra=settings.ai_provider_extra_json,
        )
        self.embedding_client = OpenAICompatibleClient(
            provider=settings.embedding_provider,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            timeout_seconds=settings.ai_timeout_seconds,
        )
        self.transcription_client = OpenAICompatibleClient(
            provider=settings.transcription_provider,
            base_url=settings.transcription_base_url,
            api_key=settings.transcription_api_key,
            timeout_seconds=settings.ai_timeout_seconds,
        )

    async def generate_snapshot_question(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        context: dict[str, Any],
    ) -> tuple[QuestionResult, UUID | None]:
        route = Route(
            model=self.settings.ai_live_model,
            thinking=self.settings.ai_live_thinking,
            temperature=self.settings.ai_temperature,
        )
        fallback = QuestionResult(
            question="Що зараз відбувається і що було з минулого запису? Можна кількома словами.",
            intent="state_and_activity",
        )
        return await self._json_task(
            session,
            user_id=user_id,
            task_name="generate_snapshot_question",
            route=route,
            schema_model=QuestionResult,
            system=SYSTEM_STYLE,
            prompt=QUESTION_PROMPT,
            payload=context,
            fallback=fallback,
        )

    async def generate_clarification(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        context: dict[str, Any],
    ) -> tuple[ClarificationResult, UUID | None]:
        route = Route(
            model=self.settings.ai_live_model,
            thinking=self.settings.ai_live_thinking,
            temperature=self.settings.ai_temperature,
        )
        fallback = ClarificationResult(
            question="Уточню одне і закрию: що саме тут важливо зафіксувати?",
            expected_gain="free_text_specificity",
            should_clarify=True,
        )
        return await self._json_task(
            session,
            user_id=user_id,
            task_name="generate_clarification_question",
            route=route,
            schema_model=ClarificationResult,
            system=SYSTEM_STYLE,
            prompt=CLARIFICATION_PROMPT,
            payload=context,
            fallback=fallback,
        )

    async def extract_entry_features(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        context: dict[str, Any],
    ) -> tuple[EntryFeatures, UUID | None]:
        route = Route(
            model=self.settings.ai_live_model,
            thinking=self.settings.ai_live_thinking,
            temperature=0.1,
        )
        fallback = _fallback_features(context.get("raw_text") or "")
        return await self._json_task(
            session,
            user_id=user_id,
            task_name="extract_entry_features",
            route=route,
            schema_model=EntryFeatures,
            system=SYSTEM_STYLE,
            prompt=EXTRACTION_PROMPT,
            payload=context,
            fallback=fallback,
        )

    async def generate_micro_summary(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        context: dict[str, Any],
    ) -> tuple[MicroSummary, UUID | None]:
        route = Route(
            model=self.settings.ai_live_model,
            thinking=self.settings.ai_live_thinking,
            temperature=self.settings.ai_temperature,
        )
        raw_text = context.get("raw_text") or "короткий запис"
        fallback = MicroSummary(text=f"Я почув це як короткий зріз: {raw_text}. Зберіг, на цьому все.")
        return await self._json_task(
            session,
            user_id=user_id,
            task_name="generate_micro_summary",
            route=route,
            schema_model=MicroSummary,
            system=SYSTEM_STYLE,
            prompt=MICRO_SUMMARY_PROMPT,
            payload=context,
            fallback=fallback,
        )

    async def generate_semantic_memory_text(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        context: dict[str, Any],
    ) -> tuple[SemanticMemoryText, UUID | None]:
        route = Route(
            model=self.settings.ai_live_model,
            thinking=False,
            temperature=0.1,
        )
        fallback = SemanticMemoryText(text=_compact_semantic_text(context))
        return await self._json_task(
            session,
            user_id=user_id,
            task_name="generate_semantic_memory_text",
            route=route,
            schema_model=SemanticMemoryText,
            system=SYSTEM_STYLE,
            prompt=SEMANTIC_MEMORY_PROMPT,
            payload=context,
            fallback=fallback,
        )

    async def extract_life_context_candidates(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        context: dict[str, Any],
    ) -> tuple[LifeContextExtraction, UUID | None]:
        route = Route(
            model=self.settings.ai_live_model,
            thinking=False,
            temperature=0.15,
        )
        fallback = LifeContextExtraction(candidates=[])
        return await self._json_task(
            session,
            user_id=user_id,
            task_name="extract_life_context_candidates",
            route=route,
            schema_model=LifeContextExtraction,
            system=SYSTEM_STYLE,
            prompt=LIFE_CONTEXT_PROMPT,
            payload=context,
            fallback=fallback,
        )

    async def generate_daily_summary(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        context: dict[str, Any],
    ) -> tuple[DailySummary, UUID | None]:
        route = Route(
            model=self.settings.ai_heavy_model,
            thinking=self.settings.ai_heavy_thinking,
            temperature=0.25,
        )
        fallback = DailySummary(
            short_text="Даних за день поки мало. Я зберіг наявні записи й позначив прогалини.",
            story="Даних недостатньо для повної історії дня.",
            data_gaps=["даних мало або підсумок згенерований без AI"],
            data_quality="low",
        )
        return await self._json_task(
            session,
            user_id=user_id,
            task_name="generate_daily_summary",
            route=route,
            schema_model=DailySummary,
            system=SYSTEM_STYLE,
            prompt=DAILY_SUMMARY_PROMPT,
            payload=context,
            fallback=fallback,
        )

    async def generate_period_summary(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        period_type: str,
        context: dict[str, Any],
    ) -> tuple[PeriodSummary, UUID | None]:
        route = Route(
            model=self.settings.ai_heavy_model,
            thinking=self.settings.ai_heavy_thinking,
            temperature=0.25,
        )
        fallback = PeriodSummary(
            short_text=f"Даних за {period_type} поки мало. Я зберіг наявні записи й позначив прогалини.",
            period_story=f"Даних недостатньо для повного підсумку за {period_type}.",
            data_gaps=["даних мало або підсумок згенерований без AI"],
            data_quality="low",
        )
        prompt = MONTHLY_SUMMARY_PROMPT if period_type == "monthly" else WEEKLY_SUMMARY_PROMPT
        return await self._json_task(
            session,
            user_id=user_id,
            task_name=f"generate_{period_type}_summary",
            route=route,
            schema_model=PeriodSummary,
            system=SYSTEM_STYLE,
            prompt=prompt,
            payload=context,
            fallback=fallback,
        )

    async def create_embedding(self, text: str) -> ModelCallResult:
        return await self.embedding_client.embed(model=self.settings.embedding_model, text=text)

    async def transcribe_voice(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        file_path: str,
        duration_seconds: int | None,
    ) -> tuple[str, UUID | None]:
        task_name = "transcribe_voice"
        if not self.settings.transcription_enabled:
            run = await repo.create_model_run(
                session,
                user_id=user_id,
                provider=self.settings.transcription_provider,
                model=self.settings.transcription_model,
                task_name=task_name,
                status="skipped_disabled",
                meta={"duration_seconds": duration_seconds},
            )
            return "", run.id
        if not self.settings.transcription_api_key:
            run = await repo.create_model_run(
                session,
                user_id=user_id,
                provider=self.settings.transcription_provider,
                model=self.settings.transcription_model,
                task_name=task_name,
                status="skipped_no_api_key",
                meta={"duration_seconds": duration_seconds},
            )
            return "", run.id

        result = await self.transcription_client.transcribe(
            model=self.settings.transcription_model,
            file_path=file_path,
            language=self.settings.transcription_language,
            prompt=(
                "Це коротке Telegram-голосове повідомлення для особистого щоденника. "
                "Транскрибуй дослівно, зберігай розмовний стиль, назви треків, імена та змішану українську/англійську лексику."
            ),
        )
        request_hash = self.transcription_client.request_hash(
            {
                "task": task_name,
                "model": self.settings.transcription_model,
                "file_path": file_path,
                "duration_seconds": duration_seconds,
            }
        )
        run = await repo.create_model_run(
            session,
            user_id=user_id,
            provider=result.provider,
            model=result.model,
            task_name=task_name,
            status="success",
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            reasoning_tokens=result.usage.reasoning_tokens,
            total_tokens=result.usage.total_tokens,
            estimated_cost_usd=estimate_transcription_cost_usd(result.model, duration_seconds),
            latency_ms=result.latency_ms,
            request_hash=request_hash,
            meta={"duration_seconds": duration_seconds, "raw_usage": result.usage.model_dump()},
        )
        return result.text.strip(), run.id

    async def _json_task[T: BaseModel](
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        task_name: str,
        route: Route,
        schema_model: type[T],
        system: str,
        prompt: str,
        payload: dict[str, Any],
        fallback: T,
    ) -> tuple[T, UUID | None]:
        if not self.settings.ai_api_key:
            run = await repo.create_model_run(
                session,
                user_id=user_id,
                provider=self.settings.ai_provider,
                model=route.model,
                task_name=task_name,
                status="skipped_no_api_key",
                meta={"fallback": True},
            )
            return fallback, run.id

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"{prompt}\n\nКонтекст JSON:\n{json.dumps(payload, ensure_ascii=False)}",
            },
        ]
        request_hash = self.chat_client.request_hash(
            {"task": task_name, "model": route.model, "messages": messages}
        )
        try:
            result = await self.chat_client.chat(
                task_name=task_name,
                model=route.model,
                messages=messages,
                temperature=route.temperature,
                json_schema=schema_model.model_json_schema(),
                thinking=route.thinking,
            )
            parsed = schema_model.model_validate(result.data)
            run = await self._store_success_run(session, user_id=user_id, result=result, request_hash=request_hash)
            return parsed, run.id
        except (
            RuntimeError,
            ValidationError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
            httpx.HTTPError,
        ) as exc:
            logger.warning("AI task failed, using fallback", extra={"task": task_name, "error": str(exc)})
            run = await repo.create_model_run(
                session,
                user_id=user_id,
                provider=self.settings.ai_provider,
                model=route.model,
                task_name=task_name,
                status="fallback_after_error",
                error_message=str(exc),
                request_hash=request_hash,
                meta={"fallback": True},
            )
            return fallback, run.id

    async def _store_success_run(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        result: ModelCallResult,
        request_hash: str,
    ):
        return await repo.create_model_run(
            session,
            user_id=user_id,
            provider=result.provider,
            model=result.model,
            task_name=result.task_name,
            status="success",
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            reasoning_tokens=result.usage.reasoning_tokens,
            total_tokens=result.usage.total_tokens,
            estimated_cost_usd=estimate_cost_usd(result.provider, result.model, result.usage),
            latency_ms=result.latency_ms,
            request_hash=request_hash,
            meta={"raw_usage": result.usage.model_dump()},
        )


def _fallback_features(raw_text: str) -> EntryFeatures:
    confidence = Decimal("0.35")
    return EntryFeatures(
        activity_labels=[],
        state_labels=[],
        data_quality="very_low" if len(raw_text.strip()) < 10 else "partial",
        uncertainty_notes=["AI feature extraction unavailable; no keyword-based interpretation was applied"],
        confidence=float(confidence),
    )


def _compact_semantic_text(context: dict[str, Any]) -> str:
    raw = context.get("raw_text") or ""
    features = context.get("features") or {}
    summary = context.get("micro_summary") or ""
    return f"Raw: {raw}\nFeatures: {json.dumps(features, ensure_ascii=False)}\nMicro-summary: {summary}"
