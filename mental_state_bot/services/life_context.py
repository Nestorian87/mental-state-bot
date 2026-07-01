from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.ai.schemas import LifeContextCandidate
from mental_state_bot.ai.service import AIService
from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import User, UserSettings
from mental_state_bot.services.preferences import (
    life_context_items,
    pending_life_context_review,
    settings_json_with_life_context_items,
    settings_json_with_pending_life_context_review,
)

MAX_REVIEW_CANDIDATES = 5


async def start_life_context_review(
    session: AsyncSession,
    *,
    user: User,
    user_settings: UserSettings,
    ai_service: AIService,
) -> tuple[str, dict[str, Any] | None]:
    entries = await repo.get_recent_entries(session, user_id=user.id, limit=40)
    usable_entries = [entry for entry in entries if (entry.raw_text or "").strip()]
    if not usable_entries:
        return "Поки немає достатньо текстових записів, з яких можна обережно витягти контекст.", None

    existing_items = life_context_items(user_settings)
    extraction, model_run_id = await ai_service.extract_life_context_candidates(
        session,
        user_id=user.id,
        context={
            "existing_life_context": existing_items[-40:],
            "recent_entries": [
                {
                    "id": str(entry.id),
                    "created_at": entry.created_at.isoformat() if entry.created_at else None,
                    "local_timestamp": entry.local_timestamp.isoformat() if entry.local_timestamp else None,
                    "source": entry.source,
                    "raw_text": entry.raw_text,
                }
                for entry in usable_entries[-30:]
            ],
        },
    )
    candidates = [_candidate_payload(candidate) for candidate in extraction.candidates]
    candidates = _dedupe_candidates(candidates, existing_items)[:MAX_REVIEW_CANDIDATES]
    if not candidates:
        return "Я не бачу зараз сильних нових припущень для живого контексту. Це нормально.", None

    review = {
        "id": str(uuid4()),
        "index": 0,
        "candidates": candidates,
        "model_run_id": str(model_run_id) if model_run_id else None,
        "created_at": datetime.now(UTC).isoformat(),
    }
    await repo.update_user_settings(
        session,
        user_id=user.id,
        values={"settings_json": settings_json_with_pending_life_context_review(user_settings, review)},
    )
    return "Маю кілька припущень про живий контекст. Перевіримо по одному.", review


async def answer_life_context_candidate(
    session: AsyncSession,
    *,
    user: User,
    user_settings: UserSettings,
    answer: str,
    answer_kind: str,
) -> tuple[str, dict[str, Any] | None]:
    review = pending_life_context_review(user_settings)
    if not review:
        return "Не бачу активної перевірки живого контексту.", None

    candidate = current_life_context_candidate(review)
    if candidate is None:
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_with_pending_life_context_review(user_settings, None)},
        )
        return "Перевірку живого контексту вже завершено.", None

    normalized = " ".join(answer.split())[:800]
    should_store = answer_kind not in {"skip", "reject", "stop", "no"} and bool(normalized)
    updated_items = life_context_items(user_settings)
    if should_store:
        updated_items = _upsert_life_context_item(
            updated_items,
            candidate=candidate,
            answer=normalized,
            answer_kind=answer_kind,
        )

    if answer_kind == "stop":
        next_review = None
        text = "Ок, зупинив перевірку живого контексту."
    else:
        next_review = dict(review)
        next_review["index"] = int(next_review.get("index") or 0) + 1
        if current_life_context_candidate(next_review) is None:
            next_review = None
            text = "Ок, оновив живий контекст." if should_store else "Ок, завершив перевірку живого контексту."
        else:
            text = "Запам’ятав це в живому контексті." if should_store else "Ок, пропускаю це припущення."

    temp_settings = _settings_with_json(user_settings, settings_json_with_life_context_items(user_settings, updated_items))
    next_settings_json = settings_json_with_pending_life_context_review(temp_settings, next_review)
    await repo.update_user_settings(session, user_id=user.id, values={"settings_json": next_settings_json})
    return text, next_review


def current_life_context_candidate(review: dict[str, Any] | None) -> dict[str, Any] | None:
    if not review:
        return None
    candidates = review.get("candidates")
    if not isinstance(candidates, list):
        return None
    index = int(review.get("index") or 0)
    if index < 0 or index >= len(candidates):
        return None
    candidate = candidates[index]
    return candidate if isinstance(candidate, dict) else None


def format_life_context_question(review: dict[str, Any]) -> str:
    candidate = current_life_context_candidate(review)
    if not candidate:
        return "Перевірку живого контексту завершено."
    index = int(review.get("index") or 0) + 1
    total = len(review.get("candidates") or [])
    prefix = f"{index}/{total}. "
    why = candidate.get("why_it_matters")
    parts = [
        f"{prefix}{candidate.get('question') or candidate.get('hypothesis') or 'Як це краще запам’ятати?'}",
    ]
    if isinstance(why, str) and why.strip():
        parts.append(f"Навіщо: {why.strip()[:240]}")
    return "\n\n".join(parts)


def format_life_context_items(items: list[dict[str, Any]]) -> str:
    if not items:
        return "Живий контекст поки порожній. Його можна поступово зібрати з записів і твоїх підтверджень."
    lines = ["Живий контекст"]
    for item in sorted(items, key=lambda item: str(item.get("updated_at") or ""), reverse=True)[:40]:
        category = _category_label(str(item.get("category") or "other"))
        label = item.get("label") or "без назви"
        answer = item.get("answer") or item.get("hypothesis") or ""
        lines.append(f"- {category}: {label} — {answer}")
    return "\n".join(lines)


def _candidate_payload(candidate: LifeContextCandidate) -> dict[str, Any]:
    options = [" ".join(option.split())[:80] for option in candidate.options if option.strip()]
    return {
        "id": str(uuid4()),
        "category": candidate.category,
        "label": " ".join(candidate.label.split())[:80],
        "hypothesis": " ".join(candidate.hypothesis.split())[:400],
        "question": " ".join(candidate.question.split())[:400],
        "question_type": candidate.question_type,
        "options": options[:5],
        "why_it_matters": " ".join(candidate.why_it_matters.split())[:240] if candidate.why_it_matters else None,
        "sensitivity": candidate.sensitivity,
        "confidence": candidate.confidence,
    }


def _dedupe_candidates(candidates: list[dict[str, Any]], existing_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_keys = {
        (str(item.get("category") or "").lower(), str(item.get("label") or "").strip().lower())
        for item in existing_items
    }
    seen: set[tuple[str, str]] = set()
    result = []
    for candidate in candidates:
        key = (
            str(candidate.get("category") or "").lower(),
            str(candidate.get("label") or "").strip().lower(),
        )
        if not key[1] or key in seen:
            continue
        seen.add(key)
        if key in existing_keys and float(candidate.get("confidence") or 0) < 0.8:
            continue
        result.append(candidate)
    return result


def _upsert_life_context_item(
    items: list[dict[str, Any]],
    *,
    candidate: dict[str, Any],
    answer: str,
    answer_kind: str,
) -> list[dict[str, Any]]:
    now = datetime.now(UTC).isoformat()
    key = (
        str(candidate.get("category") or "other").lower(),
        str(candidate.get("label") or "").strip().lower(),
    )
    updated = []
    replaced = False
    for item in items:
        item_key = (
            str(item.get("category") or "other").lower(),
            str(item.get("label") or "").strip().lower(),
        )
        if item_key == key:
            updated.append(
                {
                    **item,
                    "hypothesis": candidate.get("hypothesis"),
                    "answer": answer,
                    "answer_kind": answer_kind,
                    "status": "confirmed",
                    "updated_at": now,
                }
            )
            replaced = True
        else:
            updated.append(item)
    if not replaced:
        updated.append(
            {
                "id": str(uuid4()),
                "category": candidate.get("category") or "other",
                "label": candidate.get("label") or "без назви",
                "hypothesis": candidate.get("hypothesis"),
                "answer": answer,
                "answer_kind": answer_kind,
                "status": "confirmed",
                "sensitivity": candidate.get("sensitivity") or "normal",
                "created_at": now,
                "updated_at": now,
            }
        )
    return updated[-80:]


def _settings_with_json(settings: UserSettings, settings_json: dict[str, Any]):
    class SettingsView:
        pass

    view = SettingsView()
    view.settings_json = settings_json
    return view


def _category_label(category: str) -> str:
    return {
        "person": "людина",
        "place": "місце",
        "project": "проєкт",
        "routine": "рутина",
        "theme": "тема",
        "term": "назва/термін",
        "other": "контекст",
    }.get(category, "контекст")
