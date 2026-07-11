from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.ai.schemas import LifeContextCandidate, LifeContextRewriteItem
from mental_state_bot.ai.service import AIService
from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import User, UserSettings
from mental_state_bot.services.memory_graph import sync_confirmed_life_context_item
from mental_state_bot.services.preferences import (
    LIFE_CONTEXT_LAST_AUTO_OFFER_AT_KEY,
    life_context_items,
    pending_input,
    pending_life_context_review,
    pending_life_context_rewrite,
    settings_json_with_life_context_items,
    settings_json_with_pending_life_context_review,
    settings_json_with_pending_life_context_rewrite,
)

logger = logging.getLogger(__name__)

MAX_REVIEW_CANDIDATES = 5
AUTO_REVIEW_COOLDOWN = timedelta(hours=10)
AUTO_REVIEW_MIN_TEXT_ENTRIES = 4
LIFE_CONTEXT_DECAY_GRACE_PERIOD = timedelta(days=7)
LIFE_CONTEXT_REVIEW_HISTORY_KEY = "life_context_review_history"
LIFE_CONTEXT_QUESTION_COOLDOWN = timedelta(days=14)
LIFE_CONTEXT_REVIEW_HISTORY_LIMIT = 80


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

    user_settings = await prune_life_context_items_if_needed(
        session,
        user=user,
        user_settings=user_settings,
        ai_service=ai_service,
        recent_entries=usable_entries[-30:],
    )
    existing_items = life_context_items(user_settings)
    stale_candidates = _stale_life_context_candidates(existing_items)
    review_history = _life_context_review_history(user_settings)
    extraction, model_run_id = await ai_service.extract_life_context_candidates(
        session,
        user_id=user.id,
        context={
            "existing_life_context": existing_items[-40:],
            "recent_life_context_questions": review_history[-25:],
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
    candidates = _filter_recently_asked_candidates(
        [*stale_candidates, *_dedupe_candidates(candidates, existing_items)],
        review_history,
        now=datetime.now(UTC),
    )[:MAX_REVIEW_CANDIDATES]
    if not candidates:
        return "Я не бачу зараз сильних нових припущень для живого контексту. Це нормально.", None

    settings_with_history = _settings_with_json(
        user_settings,
        _settings_json_with_review_history(user_settings, candidates, now=datetime.now(UTC)),
    )
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
        values={"settings_json": settings_json_with_pending_life_context_review(settings_with_history, review)},
    )
    return "Маю кілька припущень про живий контекст. Перевіримо по одному.", review


async def prune_life_context_items_if_needed(
    session: AsyncSession,
    *,
    user: User,
    user_settings: UserSettings,
    ai_service: AIService,
    recent_entries: list[Any] | None = None,
    now: datetime | None = None,
) -> UserSettings:
    items = life_context_items(user_settings)
    if not items:
        return user_settings

    structurally_valid = [
        item
        for item in items
        if str(item.get("id") or "").strip()
        and str(item.get("label") or "").strip()
        and str(item.get("answer") or item.get("hypothesis") or "").strip()
    ]
    if len(structurally_valid) != len(items):
        user_settings = await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_with_life_context_items(user_settings, structurally_valid)},
        )
        items = structurally_valid
    if not items:
        return user_settings

    prune_result, _ = await ai_service.prune_life_context_items(
        session,
        user_id=user.id,
        context={
            "current_time": (now or datetime.now(UTC)).isoformat(),
            "decay_policy": {
                "first_drop": "mark_stale",
                "delete_after": "7 days if still returned in drop",
                "revive_if_kept": True,
            },
            "life_context_items": items,
            "recent_entries": [
                {
                    "id": str(entry.id),
                    "local_timestamp": entry.local_timestamp.isoformat() if entry.local_timestamp else None,
                    "raw_text": entry.raw_text,
                }
                for entry in (recent_entries or [])[-20:]
            ],
        },
    )
    drop_ids = {str(item_id) for item_id in prune_result.drop_item_ids}
    decayed_items, changed = _apply_life_context_decay(
        items,
        drop_ids=drop_ids,
        now=now or datetime.now(UTC),
    )
    if not changed:
        return user_settings

    return await repo.update_user_settings(
        session,
        user_id=user.id,
        values={"settings_json": settings_json_with_life_context_items(user_settings, decayed_items)},
    )


async def maybe_start_auto_life_context_review(
    session: AsyncSession,
    *,
    user: User,
    user_settings: UserSettings,
    ai_service: AIService,
    now: datetime | None = None,
) -> tuple[str, dict[str, Any]] | None:
    current_time = now or datetime.now(UTC)
    if pending_life_context_review(user_settings) or pending_input(user_settings):
        return None
    if _last_auto_offer_too_recent(user_settings, current_time):
        return None

    entries = await repo.get_recent_entries(session, user_id=user.id, limit=20)
    text_entries = [entry for entry in entries if (entry.raw_text or "").strip()]
    if len(text_entries) < AUTO_REVIEW_MIN_TEXT_ENTRIES:
        return None

    user_settings = await repo.update_user_settings(
        session,
        user_id=user.id,
        values={"settings_json": _settings_json_with_last_auto_offer(user_settings, current_time)},
    )
    lead_text, review = await start_life_context_review(
        session,
        user=user,
        user_settings=user_settings,
        ai_service=ai_service,
    )
    if not review:
        return None
    return lead_text, review


async def start_life_context_rewrite(
    session: AsyncSession,
    *,
    user: User,
    user_settings: UserSettings,
    ai_service: AIService,
) -> tuple[str, dict[str, Any] | None]:
    items = life_context_items(user_settings)
    if not items:
        return "Живий контекст поки порожній, переписувати нічого.", None
    rewrite_result, model_run_id = await ai_service.rewrite_life_context_items(
        session,
        user_id=user.id,
        context={"life_context_items": items},
    )
    proposed_items, changes = _proposed_rewritten_life_context_items(items, rewrite_result.items)
    if not changes:
        return "Не бачу, що тут варто переписувати. Живий контекст виглядає достатньо охайно.", None
    rewrite = {
        "id": str(uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "model_run_id": str(model_run_id) if model_run_id else None,
        "items": proposed_items,
        "changes": changes,
    }
    await repo.update_user_settings(
        session,
        user_id=user.id,
        values={"settings_json": settings_json_with_pending_life_context_rewrite(user_settings, rewrite)},
    )
    return format_life_context_rewrite_preview(rewrite), rewrite


async def apply_life_context_rewrite(
    session: AsyncSession,
    *,
    user: User,
    user_settings: UserSettings,
) -> str:
    rewrite = pending_life_context_rewrite(user_settings)
    if not rewrite:
        return "Не бачу підготовленої ревізії живого контексту."
    items = rewrite.get("items")
    if not isinstance(items, list):
        return "Підготовлена ревізія виглядає пошкодженою, не застосовую."
    next_settings = _settings_with_json(user_settings, settings_json_with_life_context_items(user_settings, items))
    await repo.update_user_settings(
        session,
        user_id=user.id,
        values={"settings_json": settings_json_with_pending_life_context_rewrite(next_settings, None)},
    )
    return "Ок, оновив живий контекст."


async def cancel_life_context_rewrite(
    session: AsyncSession,
    *,
    user: User,
    user_settings: UserSettings,
) -> str:
    await repo.update_user_settings(
        session,
        user_id=user.id,
        values={"settings_json": settings_json_with_pending_life_context_rewrite(user_settings, None)},
    )
    return "Ок, не змінюю живий контекст."


async def answer_life_context_candidate(
    session: AsyncSession,
    *,
    user: User,
    user_settings: UserSettings,
    answer: str,
    answer_kind: str,
    ai_service: AIService | None = None,
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

    if answer_kind == "free" and ai_service is not None:
        return await _review_free_life_context_answer(
            session,
            user=user,
            user_settings=user_settings,
            review=review,
            candidate=candidate,
            answer=answer,
            ai_service=ai_service,
        )

    if candidate.get("context_action") == "relevance_check":
        return await _answer_life_context_relevance_check(
            session,
            user=user,
            user_settings=user_settings,
            review=review,
            candidate=candidate,
            answer=answer,
            answer_kind=answer_kind,
        )

    normalized = _confirmed_life_context_answer(candidate, answer, answer_kind)
    should_store = answer_kind not in {"skip", "reject", "stop", "no"} and bool(normalized)
    store_candidate = _candidate_with_confirmed_revision(candidate) if should_store else candidate
    updated_items = life_context_items(user_settings)
    if should_store:
        updated_items = _upsert_life_context_item(
            updated_items,
            candidate=store_candidate,
            answer=normalized,
            answer_kind=answer_kind,
        )
        stored_item = _matching_life_context_item(updated_items, store_candidate)
        if stored_item is not None:
            await _safe_sync_confirmed_life_context_item(
                session,
                user_id=user.id,
                item=stored_item,
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


async def _review_free_life_context_answer(
    session: AsyncSession,
    *,
    user: User,
    user_settings: UserSettings,
    review: dict[str, Any],
    candidate: dict[str, Any],
    answer: str,
    ai_service: AIService,
) -> tuple[str, dict[str, Any] | None]:
    normalized_input = " ".join(answer.split())[:1200]
    if not normalized_input:
        return "Не бачу тексту для перевірки.", review
    answer_review, _ = await ai_service.review_life_context_answer(
        session,
        user_id=user.id,
        context={
            "candidate": candidate,
            "user_answer": normalized_input,
            "existing_life_context": life_context_items(user_settings)[-40:],
        },
    )
    decision = answer_review.decision
    if decision == "store" and answer_review.normalized_answer:
        next_review = _review_with_current_candidate(
            review,
            _candidate_waiting_for_confirmation(
                candidate,
                normalized_answer=answer_review.normalized_answer,
                revised_category=answer_review.revised_category,
                revised_label=answer_review.revised_label,
                revised_hypothesis=answer_review.revised_hypothesis,
            ),
        )
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_with_pending_life_context_review(user_settings, next_review)},
        )
        return "Перевір, чи я правильно нормалізував це для живого контексту.", next_review
    if decision == "ask_followup" and answer_review.followup_question:
        next_review = _review_with_current_candidate(
            review,
            {
                **candidate,
                "question": " ".join(answer_review.followup_question.split())[:400],
                "question_type": "open",
                "options": [],
            },
        )
        await repo.update_user_settings(
            session,
            user_id=user.id,
            values={"settings_json": settings_json_with_pending_life_context_review(user_settings, next_review)},
        )
        return "Не хочу записувати це як факт без уточнення.", next_review
    next_review = dict(review)
    next_review["index"] = int(next_review.get("index") or 0) + 1
    if current_life_context_candidate(next_review) is None:
        next_review = None
    await repo.update_user_settings(
        session,
        user_id=user.id,
        values={"settings_json": settings_json_with_pending_life_context_review(user_settings, next_review)},
    )
    return "Не записую це в живий контекст.", next_review


async def _answer_life_context_relevance_check(
    session: AsyncSession,
    *,
    user: User,
    user_settings: UserSettings,
    review: dict[str, Any],
    candidate: dict[str, Any],
    answer: str,
    answer_kind: str,
) -> tuple[str, dict[str, Any] | None]:
    item_id = str(candidate.get("existing_item_id") or "")
    items = life_context_items(user_settings)
    if answer_kind == "free":
        return "Напиши коротко, як це краще переформулювати.", review
    normalized_answer = " ".join(answer.split()).lower()
    if answer_kind in {"no", "skip", "reject"} or "стар" in normalized_answer or "неакту" in normalized_answer:
        updated_items = [item for item in items if str(item.get("id") or "") != item_id]
        old_item = next((item for item in items if str(item.get("id") or "") == item_id), None)
        if old_item is not None:
            await _safe_sync_confirmed_life_context_item(
                session,
                user_id=user.id,
                item=old_item,
                active=False,
            )
        text = "Ок, прибрав це з живого контексту."
    else:
        updated_items = [_revive_life_context_item(item) if str(item.get("id") or "") == item_id else item for item in items]
        revived_item = next((item for item in updated_items if str(item.get("id") or "") == item_id), None)
        if revived_item is not None:
            await _safe_sync_confirmed_life_context_item(
                session,
                user_id=user.id,
                item=revived_item,
            )
        text = "Ок, залишив це як актуальний контекст."

    next_review = _next_review_after_current(review)
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


def _next_review_after_current(review: dict[str, Any]) -> dict[str, Any] | None:
    next_review = dict(review)
    next_review["index"] = int(next_review.get("index") or 0) + 1
    if current_life_context_candidate(next_review) is None:
        return None
    return next_review


def _review_with_current_candidate(review: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    next_review = dict(review)
    candidates = list(next_review.get("candidates") or [])
    index = int(next_review.get("index") or 0)
    if 0 <= index < len(candidates):
        candidates[index] = candidate
    next_review["candidates"] = candidates
    return next_review


def _candidate_waiting_for_confirmation(
    candidate: dict[str, Any],
    *,
    normalized_answer: str,
    revised_category: str | None,
    revised_label: str | None,
    revised_hypothesis: str | None,
) -> dict[str, Any]:
    normalized = " ".join(normalized_answer.split())[:800]
    return {
        **candidate,
        "category": revised_category or candidate.get("category") or "other",
        "label": " ".join((revised_label or candidate.get("label") or "без назви").split())[:80],
        "hypothesis": " ".join((revised_hypothesis or candidate.get("hypothesis") or "").split())[:400],
        "question": f"Я б записав так: «{normalized}». Зберегти саме так?",
        "question_type": "confirm",
        "options": [],
        "pending_normalized_answer": normalized,
    }


def _candidate_with_confirmed_revision(candidate: dict[str, Any]) -> dict[str, Any]:
    if not candidate.get("pending_normalized_answer"):
        return candidate
    return {key: value for key, value in candidate.items() if key != "pending_normalized_answer"}


def _confirmed_life_context_answer(candidate: dict[str, Any], answer: str, answer_kind: str) -> str:
    pending_answer = candidate.get("pending_normalized_answer")
    if isinstance(pending_answer, str) and pending_answer.strip():
        return " ".join(pending_answer.split())[:800]
    hypothesis = candidate.get("hypothesis")
    if answer_kind == "yes" and isinstance(hypothesis, str) and hypothesis.strip():
        return " ".join(hypothesis.split())[:800]
    return " ".join(answer.split())[:800]


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


def _stale_life_context_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for item in items:
        if item.get("decay_status") != "stale":
            continue
        label = _compact_or_none(item.get("label")) or "цей контекст"
        answer = _compact_or_none(item.get("answer") or item.get("hypothesis")) or label
        candidates.append(
            {
                "id": str(uuid4()),
                "category": item.get("category") or "other",
                "label": label,
                "hypothesis": answer,
                "question": f"«{label}» ще допомагає розуміти твої майбутні записи, чи це вже радше старий контекст?",
                "question_type": "status",
                "options": ["Ще актуально", "Вже старий контекст"],
                "why_it_matters": "Щоб жива пам’ять поступово не тримала застарілі речі.",
                "sensitivity": item.get("sensitivity") or "normal",
                "confidence": 0.7,
                "context_action": "relevance_check",
                "existing_item_id": item.get("id"),
            }
        )
    return candidates[:2]


def format_life_context_rewrite_preview(rewrite: dict[str, Any]) -> str:
    changes = rewrite.get("changes") if isinstance(rewrite, dict) else None
    if not isinstance(changes, list) or not changes:
        return "Не бачу змін для живого контексту."
    lines = ["Пропоную так оновити живий контекст:"]
    for index, change in enumerate(changes[:12], start=1):
        if not isinstance(change, dict):
            continue
        action = str(change.get("action") or "")
        label = str(change.get("label") or "без назви")
        if action == "drop":
            lines.append(f"{index}. Прибрати: {label}")
        elif action == "rewrite":
            lines.append(f"{index}. Переписати: {label} — {change.get('answer') or ''}")
    if len(changes) > 12:
        lines.append(f"...і ще {len(changes) - 12}.")
    lines.append("\nЗастосувати ці зміни?")
    return "\n".join(lines)


def _proposed_rewritten_life_context_items(
    current_items: list[dict[str, Any]],
    rewrite_items: list[LifeContextRewriteItem],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(item.id): item for item in rewrite_items if item.id}
    proposed_items = []
    changes = []
    now = datetime.now(UTC).isoformat()
    for item in current_items:
        item_id = str(item.get("id") or "")
        rewrite = by_id.get(item_id)
        if rewrite is None or rewrite.action == "keep":
            proposed_items.append(item)
            continue
        label = _compact_or_none(rewrite.label) or _compact_or_none(item.get("label")) or "без назви"
        if rewrite.action == "drop":
            changes.append({"action": "drop", "id": item_id, "label": label, "reason": rewrite.reason})
            continue
        answer = _compact_or_none(rewrite.answer)
        if not answer:
            proposed_items.append(item)
            continue
        category = rewrite.category or item.get("category") or "other"
        next_item = {
            **item,
            "category": category,
            "label": label[:80],
            "answer": answer[:800],
            "status": item.get("status") or "confirmed",
            "updated_at": now,
        }
        proposed_items.append(next_item)
        if _life_context_item_changed(item, next_item):
            changes.append(
                {
                    "action": "rewrite",
                    "id": item_id,
                    "label": next_item["label"],
                    "answer": next_item["answer"],
                    "reason": rewrite.reason,
                }
            )
    return proposed_items[:80], changes


def _apply_life_context_decay(
    items: list[dict[str, Any]],
    *,
    drop_ids: set[str],
    now: datetime,
) -> tuple[list[dict[str, Any]], bool]:
    next_items = []
    changed = False
    now_text = now.astimezone(UTC).isoformat()
    for item in items:
        item_id = str(item.get("id") or "")
        if item_id in drop_ids:
            marked_at = _datetime_or_none(item.get("decay_marked_at"))
            if marked_at is not None and now - marked_at >= LIFE_CONTEXT_DECAY_GRACE_PERIOD:
                changed = True
                continue
            next_item = {
                **item,
                "decay_status": "stale",
                "decay_marked_at": marked_at.isoformat() if marked_at else now_text,
                "decay_checked_at": now_text,
            }
            changed = changed or next_item != item
            next_items.append(next_item)
            continue
        revived_item = {
            key: value
            for key, value in item.items()
            if key not in {"decay_status", "decay_marked_at", "decay_checked_at"}
        }
        changed = changed or revived_item != item
        next_items.append(revived_item)
    return next_items, changed


def _revive_life_context_item(item: dict[str, Any]) -> dict[str, Any]:
    revived = {
        key: value
        for key, value in item.items()
        if key not in {"decay_status", "decay_marked_at", "decay_checked_at"}
    }
    revived["updated_at"] = datetime.now(UTC).isoformat()
    return revived


def _datetime_or_none(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _life_context_item_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    keys = ("category", "label", "answer")
    return any(str(before.get(key) or "") != str(after.get(key) or "") for key in keys)


def _compact_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    compact = " ".join(value.split())
    return compact or None


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


def _filter_recently_asked_candidates(
    candidates: list[dict[str, Any]],
    history: list[dict[str, Any]],
    *,
    now: datetime,
) -> list[dict[str, Any]]:
    recent_keys = set()
    for item in history:
        asked_at = _datetime_or_none(item.get("asked_at"))
        if asked_at is None or now - asked_at > LIFE_CONTEXT_QUESTION_COOLDOWN:
            continue
        key = _candidate_history_key(item)
        if key[1]:
            recent_keys.add(key)

    result = []
    for candidate in candidates:
        key = _candidate_history_key(candidate)
        if key[1] and key in recent_keys:
            continue
        result.append(candidate)
    return result


def _life_context_review_history(settings: UserSettings) -> list[dict[str, Any]]:
    value = (getattr(settings, "settings_json", None) or {}).get(LIFE_CONTEXT_REVIEW_HISTORY_KEY)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _settings_json_with_review_history(
    settings: UserSettings,
    candidates: list[dict[str, Any]],
    *,
    now: datetime,
) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    history = _life_context_review_history(settings)
    asked_at = now.astimezone(UTC).isoformat()
    history.extend(
        {
            "category": candidate.get("category") or "other",
            "label": candidate.get("label") or "",
            "question": candidate.get("question") or candidate.get("hypothesis") or "",
            "context_action": candidate.get("context_action"),
            "asked_at": asked_at,
        }
        for candidate in candidates
    )
    current[LIFE_CONTEXT_REVIEW_HISTORY_KEY] = history[-LIFE_CONTEXT_REVIEW_HISTORY_LIMIT:]
    return current


def _candidate_history_key(candidate: dict[str, Any]) -> tuple[str, str]:
    category = " ".join(str(candidate.get("category") or "other").lower().split())
    label = " ".join(str(candidate.get("label") or "").lower().split())
    return category, label


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


def _matching_life_context_item(
    items: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    category = str(candidate.get("category") or "other").lower()
    label = str(candidate.get("label") or "").strip().lower()
    return next(
        (
            item
            for item in items
            if str(item.get("category") or "other").lower() == category
            and str(item.get("label") or "").strip().lower() == label
        ),
        None,
    )


async def _safe_sync_confirmed_life_context_item(
    session: AsyncSession,
    *,
    user_id,
    item: dict[str, Any],
    active: bool = True,
) -> None:
    try:
        await sync_confirmed_life_context_item(
            session,
            user_id=user_id,
            item=item,
            active=active,
        )
    except Exception:
        logger.warning(
            "Could not sync confirmed life context item to memory graph",
            extra={"user_id": str(user_id), "item_id": str(item.get("id") or "")},
            exc_info=True,
        )


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


def _last_auto_offer_too_recent(settings: UserSettings, now: datetime) -> bool:
    value = (getattr(settings, "settings_json", None) or {}).get(LIFE_CONTEXT_LAST_AUTO_OFFER_AT_KEY)
    if not isinstance(value, str):
        return False
    try:
        last = datetime.fromisoformat(value)
    except ValueError:
        return False
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    return now - last < AUTO_REVIEW_COOLDOWN


def _settings_json_with_last_auto_offer(settings: UserSettings, now: datetime) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    current[LIFE_CONTEXT_LAST_AUTO_OFFER_AT_KEY] = now.astimezone(UTC).isoformat()
    return current
