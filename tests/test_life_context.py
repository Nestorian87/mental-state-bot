from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from mental_state_bot.ai.schemas import (
    LifeContextCandidate,
    LifeContextExtraction,
    LifeContextPruneResult,
)
from mental_state_bot.services import life_context as life_context_module
from mental_state_bot.services.life_context import (
    answer_life_context_candidate,
    current_life_context_candidate,
    format_life_context_question,
    maybe_start_auto_life_context_review,
)


@pytest.mark.asyncio
async def test_answer_life_context_candidate_stores_confirmed_answer(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    candidate = {
        "id": "candidate-1",
        "category": "project",
        "label": "альбом",
        "hypothesis": "Альбом зараз активний творчий проєкт.",
        "question": "Альбом зараз активний проєкт?",
        "question_type": "status",
        "options": ["Активний", "Пауза"],
    }
    settings = SimpleNamespace(settings_json={"pending_life_context_review": {"index": 0, "candidates": [candidate]}})
    updates = []

    async def update_user_settings(session, *, user_id, values):
        updates.append(values["settings_json"])
        settings.settings_json = values["settings_json"]
        return settings

    monkeypatch.setattr(life_context_module.repo, "update_user_settings", update_user_settings)

    text, next_review = await answer_life_context_candidate(
        object(),
        user=user,
        user_settings=settings,
        answer="Активний",
        answer_kind="option",
    )

    assert "Ок" in text
    assert next_review is None
    assert updates[-1]["life_context_items"][0]["label"] == "альбом"
    assert updates[-1]["life_context_items"][0]["answer"] == "Активний"
    assert "pending_life_context_review" not in updates[-1]


@pytest.mark.asyncio
async def test_answer_life_context_candidate_no_does_not_store_fact(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    candidate = {
        "id": "candidate-1",
        "category": "person",
        "label": "Аня",
        "hypothesis": "Аня була поруч у реальності.",
        "question": "Це була реальна подія?",
        "question_type": "boundary",
        "options": [],
    }
    settings = SimpleNamespace(settings_json={"pending_life_context_review": {"index": 0, "candidates": [candidate]}})
    updates = []

    async def update_user_settings(session, *, user_id, values):
        updates.append(values["settings_json"])
        settings.settings_json = values["settings_json"]
        return settings

    monkeypatch.setattr(life_context_module.repo, "update_user_settings", update_user_settings)

    await answer_life_context_candidate(
        object(),
        user=user,
        user_settings=settings,
        answer="Ні, це був сон.",
        answer_kind="no",
    )

    assert updates[-1].get("life_context_items") in (None, [])


def test_format_life_context_question_includes_progress() -> None:
    review = {
        "index": 0,
        "candidates": [
            {
                "question": "“Море” — це назва треку?",
                "why_it_matters": "Щоб не плутати назву з місцем.",
            }
        ],
    }

    assert current_life_context_candidate(review)["question"] == "“Море” — це назва треку?"
    text = format_life_context_question(review)
    assert text.startswith("1/1. “Море”")
    assert "Щоб не плутати" in text


@pytest.mark.asyncio
async def test_auto_life_context_review_respects_cooldown(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    now = datetime(2026, 7, 2, 12, 0, tzinfo=UTC)
    settings = SimpleNamespace(
        settings_json={"life_context_last_auto_offer_at": (now - timedelta(hours=1)).isoformat()}
    )
    calls = {"entries": 0}

    async def get_recent_entries(session, *, user_id, limit):
        calls["entries"] += 1
        return []

    monkeypatch.setattr(life_context_module.repo, "get_recent_entries", get_recent_entries)

    result = await maybe_start_auto_life_context_review(
        object(),
        user=user,
        user_settings=settings,
        ai_service=object(),
        now=now,
    )

    assert result is None
    assert calls["entries"] == 0


@pytest.mark.asyncio
async def test_auto_life_context_review_creates_pending_review(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    settings = SimpleNamespace(settings_json={})
    entries = [
        SimpleNamespace(id=uuid4(), raw_text=f"Запис {index} про альбом", created_at=None, local_timestamp=None, source="manual")
        for index in range(5)
    ]

    class FakeAI:
        async def prune_life_context_items(self, session, *, user_id, context):
            return LifeContextPruneResult(keep_item_ids=[], drop_item_ids=[]), None

        async def extract_life_context_candidates(self, session, *, user_id, context):
            return LifeContextExtraction(
                candidates=[
                    LifeContextCandidate(
                        category="project",
                        label="альбом",
                        hypothesis="Альбом схожий на активний творчий проєкт.",
                        question="Альбом зараз активний творчий проєкт?",
                        question_type="status",
                        options=["Активний", "Пауза"],
                        confidence=0.9,
                    )
                ]
            ), uuid4()

    async def get_recent_entries(session, *, user_id, limit):
        return entries

    async def update_user_settings(session, *, user_id, values):
        settings.settings_json = values["settings_json"]
        return settings

    monkeypatch.setattr(life_context_module.repo, "get_recent_entries", get_recent_entries)
    monkeypatch.setattr(life_context_module.repo, "update_user_settings", update_user_settings)

    result = await maybe_start_auto_life_context_review(
        object(),
        user=user,
        user_settings=settings,
        ai_service=FakeAI(),
        now=datetime(2026, 7, 2, 12, 0, tzinfo=UTC),
    )

    assert result is not None
    _lead_text, review = result
    assert review["candidates"][0]["label"] == "альбом"
    assert settings.settings_json["pending_life_context_review"]["id"] == review["id"]
    assert "life_context_last_auto_offer_at" in settings.settings_json
