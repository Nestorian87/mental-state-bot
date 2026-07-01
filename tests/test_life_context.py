from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from mental_state_bot.services import life_context as life_context_module
from mental_state_bot.services.life_context import (
    answer_life_context_candidate,
    current_life_context_candidate,
    format_life_context_question,
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
