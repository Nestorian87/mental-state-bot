from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from mental_state_bot.ai.schemas import (
    LifeContextAnswerReview,
    LifeContextCandidate,
    LifeContextExtraction,
    LifeContextPruneResult,
    LifeContextRewriteItem,
    LifeContextRewriteResult,
)
from mental_state_bot.services import life_context as life_context_module
from mental_state_bot.services.life_context import (
    answer_life_context_candidate,
    apply_life_context_rewrite,
    current_life_context_candidate,
    format_life_context_question,
    maybe_start_auto_life_context_review,
    prune_life_context_items_if_needed,
    start_life_context_review,
    start_life_context_rewrite,
)


@pytest.mark.asyncio
async def test_answer_life_context_candidate_stores_confirmed_answer(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    candidate = {
        "id": "candidate-1",
        "category": "project",
        "label": "проєкт",
        "hypothesis": "Проєкт зараз активний творчий проєкт.",
        "question": "Проєкт зараз активний проєкт?",
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
    assert updates[-1]["life_context_items"][0]["label"] == "проєкт"
    assert updates[-1]["life_context_items"][0]["answer"] == "Активний"
    assert "pending_life_context_review" not in updates[-1]


@pytest.mark.asyncio
async def test_answer_life_context_candidate_no_does_not_store_fact(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    candidate = {
        "id": "candidate-1",
        "category": "person",
        "label": "контакт",
        "hypothesis": "Контакт був поруч у реальності.",
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


@pytest.mark.asyncio
async def test_yes_life_context_answer_stores_hypothesis_not_yes(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    candidate = {
        "id": "candidate-1",
        "category": "project",
        "label": "проєкт",
        "hypothesis": "Проєкт зараз активний творчий проєкт.",
        "question": "Проєкт зараз активний творчий проєкт?",
        "question_type": "confirm",
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
        answer="Так",
        answer_kind="yes",
    )

    assert updates[-1]["life_context_items"][0]["answer"] == "Проєкт зараз активний творчий проєкт."


@pytest.mark.asyncio
async def test_free_life_context_answer_waits_for_confirmation(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    candidate = {
        "id": "candidate-1",
        "category": "term",
        "label": "Назва",
        "hypothesis": "“Назва” може бути назвою треку.",
        "question": "“Назва” — це назва елемента проєкту?",
        "question_type": "boundary",
        "options": [],
    }
    settings = SimpleNamespace(settings_json={"pending_life_context_review": {"index": 0, "candidates": [candidate]}})
    updates = []

    class FakeAI:
        async def review_life_context_answer(self, session, *, user_id, context):
            return (
                LifeContextAnswerReview(
                    decision="store",
                    normalized_answer="“Назва” — це назва елемента проєкту, а не місце.",
                    confidence=0.9,
                ),
                None,
            )

    async def update_user_settings(session, *, user_id, values):
        updates.append(values["settings_json"])
        settings.settings_json = values["settings_json"]
        return settings

    monkeypatch.setattr(life_context_module.repo, "update_user_settings", update_user_settings)

    text, next_review = await answer_life_context_candidate(
        object(),
        user=user,
        user_settings=settings,
        answer="це трек, я не на морі",
        answer_kind="free",
        ai_service=FakeAI(),
    )

    assert "Перевір" in text
    assert next_review is not None
    assert updates[-1].get("life_context_items") in (None, [])
    current = current_life_context_candidate(next_review)
    assert current["pending_normalized_answer"] == "“Назва” — це назва елемента проєкту, а не місце."
    assert "Зберегти саме так" in current["question"]


@pytest.mark.asyncio
async def test_confirmed_life_context_answer_stores_normalized_text(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    candidate = {
        "id": "candidate-1",
        "category": "term",
        "label": "Назва",
        "hypothesis": "“Назва” може бути назвою треку.",
        "question": "Я б записав так: «“Назва” — це назва елемента проєкту, а не місце.». Зберегти саме так?",
        "question_type": "confirm",
        "options": [],
        "pending_normalized_answer": "“Назва” — це назва елемента проєкту, а не місце.",
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
        answer="Так",
        answer_kind="yes",
    )

    item = updates[-1]["life_context_items"][0]
    assert item["answer"] == "“Назва” — це назва елемента проєкту, а не місце."
    assert "pending_normalized_answer" not in item


@pytest.mark.asyncio
async def test_life_context_rewrite_proposes_changes_without_applying(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    settings = SimpleNamespace(
        settings_json={
            "life_context_items": [
                {
                    "id": "item-1",
                    "category": "project",
                    "label": "проєкт",
                    "hypothesis": "Проєкт зараз активний творчий проєкт.",
                    "answer": "Так",
                }
            ]
        }
    )
    updates = []

    class FakeAI:
        async def rewrite_life_context_items(self, session, *, user_id, context):
            return (
                LifeContextRewriteResult(
                    items=[
                        LifeContextRewriteItem(
                            id="item-1",
                            action="rewrite",
                            label="проєкт",
                            answer="Проєкт зараз активний творчий проєкт.",
                        ),
                        LifeContextRewriteItem(
                            id="invented",
                            action="rewrite",
                            label="вигадане",
                            answer="Цього не має бути.",
                        ),
                    ]
                ),
                None,
            )

    async def update_user_settings(session, *, user_id, values):
        updates.append(values["settings_json"])
        settings.settings_json = values["settings_json"]
        return settings

    monkeypatch.setattr(life_context_module.repo, "update_user_settings", update_user_settings)

    text, rewrite = await start_life_context_rewrite(
        object(),
        user=user,
        user_settings=settings,
        ai_service=FakeAI(),
    )

    assert rewrite is not None
    assert "Переписати" in text
    assert updates[-1]["life_context_items"][0]["answer"] == "Так"
    assert updates[-1]["pending_life_context_rewrite"]["items"][0]["answer"] == (
        "Проєкт зараз активний творчий проєкт."
    )
    assert all(item["id"] != "invented" for item in updates[-1]["pending_life_context_rewrite"]["items"])


@pytest.mark.asyncio
async def test_apply_life_context_rewrite_updates_items(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    rewritten_items = [
        {
            "id": "item-1",
            "category": "project",
            "label": "проєкт",
            "answer": "Проєкт зараз активний творчий проєкт.",
        }
    ]
    settings = SimpleNamespace(
        settings_json={
            "life_context_items": [{"id": "item-1", "category": "project", "label": "проєкт", "answer": "Так"}],
            "pending_life_context_rewrite": {"items": rewritten_items, "changes": []},
        }
    )
    updates = []

    async def update_user_settings(session, *, user_id, values):
        updates.append(values["settings_json"])
        settings.settings_json = values["settings_json"]
        return settings

    monkeypatch.setattr(life_context_module.repo, "update_user_settings", update_user_settings)

    text = await apply_life_context_rewrite(object(), user=user, user_settings=settings)

    assert "оновив" in text
    assert updates[-1]["life_context_items"] == rewritten_items
    assert "pending_life_context_rewrite" not in updates[-1]


@pytest.mark.asyncio
async def test_life_context_prune_marks_stale_before_deleting(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    settings = SimpleNamespace(
        settings_json={
            "life_context_items": [
                {"id": "item-1", "category": "project", "label": "старий проєкт", "answer": "Колись був активний."}
            ]
        }
    )
    updates = []

    class FakeAI:
        async def prune_life_context_items(self, session, *, user_id, context):
            return LifeContextPruneResult(keep_item_ids=[], drop_item_ids=["item-1"]), None

    async def update_user_settings(session, *, user_id, values):
        updates.append(values["settings_json"])
        settings.settings_json = values["settings_json"]
        return settings

    monkeypatch.setattr(life_context_module.repo, "update_user_settings", update_user_settings)

    await prune_life_context_items_if_needed(
        object(),
        user=user,
        user_settings=settings,
        ai_service=FakeAI(),
        now=datetime(2026, 7, 1, tzinfo=UTC),
    )

    item = updates[-1]["life_context_items"][0]
    assert item["id"] == "item-1"
    assert item["decay_status"] == "stale"


@pytest.mark.asyncio
async def test_life_context_prune_deletes_stale_item_after_grace_period(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    settings = SimpleNamespace(
        settings_json={
            "life_context_items": [
                {
                    "id": "item-1",
                    "category": "project",
                    "label": "старий проєкт",
                    "answer": "Колись був активний.",
                    "decay_status": "stale",
                    "decay_marked_at": "2026-07-01T00:00:00+00:00",
                }
            ]
        }
    )
    updates = []

    class FakeAI:
        async def prune_life_context_items(self, session, *, user_id, context):
            return LifeContextPruneResult(keep_item_ids=[], drop_item_ids=["item-1"]), None

    async def update_user_settings(session, *, user_id, values):
        updates.append(values["settings_json"])
        settings.settings_json = values["settings_json"]
        return settings

    monkeypatch.setattr(life_context_module.repo, "update_user_settings", update_user_settings)

    await prune_life_context_items_if_needed(
        object(),
        user=user,
        user_settings=settings,
        ai_service=FakeAI(),
        now=datetime(2026, 7, 9, tzinfo=UTC),
    )

    assert updates[-1]["life_context_items"] == []


@pytest.mark.asyncio
async def test_life_context_prune_revives_stale_item_when_kept(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    settings = SimpleNamespace(
        settings_json={
            "life_context_items": [
                {
                    "id": "item-1",
                    "category": "project",
                    "label": "проєкт",
                    "answer": "Проєкт зараз активний.",
                    "decay_status": "stale",
                    "decay_marked_at": "2026-07-01T00:00:00+00:00",
                }
            ]
        }
    )
    updates = []

    class FakeAI:
        async def prune_life_context_items(self, session, *, user_id, context):
            return LifeContextPruneResult(keep_item_ids=["item-1"], drop_item_ids=[]), None

    async def update_user_settings(session, *, user_id, values):
        updates.append(values["settings_json"])
        settings.settings_json = values["settings_json"]
        return settings

    monkeypatch.setattr(life_context_module.repo, "update_user_settings", update_user_settings)

    await prune_life_context_items_if_needed(
        object(),
        user=user,
        user_settings=settings,
        ai_service=FakeAI(),
        now=datetime(2026, 7, 2, tzinfo=UTC),
    )

    item = updates[-1]["life_context_items"][0]
    assert item["id"] == "item-1"
    assert "decay_status" not in item


@pytest.mark.asyncio
async def test_life_context_review_asks_about_stale_item_relevance(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    settings = SimpleNamespace(
        settings_json={
            "life_context_items": [
                {
                    "id": "item-1",
                    "category": "project",
                    "label": "старий проєкт",
                    "answer": "Старий проєкт колись був активний.",
                    "decay_status": "stale",
                    "decay_marked_at": "2999-07-01T00:00:00+00:00",
                }
            ]
        }
    )
    updates = []
    entry = SimpleNamespace(
        id=uuid4(),
        created_at=datetime(2026, 7, 2, tzinfo=UTC),
        local_timestamp=None,
        source="manual",
        raw_text="Сьогодні працюю над іншим.",
    )

    class FakeAI:
        async def prune_life_context_items(self, session, *, user_id, context):
            return LifeContextPruneResult(keep_item_ids=[], drop_item_ids=["item-1"]), None

        async def extract_life_context_candidates(self, session, *, user_id, context):
            return LifeContextExtraction(candidates=[]), None

    async def get_recent_entries(session, *, user_id, limit):
        return [entry]

    async def update_user_settings(session, *, user_id, values):
        updates.append(values["settings_json"])
        settings.settings_json = values["settings_json"]
        return settings

    monkeypatch.setattr(life_context_module.repo, "get_recent_entries", get_recent_entries)
    monkeypatch.setattr(life_context_module.repo, "update_user_settings", update_user_settings)

    text, review = await start_life_context_review(
        object(),
        user=user,
        user_settings=settings,
        ai_service=FakeAI(),
    )

    assert review is not None
    candidate = current_life_context_candidate(review)
    assert candidate["context_action"] == "relevance_check"
    assert "старий проєкт" in candidate["question"]
    assert "актуально" in " ".join(candidate["options"])
    assert "припущень" in text


@pytest.mark.asyncio
async def test_life_context_relevance_answer_can_remove_old_item(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    candidate = {
        "id": "candidate-1",
        "category": "project",
        "label": "старий проєкт",
        "hypothesis": "Старий проєкт колись був активний.",
        "question": "Це ще актуально?",
        "question_type": "status",
        "options": ["Ще актуально", "Вже старий контекст"],
        "context_action": "relevance_check",
        "existing_item_id": "item-1",
    }
    settings = SimpleNamespace(
        settings_json={
            "life_context_items": [
                {
                    "id": "item-1",
                    "category": "project",
                    "label": "старий проєкт",
                    "answer": "Старий проєкт колись був активний.",
                    "decay_status": "stale",
                }
            ],
            "pending_life_context_review": {"index": 0, "candidates": [candidate]},
        }
    )
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
        answer="Вже старий контекст",
        answer_kind="option",
    )

    assert "прибрав" in text
    assert next_review is None
    assert updates[-1]["life_context_items"] == []


def test_format_life_context_question_includes_progress() -> None:
    review = {
        "index": 0,
        "candidates": [
            {
                "question": "“Назва” — це назва елемента проєкту?",
                "why_it_matters": "Щоб не плутати назву з місцем.",
            }
        ],
    }

    assert current_life_context_candidate(review)["question"] == "“Назва” — це назва елемента проєкту?"
    text = format_life_context_question(review)
    assert text.startswith("1/1. “Назва”")
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
        SimpleNamespace(id=uuid4(), raw_text=f"Запис {index} про проєкт", created_at=None, local_timestamp=None, source="manual")
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
                        label="проєкт",
                        hypothesis="Проєкт схожий на активний творчий проєкт.",
                        question="Проєкт зараз активний творчий проєкт?",
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
    assert review["candidates"][0]["label"] == "проєкт"
    assert settings.settings_json["pending_life_context_review"]["id"] == review["id"]
    assert "life_context_last_auto_offer_at" in settings.settings_json


async def test_life_context_review_skips_recently_asked_same_label(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid4())
    settings = SimpleNamespace(
        settings_json={
            "life_context_review_history": [
                {
                    "category": "project",
                    "label": "проєкт",
                    "question": "Проєкт зараз активний творчий проєкт?",
                    "asked_at": "2026-07-02T11:00:00+00:00",
                }
            ]
        }
    )
    entries = [
        SimpleNamespace(id=uuid4(), raw_text="Знову думаю про проєкт", created_at=None, local_timestamp=None, source="manual")
    ]
    updates = []

    class FakeAI:
        async def prune_life_context_items(self, session, *, user_id, context):
            return LifeContextPruneResult(keep_item_ids=[], drop_item_ids=[]), None

        async def extract_life_context_candidates(self, session, *, user_id, context):
            assert context["recent_life_context_questions"][0]["label"] == "проєкт"
            return LifeContextExtraction(
                candidates=[
                    LifeContextCandidate(
                        category="project",
                        label="проєкт",
                        hypothesis="Проєкт схожий на активний творчий проєкт.",
                        question="Проєкт зараз активний творчий проєкт?",
                        question_type="status",
                        options=["Активний", "Пауза"],
                        confidence=0.9,
                    )
                ]
            ), uuid4()

    async def get_recent_entries(session, *, user_id, limit):
        return entries

    async def update_user_settings(session, *, user_id, values):
        updates.append(values["settings_json"])
        settings.settings_json = values["settings_json"]
        return settings

    monkeypatch.setattr(life_context_module.repo, "get_recent_entries", get_recent_entries)
    monkeypatch.setattr(life_context_module.repo, "update_user_settings", update_user_settings)

    text, review = await start_life_context_review(
        object(),
        user=user,
        user_settings=settings,
        ai_service=FakeAI(),
    )

    assert review is None
    assert "не бачу" in text.lower()
    assert updates == []
