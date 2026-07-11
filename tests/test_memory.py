from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import mental_state_bot.services.memory as memory_module
from mental_state_bot.services.memory import build_entry_memory_context


async def test_build_entry_memory_context_collects_capsule_context(monkeypatch) -> None:
    user_id = uuid4()
    day_id = uuid4()
    snapshot_id = uuid4()
    target_id = uuid4()
    before = SimpleNamespace(
        id=uuid4(),
        user_id=user_id,
        day_id=day_id,
        snapshot_id=None,
        source="manual",
        raw_text="Перед цим ішов на зустріч",
        created_at=datetime(2026, 7, 9, 9, 0, tzinfo=UTC),
        local_timestamp=datetime(2026, 7, 9, 12, 0, tzinfo=UTC),
        meta={},
    )
    target = SimpleNamespace(
        id=target_id,
        user_id=user_id,
        day_id=day_id,
        snapshot_id=snapshot_id,
        source="text",
        raw_text='Працюю над "Назва", настрій гарний',
        created_at=datetime(2026, 7, 9, 10, 0, tzinfo=UTC),
        local_timestamp=datetime(2026, 7, 9, 13, 0, tzinfo=UTC),
        meta={"voice": False},
    )
    after = SimpleNamespace(
        id=uuid4(),
        user_id=user_id,
        day_id=day_id,
        snapshot_id=None,
        source="manual",
        raw_text="Потім вийшов гуляти",
        created_at=datetime(2026, 7, 9, 11, 0, tzinfo=UTC),
        local_timestamp=datetime(2026, 7, 9, 14, 0, tzinfo=UTC),
        meta={},
    )
    prompts = [
        SimpleNamespace(
            prompt_kind="initial",
            text="Що зараз робиш з проєктом?",
            sent_at=datetime(2026, 7, 9, 9, 55, tzinfo=UTC),
        )
    ]
    analyses = [
        SimpleNamespace(
            task_name="extract_entry_features",
            result={
                "activity_labels": ["мастеринг"],
                "emotions": [
                    {
                        "label": "радість",
                        "intensity_level": "moderate",
                        "evidence": "настрій гарний",
                        "time_scope": "current",
                    }
                ],
            },
        ),
        SimpleNamespace(task_name="generate_micro_summary", result={"text": "Я почув, що ти опрацьовуєш трек."}),
        SimpleNamespace(
            task_name="apply_correction",
            result={"correction_text": '"Назва" — це назва елемента проєкту', "corrected_at": "2026-07-09T13:10:00+03:00"},
        ),
    ]
    settings = SimpleNamespace(
        settings_json={
            "life_context_items": [
                {"category": "project", "label": "проєкт", "value": "важливий музичний проєкт"},
                {"category": "term", "label": "Назва", "value": "назва елемента проєкту"},
            ]
        }
    )

    async def list_analyses_for_targets(session, *, target_type, target_ids):
        assert target_type == "entry"
        assert target_ids == [target_id]
        return analyses

    async def list_day_entries(session, *, day_id):
        return [before, target, after]

    async def get_snapshot_prompts(session, *, snapshot_id):
        return prompts

    async def get_user_settings(session, user_id):
        return settings

    async def relevant_graph_context(session, *, user_id, text, limit=12):
        assert "Назва" in text
        return {"nodes": [{"label": "Назва"}], "edges": [], "matched": ["Назва"]}

    monkeypatch.setattr(memory_module.repo, "list_analyses_for_targets", list_analyses_for_targets)
    monkeypatch.setattr(memory_module.repo, "list_day_entries", list_day_entries)
    monkeypatch.setattr(memory_module.repo, "get_snapshot_prompts", get_snapshot_prompts)
    monkeypatch.setattr(memory_module.repo, "get_user_settings", get_user_settings)
    monkeypatch.setattr(memory_module, "relevant_memory_context_for_text", relevant_graph_context)

    context = await build_entry_memory_context(object(), entry=target, user_id=user_id)

    assert context["memory_kind"] == "contextual_entry_capsule"
    assert context["entry"]["raw_text"] == 'Працюю над "Назва", настрій гарний'
    assert context["snapshot"]["latest_prompt"] == "Що зараз робиш з проєктом?"
    assert [item["relation_to_target"] for item in context["local_day_window"]["entries"]] == [
        "before",
        "target",
        "after",
    ]
    assert context["features"]["activity_labels"] == ["мастеринг"]
    assert context["micro_summary"] == "Я почув, що ти опрацьовуєш трек."
    assert context["corrections"][0]["correction_text"] == '"Назва" — це назва елемента проєкту'
    assert context["life_context"][1]["label"] == "Назва"
    assert context["relevant_memory_graph"]["matched"] == ["Назва"]
