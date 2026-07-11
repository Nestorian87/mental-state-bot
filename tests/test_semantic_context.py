from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import mental_state_bot.services.semantic_context as semantic_context_module
from mental_state_bot.services.semantic_context import (
    _situation_context_for_records,
    verified_semantic_memory_insight,
)


def test_semantic_memory_insight_keeps_only_retrieved_evidence() -> None:
    insight = verified_semantic_memory_insight(
        {
            "used": True,
            "hypothesis": "схожа ситуація з очікуванням відповіді може бути доречною довідкою",
            "evidence_entry_ids": ["entry-a", "invented"],
            "confidence": 0.8,
        },
        [{"target_id": "entry-a"}, {"target_id": "entry-b"}],
    )

    assert insight == {
        "used": True,
        "hypothesis": "схожа ситуація з очікуванням відповіді може бути доречною довідкою",
        "evidence_entry_ids": ["entry-a"],
        "confidence": 0.8,
    }


def test_semantic_memory_insight_keeps_situations_only_from_cited_evidence() -> None:
    insight = verified_semantic_memory_insight(
        {
            "used": True,
            "hypothesis": "обережна схожість",
            "evidence_entry_ids": ["entry-a"],
            "confidence": 0.7,
        },
        [
            {"target_id": "entry-a", "situations": [{"label": "повторюваний контекст"}]},
            {"target_id": "entry-b", "situations": [{"label": "не використовувати"}]},
        ],
    )

    assert insight is not None
    assert insight["situation_labels"] == ["повторюваний контекст"]


def test_semantic_memory_insight_requires_hypothesis_and_real_evidence() -> None:
    assert verified_semantic_memory_insight(
        {"used": True, "hypothesis": "припущення", "evidence_entry_ids": ["invented"]},
        [{"target_id": "entry-a"}],
    ) is None


async def test_situation_context_groups_evidence_by_retrieved_entry(monkeypatch) -> None:
    entry_id = uuid4()
    node = SimpleNamespace(
        id=uuid4(),
        label="очікування без відповіді",
        summary="Очікування іншої людини лишилося невизначеним.",
        status="hypothesis",
        confidence=0.7,
        weight=0.6,
        meta={"situation": {"evidence_count": 2}},
    )

    async def list_situation_nodes_for_entry_targets(session, *, user_id, entry_ids):
        assert entry_ids == [entry_id]
        return [(entry_id, node), (entry_id, node)]

    monkeypatch.setattr(
        semantic_context_module.repo,
        "list_situation_nodes_for_entry_targets",
        list_situation_nodes_for_entry_targets,
    )
    context = await _situation_context_for_records(
        object(),
        user_id=uuid4(),
        records=[SimpleNamespace(target_type="entry", target_id=entry_id)],
    )

    assert context[str(entry_id)][0]["label"] == "очікування без відповіді"
    assert context[str(entry_id)][0]["evidence_count"] == 2
    assert len(context[str(entry_id)]) == 1
