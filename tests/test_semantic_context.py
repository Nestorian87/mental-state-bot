from __future__ import annotations

from mental_state_bot.services.semantic_context import verified_semantic_memory_insight


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


def test_semantic_memory_insight_requires_hypothesis_and_real_evidence() -> None:
    assert verified_semantic_memory_insight(
        {"used": True, "hypothesis": "припущення", "evidence_entry_ids": ["invented"]},
        [{"target_id": "entry-a"}],
    ) is None
