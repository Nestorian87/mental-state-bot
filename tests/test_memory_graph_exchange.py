from __future__ import annotations

import json
from uuid import uuid4

import pytest

from mental_state_bot.services.memory_graph_exchange import (
    MEMORY_GRAPH_EXPORT_SCHEMA,
    parse_memory_graph_import,
)


def test_graph_import_keeps_valid_structure_and_demotes_lexicon_confirmation() -> None:
    entry_id = uuid4()
    raw = json.dumps(
        {
            "schema_version": MEMORY_GRAPH_EXPORT_SCHEMA,
            "nodes": [
                {
                    "key": "phrase",
                    "label": "умовний вираз",
                    "kind": "lexicon",
                    "summary": "має обережне значення в цьому контексті",
                    "confidence": 0.99,
                    "weight": 0.8,
                    "status": "confirmed",
                },
                {
                    "key": "theme",
                    "label": "важлива тема",
                    "kind": "theme",
                    "confidence": 0.8,
                    "weight": 0.7,
                    "status": "hypothesis",
                },
            ],
            "edges": [
                {
                    "key": "relation",
                    "source_key": "phrase",
                    "target_key": "theme",
                    "relation_label": "clarifies",
                    "confidence": 0.7,
                    "weight": 0.5,
                    "status": "hypothesis",
                },
                {
                    "key": "broken",
                    "source_key": "missing",
                    "target_key": "theme",
                    "relation_label": "related_to",
                },
            ],
            "evidence": [
                {
                    "node_key": "phrase",
                    "target_type": "entry",
                    "target_id": str(entry_id),
                    "evidence_text": "коротке пояснення виразу",
                    "confidence": 0.8,
                },
                {
                    "edge_key": "broken",
                    "target_type": "entry",
                    "target_id": str(entry_id),
                    "evidence_text": "невалідний зв’язок",
                },
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")

    payload, preview = parse_memory_graph_import(raw)

    assert preview.nodes == 2
    assert preview.edges == 1
    assert preview.evidence == 1
    assert preview.skipped == 2
    assert payload["nodes"][0]["status"] == "hypothesis"
    assert payload["edges"][0]["source_key"] == "phrase"


def test_graph_import_rejects_a_file_without_valid_nodes() -> None:
    raw = json.dumps(
        {
            "schema_version": MEMORY_GRAPH_EXPORT_SCHEMA,
            "nodes": [{"key": "empty", "label": ""}],
        }
    ).encode("utf-8")

    with pytest.raises(ValueError, match="немає жодного коректного вузла"):
        parse_memory_graph_import(raw)
