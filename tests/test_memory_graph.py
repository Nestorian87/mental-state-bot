from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import mental_state_bot.services.memory_graph as graph_module
from mental_state_bot.ai.schemas import (
    MemoryGraphEdgeCandidate,
    MemoryGraphExtraction,
    MemoryGraphNodeCandidate,
    MemoryGraphReviewDecision,
    MemoryGraphReviewResult,
)
from mental_state_bot.services.memory_graph import (
    apply_memory_graph_extraction,
    decay_memory_graph,
    format_personal_lexicon_view,
    maintain_memory_graph,
    relevant_memory_context_for_text,
    review_memory_graph_duplicates,
)


async def test_apply_memory_graph_extraction_upserts_nodes_edges_and_evidence(monkeypatch) -> None:
    user_id = uuid4()
    entry_id = uuid4()
    nodes = {}
    edges = {}
    evidence = []

    async def get_memory_nodes_by_normalized_labels(session, *, user_id, labels):
        return [nodes[label] for label in labels if label in nodes]

    async def add_memory_node(session, **kwargs):
        node = SimpleNamespace(id=uuid4(), **kwargs)
        nodes[kwargs["normalized_label"]] = node
        return node

    async def get_memory_edge(session, *, user_id, source_node_id, target_node_id, relation_label):
        return edges.get((source_node_id, target_node_id, relation_label))

    async def add_memory_edge(session, **kwargs):
        edge = SimpleNamespace(id=uuid4(), **kwargs)
        edges[(kwargs["source_node_id"], kwargs["target_node_id"], kwargs["relation_label"])] = edge
        return edge

    async def add_memory_evidence(session, **kwargs):
        evidence.append(kwargs)
        return SimpleNamespace(id=uuid4(), **kwargs)

    monkeypatch.setattr(graph_module.repo, "get_memory_nodes_by_normalized_labels", get_memory_nodes_by_normalized_labels)
    monkeypatch.setattr(graph_module.repo, "add_memory_node", add_memory_node)
    monkeypatch.setattr(graph_module.repo, "get_memory_edge", get_memory_edge)
    monkeypatch.setattr(graph_module.repo, "add_memory_edge", add_memory_edge)
    monkeypatch.setattr(graph_module.repo, "add_memory_evidence", add_memory_evidence)

    extraction = MemoryGraphExtraction(
        nodes=[
            MemoryGraphNodeCandidate(
                label="Проєкт",
                kind="project",
                summary="Музичний проєкт",
                evidence="працював над проєктом",
                confidence=0.8,
            ),
            MemoryGraphNodeCandidate(
                label="Натхнення",
                kind="state",
                evidence="з'явилось натхнення",
                confidence=0.7,
            ),
        ],
        edges=[
            MemoryGraphEdgeCandidate(
                source_label="Проєкт",
                relation_label="related_to",
                target_label="Натхнення",
                evidence="робота над проєктом дала натхнення",
                confidence=0.75,
            )
        ],
    )

    result = await apply_memory_graph_extraction(
        object(),
        user_id=user_id,
        entry=SimpleNamespace(id=entry_id, created_at=datetime(2026, 7, 9, tzinfo=UTC)),
        extraction=extraction,
    )

    assert result.nodes_created == 2
    assert result.edges_created == 1
    assert set(nodes) == {"проєкт", "натхнення"}
    assert len(edges) == 1
    assert len(evidence) == 3


async def test_decay_memory_graph_reduces_old_candidates_without_deleting_them(monkeypatch) -> None:
    node = SimpleNamespace(
        id=uuid4(),
        weight=1,
        confidence=0.5,
        status="hypothesis",
        meta={"last_decay_at": "2020-01-01T00:00:00+00:00"},
    )
    edge = SimpleNamespace(
        id=uuid4(),
        weight=1,
        confidence=0.5,
        status="candidate",
        meta={"last_decay_at": "2020-01-01T00:00:00+00:00"},
    )

    async def list_nodes(session, *, user_id, limit):
        return [node]

    async def list_edges(session, *, user_id, limit):
        return [edge]

    monkeypatch.setattr(graph_module.repo, "list_memory_nodes", list_nodes)
    monkeypatch.setattr(graph_module.repo, "list_memory_edges", list_edges)

    result = await decay_memory_graph(
        object(),
        user_id=uuid4(),
        now=datetime(2026, 7, 10, tzinfo=UTC),
    )

    assert result.nodes_updated == 1
    assert result.edges_updated == 1
    assert result.nodes_staled == 1
    assert result.edges_staled == 1
    assert node.status == "stale"
    assert edge.status == "stale"
    assert node.weight >= 0
    assert edge.weight >= 0


async def test_maintain_memory_graph_marks_possible_duplicates_without_staling(monkeypatch) -> None:
    album = SimpleNamespace(
        id=uuid4(),
        label="Проєкт",
        aliases=[],
        kind="project",
        weight=0.8,
        confidence=0.8,
        status="hypothesis",
        summary="Музичний проєкт",
        last_seen_at=datetime(2026, 7, 10, tzinfo=UTC),
        meta={},
    )
    my_album = SimpleNamespace(
        id=uuid4(),
        label="Мій проєкт",
        aliases=[],
        kind="project",
        weight=0.4,
        confidence=0.4,
        status="candidate",
        summary=None,
        last_seen_at=datetime(2026, 7, 9, tzinfo=UTC),
        meta={},
    )

    async def list_nodes(session, *, user_id, limit):
        return [album, my_album]

    async def list_edges(session, *, user_id, limit):
        return []

    monkeypatch.setattr(graph_module.repo, "list_memory_nodes", list_nodes)
    monkeypatch.setattr(graph_module.repo, "list_memory_edges", list_edges)

    result = await maintain_memory_graph(
        object(),
        user_id=uuid4(),
        now=datetime(2026, 7, 10, tzinfo=UTC),
    )

    assert result.nodes_checked == 2
    assert result.duplicate_pairs_found == 1
    assert result.nodes_marked_candidate == 2
    assert album.status == "hypothesis"
    assert my_album.status == "candidate"
    assert album.meta["possible_duplicates"][0]["node_id"] == str(my_album.id)
    assert my_album.meta["possible_duplicates"][0]["node_id"] == str(album.id)


async def test_review_memory_graph_duplicates_applies_high_confidence_alias(monkeypatch) -> None:
    album_id = uuid4()
    my_album_id = uuid4()
    album = SimpleNamespace(
        id=album_id,
        label="Проєкт",
        aliases=[],
        kind="project",
        weight=0.8,
        confidence=0.8,
        status="hypothesis",
        summary="Музичний проєкт",
        last_seen_at=datetime(2026, 7, 10, tzinfo=UTC),
        meta={"possible_duplicates": [{"node_id": str(my_album_id), "label": "Мій проєкт", "score": 0.96}]},
    )
    my_album = SimpleNamespace(
        id=my_album_id,
        label="Мій проєкт",
        aliases=[],
        kind="project",
        weight=0.4,
        confidence=0.4,
        status="candidate",
        summary=None,
        last_seen_at=datetime(2026, 7, 9, tzinfo=UTC),
        meta={"possible_duplicates": [{"node_id": str(album_id), "label": "Проєкт", "score": 0.96}]},
    )

    async def list_nodes(session, *, user_id, limit):
        return [album, my_album]

    class FakeAIService:
        async def review_memory_graph_pairs(self, session, *, user_id, context):
            return (
                MemoryGraphReviewResult(
                    decisions=[
                        MemoryGraphReviewDecision(
                            pair_id=context["pairs"][0]["pair_id"],
                            decision="alias",
                            canonical_node_id=str(album_id),
                            alias_text="Мій проєкт",
                            confidence=0.93,
                        )
                    ]
                ),
                uuid4(),
            )

    monkeypatch.setattr(graph_module.repo, "list_memory_nodes", list_nodes)

    result = await review_memory_graph_duplicates(
        object(),
        user_id=uuid4(),
        ai_service=FakeAIService(),
    )

    assert result.pairs_selected == 1
    assert result.aliases_added == 1
    assert result.nodes_staled_as_duplicate == 1
    assert "Мій проєкт" in album.aliases
    assert my_album.status == "stale"
    assert my_album.meta["duplicate_of"] == str(album_id)


async def test_relevant_memory_context_matches_short_inflected_labels(monkeypatch) -> None:
    node = SimpleNamespace(
        id=uuid4(),
        label="Лумія",
        aliases=[],
        kind="person",
        weight=0.7,
        confidence=0.8,
        status="hypothesis",
        summary="Нейтральний тестовий вузол",
    )

    async def list_nodes(session, *, user_id, limit):
        return [node]

    async def list_edges_for_nodes(session, *, user_id, node_ids, limit):
        return []

    async def add_retrieval_log(session, **kwargs):
        return SimpleNamespace(id=uuid4(), **kwargs)

    monkeypatch.setattr(graph_module.repo, "list_memory_nodes", list_nodes)
    monkeypatch.setattr(graph_module.repo, "list_memory_edges_for_nodes", list_edges_for_nodes)
    monkeypatch.setattr(graph_module.repo, "add_retrieval_log", add_retrieval_log)

    context = await relevant_memory_context_for_text(
        object(),
        user_id=uuid4(),
        text="сьогодні говорив із Лумією",
    )

    assert context["matched"] == ["Лумія"]
    assert context["nodes"][0]["label"] == "Лумія"


async def test_personal_lexicon_node_stays_conditional_even_if_ai_marks_it_confirmed(monkeypatch) -> None:
    user_id = uuid4()
    entry_id = uuid4()
    nodes = {}

    async def get_memory_nodes_by_normalized_labels(session, *, user_id, labels):
        return [nodes[label] for label in labels if label in nodes]

    async def add_memory_node(session, **kwargs):
        node = SimpleNamespace(id=uuid4(), **kwargs)
        nodes[kwargs["normalized_label"]] = node
        return node

    async def add_memory_evidence(session, **kwargs):
        return SimpleNamespace(id=uuid4(), **kwargs)

    monkeypatch.setattr(graph_module.repo, "get_memory_nodes_by_normalized_labels", get_memory_nodes_by_normalized_labels)
    monkeypatch.setattr(graph_module.repo, "add_memory_node", add_memory_node)
    monkeypatch.setattr(graph_module.repo, "add_memory_evidence", add_memory_evidence)

    await apply_memory_graph_extraction(
        object(),
        user_id=user_id,
        entry=SimpleNamespace(id=entry_id, created_at=datetime(2026, 7, 10, tzinfo=UTC)),
        extraction=MemoryGraphExtraction(
            nodes=[
                MemoryGraphNodeCandidate(
                    label="умовний вираз",
                    kind="lexicon",
                    summary="у цьому контексті означає коротке відчуття виснаження",
                    evidence="умовний вираз, бо після справи зовсім без сил",
                    confidence=1.0,
                    weight=1.0,
                    status="confirmed",
                )
            ]
        ),
    )

    node = nodes["умовний вираз"]
    assert node.kind == "lexicon"
    assert node.status == "hypothesis"
    assert float(node.confidence) == 0.75
    assert float(node.weight) == 0.7
    assert node.meta["personal_lexicon"]["conditional"] is True
    assert node.meta["personal_lexicon"]["evidence_count"] == 1


async def test_personal_lexicon_view_lists_conditional_candidates(monkeypatch) -> None:
    node = SimpleNamespace(
        label="умовний вираз",
        kind="lexicon",
        summary="обережне значення в певному контексті",
        status="candidate",
        confidence=0.55,
        weight=0.4,
        meta={"personal_lexicon": {"evidence_count": 2}},
    )

    async def list_nodes(session, *, user_id, limit):
        return [node]

    monkeypatch.setattr(graph_module.repo, "list_memory_nodes", list_nodes)

    text = await format_personal_lexicon_view(object(), user_id=uuid4())

    assert "Фрази й значення" in text
    assert "умовний вираз" in text
    assert "кандидат" in text
    assert "доказів: 2" in text
