from __future__ import annotations

import re
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.ai.schemas import MemoryGraphExtraction, MemoryGraphReviewDecision
from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import Entry, MemoryEdge, MemoryNode
from mental_state_bot.time_utils import utc_now

if TYPE_CHECKING:
    from mental_state_bot.ai.service import AIService


@dataclass(frozen=True)
class MemoryGraphUpdateResult:
    nodes_seen: int = 0
    nodes_created: int = 0
    edges_seen: int = 0
    edges_created: int = 0
    touched_node_ids: tuple[uuid.UUID, ...] = ()


@dataclass(frozen=True)
class MemoryGraphDecayResult:
    nodes_updated: int = 0
    edges_updated: int = 0
    nodes_staled: int = 0
    edges_staled: int = 0


@dataclass(frozen=True)
class MemoryGraphMaintenanceResult:
    nodes_checked: int = 0
    duplicate_pairs_found: int = 0
    nodes_marked_candidate: int = 0
    decay: MemoryGraphDecayResult = MemoryGraphDecayResult()


@dataclass(frozen=True)
class MemoryGraphAIReviewResult:
    pairs_selected: int = 0
    embedding_pairs_found: int = 0
    decisions_received: int = 0
    aliases_added: int = 0
    nodes_staled_as_duplicate: int = 0
    pairs_marked_separate: int = 0
    pairs_needing_confirmation: int = 0
    confirmation_candidates: tuple[MemoryGraphConfirmationCandidate, ...] = ()
    run_id: uuid.UUID | None = None


@dataclass(frozen=True)
class MemoryGraphConfirmationCandidate:
    left_node_id: uuid.UUID
    right_node_id: uuid.UUID
    question: str
    options: tuple[dict[str, str], ...]
    reason: str | None = None


async def sync_confirmed_life_context_item(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    item: dict[str, Any],
    active: bool = True,
) -> MemoryNode | None:
    """Reflect a user-confirmed life-context item in the graph without AI."""
    label = _clean_label(str(item.get("label") or ""))
    answer = _compact(str(item.get("answer") or item.get("hypothesis") or ""))
    if not label or not answer:
        return None

    normalized = _normalize_label(label)
    existing = list(
        await repo.get_memory_nodes_by_normalized_labels(
            session,
            user_id=user_id,
            labels=[normalized],
        )
    )
    now = utc_now()
    if existing:
        node = existing[0]
    else:
        node = await repo.add_memory_node(
            session,
            user_id=user_id,
            label=label,
            normalized_label=normalized,
            kind="life_context",
            aliases=[],
            summary=answer,
            confidence=Decimal("1.000"),
            weight=Decimal("1.000"),
            status="confirmed",
            last_seen_at=now,
            meta={"source": "confirmed_life_context"},
        )

    item_id = str(item.get("id") or "")
    node.kind = "life_context"
    node.summary = answer
    node.confidence = Decimal("1.000")
    node.weight = Decimal("1.000") if active else min(node.weight or Decimal("0.5"), Decimal("0.5"))
    node.status = "confirmed" if active else "stale"
    node.last_seen_at = now
    node.meta = {
        **(node.meta or {}),
        "source": "confirmed_life_context",
        "life_context_item_id": item_id,
        "life_context_active": active,
        "synced_at": now.isoformat(),
    }

    if active and item_id:
        try:
            evidence_target_id = uuid.UUID(item_id)
        except ValueError:
            evidence_target_id = None
        if evidence_target_id is not None:
            await repo.add_memory_evidence(
                session,
                user_id=user_id,
                node_id=node.id,
                edge_id=None,
                target_type="life_context",
                target_id=evidence_target_id,
                evidence_text=answer,
                confidence=Decimal("1.000"),
                meta={"source": "user_confirmation"},
            )
    return node


async def maintain_memory_graph(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    now: datetime | None = None,
    node_limit: int = 800,
) -> MemoryGraphMaintenanceResult:
    """Run cheap graph hygiene without semantic interpretation or AI calls."""
    decay = await decay_memory_graph(session, user_id=user_id, now=now)
    nodes = list(await repo.list_memory_nodes(session, user_id=user_id, limit=node_limit))
    duplicate_pairs = _potential_duplicate_nodes(nodes)
    current_time = now or utc_now()
    touched_node_ids: set[uuid.UUID] = set()

    for keeper, duplicate, score, reason in duplicate_pairs:
        changed = _mark_possible_duplicate(
            duplicate,
            other=keeper,
            score=score,
            reason=reason,
            checked_at=current_time,
        )
        changed = (
            _mark_possible_duplicate(
                keeper,
                other=duplicate,
                score=score,
                reason=reason,
                checked_at=current_time,
            )
            or changed
        )
        if changed:
            touched_node_ids.update({keeper.id, duplicate.id})

    return MemoryGraphMaintenanceResult(
        nodes_checked=len(nodes),
        duplicate_pairs_found=len(duplicate_pairs),
        nodes_marked_candidate=len(touched_node_ids),
        decay=decay,
    )


async def mark_fresh_memory_graph_duplicate_candidates(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    touched_node_ids: set[uuid.UUID],
    node_limit: int = 800,
) -> int:
    """Cheaply refresh only duplicate candidates touched by the current entry."""
    if not touched_node_ids:
        return 0
    nodes = list(await repo.list_memory_nodes(session, user_id=user_id, limit=node_limit))
    current_time = utc_now()
    changed_pairs = 0
    for keeper, duplicate, score, reason in _potential_duplicate_nodes(nodes):
        if keeper.id not in touched_node_ids and duplicate.id not in touched_node_ids:
            continue
        changed = _mark_possible_duplicate(
            duplicate,
            other=keeper,
            score=score,
            reason=reason,
            checked_at=current_time,
        )
        changed = (
            _mark_possible_duplicate(
                keeper,
                other=duplicate,
                score=score,
                reason=reason,
                checked_at=current_time,
            )
            or changed
        )
        if changed:
            changed_pairs += 1
    return changed_pairs


async def review_memory_graph_duplicates(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    ai_service: AIService,
    pair_limit: int = 12,
    use_embedding_candidates: bool = False,
    use_heavy_reasoning: bool = False,
    only_node_ids: set[uuid.UUID] | None = None,
) -> MemoryGraphAIReviewResult:
    nodes = list(await repo.list_memory_nodes(session, user_id=user_id, limit=800))
    pairs = _review_candidate_pairs(nodes, limit=pair_limit, only_node_ids=only_node_ids)
    embedding_pairs_found = 0
    evidence_by_node: dict[str, list[dict[str, Any]]] = {}
    if use_embedding_candidates:
        evidence_by_node = await _review_evidence_by_node(session, user_id=user_id, nodes=nodes)
        embedding_pairs = await _embedding_duplicate_pairs(
            session,
            user_id=user_id,
            nodes=nodes,
            evidence_by_node=evidence_by_node,
            limit=pair_limit,
            only_node_ids=only_node_ids,
        )
        embedding_pairs_found = len(embedding_pairs)
        pairs = _combine_review_pairs([*pairs, *embedding_pairs], limit=pair_limit)
    if not pairs:
        return MemoryGraphAIReviewResult(embedding_pairs_found=embedding_pairs_found)

    if not evidence_by_node:
        evidence_by_node = await _review_evidence_by_node(session, user_id=user_id, nodes=nodes)
    context = {
        "pairs": [
            _review_pair_payload(pair, evidence_by_node=evidence_by_node)
            for pair in pairs
        ]
    }
    if use_heavy_reasoning:
        review, run_id = await ai_service.review_memory_graph_pairs(
            session,
            user_id=user_id,
            context=context,
            use_heavy_reasoning=True,
        )
    else:
        review, run_id = await ai_service.review_memory_graph_pairs(session, user_id=user_id, context=context)
    decisions = {decision.pair_id: decision for decision in review.decisions}
    aliases_added = nodes_staled = separate = needs_confirmation = 0
    confirmation_candidates: list[MemoryGraphConfirmationCandidate] = []

    nodes_by_id = {str(node.id): node for node in nodes}
    for pair_id, left, right, _candidate in pairs:
        decision = decisions.get(pair_id)
        if decision is None:
            continue
        result = _apply_memory_graph_review_decision(
            left,
            right,
            decision,
            nodes_by_id=nodes_by_id,
            reviewed_at=utc_now(),
        )
        aliases_added += result["aliases_added"]
        nodes_staled += result["nodes_staled"]
        separate += result["pairs_marked_separate"]
        needs_confirmation += result["pairs_needing_confirmation"]
        candidate = _confirmation_candidate_for_decision(left, right, decision, result=result)
        if candidate is not None:
            confirmation_candidates.append(candidate)

    return MemoryGraphAIReviewResult(
        pairs_selected=len(pairs),
        embedding_pairs_found=embedding_pairs_found,
        decisions_received=len(decisions),
        aliases_added=aliases_added,
        nodes_staled_as_duplicate=nodes_staled,
        pairs_marked_separate=separate,
        pairs_needing_confirmation=needs_confirmation,
        confirmation_candidates=tuple(confirmation_candidates),
        run_id=run_id,
    )


async def apply_memory_graph_confirmation(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    left_node_id: uuid.UUID,
    right_node_id: uuid.UUID,
    outcome: str,
    reason: str | None = None,
) -> str:
    """Apply an explicit user boundary without altering source evidence."""
    nodes = {
        str(node.id): node
        for node in await repo.get_memory_nodes_by_ids(
            session,
            user_id=user_id,
            node_ids=[left_node_id, right_node_id],
        )
    }
    left = nodes.get(str(left_node_id))
    right = nodes.get(str(right_node_id))
    if left is None or right is None or left.id == right.id:
        return "unavailable"
    reviewed_at = utc_now()
    if outcome == "separate":
        decision = MemoryGraphReviewDecision(
            pair_id=f"{left.id}:{right.id}",
            decision="separate",
            confidence=1.0,
            reason=reason or "user_confirmed_separate",
        )
        _mark_review_separate(left, other=right, decision=decision, reviewed_at=reviewed_at)
        _mark_review_separate(right, other=left, decision=decision, reviewed_at=reviewed_at)
        return "separate"
    if outcome != "same":
        return "defer"

    canonical, duplicate = _duplicate_keeper(left, right)
    aliases_before = set(canonical.aliases or [])
    canonical.aliases = _merge_aliases(canonical.aliases or [], [duplicate.label, *(duplicate.aliases or [])])
    canonical.meta = {
        **(canonical.meta or {}),
        "graph_user_confirmation": {
            "outcome": "same",
            "reviewed_with": str(duplicate.id),
            "reviewed_at": reviewed_at.isoformat(),
            "reason": reason or "user_confirmed_same",
            "aliases_added": max(0, len(set(canonical.aliases or []) - aliases_before)),
        },
        "possible_duplicates": _remove_possible_duplicate((canonical.meta or {}).get("possible_duplicates"), str(duplicate.id)),
    }
    duplicate.status = "stale"
    duplicate.weight = min(duplicate.weight or Decimal("0.250"), Decimal("0.250"))
    duplicate.meta = {
        **(duplicate.meta or {}),
        "stale_reason": "user_confirmed_duplicate",
        "duplicate_of": str(canonical.id),
        "duplicate_of_label": canonical.label,
        "duplicate_confirmed_at": reviewed_at.isoformat(),
        "duplicate_confirmation_reason": reason or "user_confirmed_same",
        "possible_duplicates": _remove_possible_duplicate((duplicate.meta or {}).get("possible_duplicates"), str(canonical.id)),
    }
    return "same"


async def daily_memory_graph_review_context(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    entry_ids: list[uuid.UUID],
) -> dict[str, Any]:
    """Build the bounded daily graph slice for a dedicated model review."""
    nodes = list(await repo.list_memory_nodes_for_entry_targets(session, user_id=user_id, entry_ids=entry_ids))
    if len(nodes) < 2:
        return {"nodes_changed_today": []}
    evidence_by_node = await _review_evidence_by_node(session, user_id=user_id, nodes=nodes)
    return {
        "nodes_changed_today": [
            _review_node_payload(node, evidence=evidence_by_node.get(str(node.id), [])) for node in nodes[:80]
        ]
    }


async def apply_daily_memory_graph_candidates(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    candidates: list[Any],
) -> int:
    """Store AI-suggested daily pairs as candidates; never merge them directly."""
    node_ids: set[uuid.UUID] = set()
    for candidate in candidates:
        for value in (getattr(candidate, "left_node_id", None), getattr(candidate, "right_node_id", None)):
            with suppress(ValueError, TypeError):
                node_ids.add(uuid.UUID(str(value)))
    nodes = {
        str(node.id): node
        for node in await repo.get_memory_nodes_by_ids(session, user_id=user_id, node_ids=list(node_ids))
    }
    marked_pairs: set[tuple[str, str]] = set()
    now = utc_now()
    for candidate in candidates:
        left = nodes.get(str(getattr(candidate, "left_node_id", "")))
        right = nodes.get(str(getattr(candidate, "right_node_id", "")))
        if left is None or right is None or left.id == right.id:
            continue
        pair_key = tuple(sorted([str(left.id), str(right.id)]))
        if pair_key in marked_pairs or _already_reviewed_separate(left, right):
            continue
        confidence = max(0.5, min(0.95, float(getattr(candidate, "confidence", 0.0) or 0.0)))
        reason = _compact(str(getattr(candidate, "reason", "") or ""))[:180] or "daily_model_candidate"
        changed = _mark_possible_duplicate(
            left,
            other=right,
            score=confidence,
            reason=f"daily_model_candidate: {reason}",
            checked_at=now,
        )
        changed = (
            _mark_possible_duplicate(
                right,
                other=left,
                score=confidence,
                reason=f"daily_model_candidate: {reason}",
                checked_at=now,
            )
            or changed
        )
        if changed:
            marked_pairs.add(pair_key)
    return len(marked_pairs)


async def decay_memory_graph(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    now: datetime | None = None,
) -> MemoryGraphDecayResult:
    """Apply bounded decay without deleting evidence or raw data."""
    current_time = now or datetime.now(UTC)
    nodes = list(await repo.list_memory_nodes(session, user_id=user_id, limit=1000))
    edges = list(await repo.list_memory_edges(session, user_id=user_id, limit=2000))
    nodes_updated = nodes_staled = edges_updated = edges_staled = 0

    for node in nodes:
        factor = _decay_factor(
            node.meta or {},
            current_time,
            half_life_days=60.0,
            fallback_at=getattr(node, "last_seen_at", None) or getattr(node, "created_at", None),
        )
        if factor is None:
            continue
        node.weight = _decayed_decimal(node.weight, factor)
        node.meta = {**(node.meta or {}), "last_decay_at": current_time.isoformat()}
        if node.status != "confirmed" and float(node.weight or 0) < 0.08:
            node.status = "stale"
            nodes_staled += 1
        nodes_updated += 1

    for edge in edges:
        factor = _decay_factor(
            edge.meta or {},
            current_time,
            half_life_days=45.0,
            fallback_at=getattr(edge, "last_seen_at", None) or getattr(edge, "created_at", None),
        )
        if factor is None:
            continue
        edge.weight = _decayed_decimal(edge.weight, factor)
        edge.meta = {**(edge.meta or {}), "last_decay_at": current_time.isoformat()}
        if edge.status != "confirmed" and float(edge.weight or 0) < 0.08:
            edge.status = "stale"
            edges_staled += 1
        edges_updated += 1

    return MemoryGraphDecayResult(
        nodes_updated=nodes_updated,
        edges_updated=edges_updated,
        nodes_staled=nodes_staled,
        edges_staled=edges_staled,
    )


def _decay_factor(
    meta: dict[str, Any],
    now: datetime,
    *,
    half_life_days: float,
    fallback_at: datetime | None,
) -> float | None:
    raw = meta.get("last_decay_at")
    if isinstance(raw, str):
        try:
            previous = datetime.fromisoformat(raw)
        except ValueError:
            previous = now
        if previous.tzinfo is None:
            previous = previous.replace(tzinfo=UTC)
        elapsed_days = max(0.0, (now - previous.astimezone(UTC)).total_seconds() / 86400)
    elif fallback_at is not None:
        previous = fallback_at
        if previous.tzinfo is None:
            previous = previous.replace(tzinfo=UTC)
        elapsed_days = max(0.0, (now - previous.astimezone(UTC)).total_seconds() / 86400)
    else:
        return None
    return 0.5 ** (elapsed_days / half_life_days) if elapsed_days > 0 else None


def _decayed_decimal(value: Decimal | None, factor: float) -> Decimal:
    next_value = float(value or 0) * factor
    return Decimal(str(max(0.0, min(1.0, next_value)))).quantize(Decimal("0.001"))


async def apply_memory_graph_extraction(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    entry: Entry,
    extraction: MemoryGraphExtraction,
) -> MemoryGraphUpdateResult:
    now = entry.created_at or utc_now()
    nodes_by_label: dict[str, MemoryNode] = {}
    touched_node_ids: set[uuid.UUID] = set()
    nodes_created = 0
    for candidate in extraction.nodes[:8]:
        label = _clean_label(candidate.label)
        evidence = _compact(candidate.evidence or "")
        if not label or not evidence:
            continue
        node, created = await _upsert_node(
            session,
            user_id=user_id,
            label=label,
            kind=_clean_kind(candidate.kind),
            aliases=_clean_aliases(candidate.aliases),
            summary=_compact(candidate.summary or "") or None,
            confidence=candidate.confidence,
            weight=candidate.weight,
            status=candidate.status,
            seen_at=now,
        )
        await repo.add_memory_evidence(
            session,
            user_id=user_id,
            node_id=node.id,
            edge_id=None,
            target_type="entry",
            target_id=entry.id,
            evidence_text=evidence,
            confidence=_decimal(candidate.confidence),
            meta={"source": "memory_graph_extraction"},
        )
        _record_personal_lexicon_evidence(node, seen_at=now)
        _record_situation_evidence(node, seen_at=now)
        nodes_by_label[_normalize_label(label)] = node
        touched_node_ids.add(node.id)
        nodes_created += int(created)

    edges_created = 0
    for candidate in extraction.edges[:8]:
        source_label = _clean_label(candidate.source_label)
        target_label = _clean_label(candidate.target_label)
        evidence = _compact(candidate.evidence or "")
        relation = _clean_relation(candidate.relation_label)
        if not source_label or not target_label or not relation or not evidence:
            continue
        source = nodes_by_label.get(_normalize_label(source_label))
        if source is None:
            source, created = await _upsert_node(
                session,
                user_id=user_id,
                label=source_label,
                kind="concept",
                aliases=[],
                summary=None,
                confidence=max(0.25, candidate.confidence * 0.75),
                weight=max(0.2, candidate.weight * 0.75),
                status="candidate",
                seen_at=now,
            )
            nodes_by_label[_normalize_label(source_label)] = source
            nodes_created += int(created)
        touched_node_ids.add(source.id)
        target = nodes_by_label.get(_normalize_label(target_label))
        if target is None:
            target, created = await _upsert_node(
                session,
                user_id=user_id,
                label=target_label,
                kind="concept",
                aliases=[],
                summary=None,
                confidence=max(0.25, candidate.confidence * 0.75),
                weight=max(0.2, candidate.weight * 0.75),
                status="candidate",
                seen_at=now,
            )
            nodes_by_label[_normalize_label(target_label)] = target
            nodes_created += int(created)
        touched_node_ids.add(target.id)
        if source.id == target.id:
            continue
        edge, created = await _upsert_edge(
            session,
            user_id=user_id,
            source=source,
            target=target,
            relation_label=relation,
            summary=_compact(candidate.summary or "") or None,
            confidence=candidate.confidence,
            weight=candidate.weight,
            status=candidate.status,
            seen_at=now,
        )
        await repo.add_memory_evidence(
            session,
            user_id=user_id,
            node_id=None,
            edge_id=edge.id,
            target_type="entry",
            target_id=entry.id,
            evidence_text=evidence,
            confidence=_decimal(candidate.confidence),
            meta={"source": "memory_graph_extraction"},
        )
        edges_created += int(created)

    return MemoryGraphUpdateResult(
        nodes_seen=len(nodes_by_label),
        nodes_created=nodes_created,
        edges_seen=min(len(extraction.edges), 8),
        edges_created=edges_created,
        touched_node_ids=tuple(sorted(touched_node_ids, key=str)),
    )


async def relevant_memory_subgraph(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    labels: list[str],
    limit: int = 30,
) -> dict[str, Any]:
    normalized = [_normalize_label(label) for label in labels if _normalize_label(label)]
    nodes = list(await repo.get_memory_nodes_by_normalized_labels(session, user_id=user_id, labels=normalized))
    edges = list(await repo.list_memory_edges_for_nodes(session, user_id=user_id, node_ids=[node.id for node in nodes], limit=limit))
    node_ids = {node.id for node in nodes}
    for edge in edges:
        node_ids.add(edge.source_node_id)
        node_ids.add(edge.target_node_id)
    if node_ids:
        expanded = list(
            await repo.get_memory_nodes_by_normalized_labels(
                session,
                user_id=user_id,
                labels=[node.normalized_label for node in nodes],
            )
        )
    else:
        expanded = []
    return {
        "nodes": [_node_payload(node) for node in expanded],
        "edges": [_edge_payload(edge) for edge in edges],
    }


async def format_personal_lexicon_view(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    limit: int = 80,
) -> str:
    nodes = [
        node
        for node in await repo.list_memory_nodes(session, user_id=user_id, limit=500)
        if node.kind == "lexicon"
    ][:limit]
    if not nodes:
        return (
            "Персональний лексикон поки порожній.\n\n"
            "Він формується автоматично лише коли AI має достатньо підстав вважати, що короткий вираз "
            "має особливе значення у твоєму контексті. Це не список усіх слів, а обережні кандидати."
        )
    lines = ["Фрази й значення", "", "Це робочі гіпотези графа, а не зафіксовані правила.", ""]
    for node in nodes:
        lexicon = dict((node.meta or {}).get("personal_lexicon") or {})
        evidence_count = int(lexicon.get("evidence_count") or 0)
        status = {"candidate": "кандидат", "hypothesis": "гіпотеза", "confirmed": "підтверджено"}.get(
            node.status,
            node.status,
        )
        lines.append(f"• {node.label} — {node.summary or 'значення ще не описано'}")
        lines.append(
            f"  {status}; впевненість {float(node.confidence or 0):.0%}; "
            f"доказів: {evidence_count or 'є'}"
        )
    return "\n".join(lines)


async def relevant_memory_context_for_text(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    text: str,
    limit: int = 12,
    task_name: str = "memory_graph_lookup",
) -> dict[str, Any]:
    """Return a cheap lexical lookup of graph facts explicitly mentioned in text.

    This is lookup only, not semantic interpretation: AI still decides what the
    current entry means. Embeddings and broader graph traversal belong to a
    later, less frequent maintenance/retrieval pass.
    """
    query = _normalize_label(text)
    if not query:
        return {"nodes": [], "edges": [], "matched": []}
    candidates = list(await repo.list_memory_nodes(session, user_id=user_id, limit=300))
    scored: list[tuple[float, MemoryNode]] = []
    for node in candidates:
        labels = [node.label, *(node.aliases or [])]
        score = max((_query_label_match_score(label, query) for label in labels), default=0.0)
        if score <= 0:
            continue
        if node.status == "confirmed":
            score += 40
        scored.append((score, node))
    selected = [node for _, node in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]
    edges = list(
        await repo.list_memory_edges_for_nodes(
            session,
            user_id=user_id,
            node_ids=[node.id for node in selected],
            limit=limit * 3,
        )
    )
    selected_ids = {node.id for node in selected}
    for edge in edges:
        for node_id in (edge.source_node_id, edge.target_node_id):
            if node_id in selected_ids:
                continue
            related = next((node for node in candidates if node.id == node_id), None)
            if related is not None and len(selected) < limit * 2:
                selected.append(related)
                selected_ids.add(node_id)
    context = {
        "nodes": [_node_payload(node) for node in selected],
        "edges": [_edge_payload(edge) for edge in edges],
        "personal_lexicon": [_node_payload(node) for node in selected if node.kind == "lexicon"],
        "matched": [node.label for _, node in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]],
    }
    with suppress(Exception):
        await repo.add_retrieval_log(
            session,
            user_id=user_id,
            task_name=task_name,
            query_text=_compact(text)[:1200],
            provider="local",
            model="memory_graph_lexical_v1",
            retrieved=[
                {
                    "target_type": "memory_node",
                    "target_id": node["id"],
                    "label": node["label"],
                    "status": node["status"],
                    "weight": node["weight"],
                }
                for node in context["nodes"]
            ]
            + [
                {
                    "target_type": "memory_edge",
                    "target_id": edge["id"],
                    "relation": edge["relation_label"],
                    "source_node_id": edge["source_node_id"],
                    "target_node_id": edge["target_node_id"],
                }
                for edge in context["edges"]
            ],
        )
    return context


async def _upsert_node(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    label: str,
    kind: str,
    aliases: list[str],
    summary: str | None,
    confidence: float,
    weight: float,
    status: str,
    seen_at,
) -> tuple[MemoryNode, bool]:
    kind = _clean_kind(kind)
    if kind == "lexicon":
        confidence = min(confidence, 0.75)
        weight = min(weight, 0.70)
        status = _lexicon_status(status)
    normalized = _normalize_label(label)
    existing = list(
        await repo.get_memory_nodes_by_normalized_labels(session, user_id=user_id, labels=[normalized])
    )
    if not existing:
        node = await repo.add_memory_node(
            session,
            user_id=user_id,
            label=label,
            normalized_label=normalized,
            kind=kind,
            aliases=aliases,
            summary=summary,
            confidence=_decimal(confidence),
            weight=_decimal(weight),
            status=_status(status),
            last_seen_at=seen_at,
            meta={"source": "personal_lexicon" if kind == "lexicon" else "memory_graph_extraction"},
        )
        return node, True
    node = existing[0]
    node.aliases = _merge_aliases(node.aliases or [], aliases)
    node.summary = _prefer_text(node.summary, summary)
    if kind == "lexicon" and node.kind in {"concept", "term"}:
        node.kind = "lexicon"
    else:
        node.kind = node.kind or kind
    if node.kind == "lexicon":
        confidence = min(confidence, 0.75)
        weight = min(weight, 0.70)
        status = _lexicon_status(status)
    node.confidence = _max_decimal(node.confidence, confidence)
    node.weight = _max_decimal(node.weight, weight)
    node.status = _stronger_status(node.status, status)
    node.last_seen_at = seen_at
    return node, False


def _record_personal_lexicon_evidence(node: MemoryNode, *, seen_at: datetime) -> None:
    if node.kind != "lexicon":
        return
    previous = node.meta or {}
    lexicon = dict(previous.get("personal_lexicon") or {})
    lexicon["evidence_count"] = int(lexicon.get("evidence_count") or 0) + 1
    lexicon["last_evidence_at"] = seen_at.isoformat()
    lexicon["conditional"] = True
    node.meta = {
        **previous,
        "source": "personal_lexicon",
        "personal_lexicon": lexicon,
    }


async def _upsert_edge(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    source: MemoryNode,
    target: MemoryNode,
    relation_label: str,
    summary: str | None,
    confidence: float,
    weight: float,
    status: str,
    seen_at,
) -> tuple[MemoryEdge, bool]:
    edge = await repo.get_memory_edge(
        session,
        user_id=user_id,
        source_node_id=source.id,
        target_node_id=target.id,
        relation_label=relation_label,
    )
    if edge is None:
        edge = await repo.add_memory_edge(
            session,
            user_id=user_id,
            source_node_id=source.id,
            target_node_id=target.id,
            relation_label=relation_label,
            summary=summary,
            confidence=_decimal(confidence),
            weight=_decimal(weight),
            status=_status(status),
            evidence_count=1,
            last_seen_at=seen_at,
            meta={"source": "memory_graph_extraction"},
        )
        return edge, True
    edge.summary = _prefer_text(edge.summary, summary)
    edge.confidence = _max_decimal(edge.confidence, confidence)
    edge.weight = _max_decimal(edge.weight, weight)
    edge.status = _stronger_status(edge.status, status)
    edge.evidence_count += 1
    edge.last_seen_at = seen_at
    return edge, False


def _clean_label(value: str) -> str:
    return _compact(value)[:120]


def _record_situation_evidence(node: MemoryNode, *, seen_at) -> None:
    if node.kind != "situation":
        return
    meta = dict(node.meta or {})
    situation = dict(meta.get("situation") or {})
    situation["evidence_count"] = int(situation.get("evidence_count") or 0) + 1
    situation["last_supported_at"] = seen_at.isoformat() if seen_at else None
    meta["situation"] = situation
    node.meta = meta
    node.status = "hypothesis" if node.status == "candidate" else node.status
    node.confidence = min(node.confidence or Decimal("0"), Decimal("0.800"))
    node.weight = min(node.weight or Decimal("0"), Decimal("0.850"))


def _clean_kind(value: str) -> str:
    text = re.sub(r"[^a-zA-Zа-яА-ЯіїєґІЇЄҐ0-9_-]+", "_", _compact(value).lower())
    return (text or "concept")[:64]


def _clean_relation(value: str) -> str:
    text = re.sub(r"[^a-zA-Zа-яА-ЯіїєґІЇЄҐ0-9_-]+", "_", _compact(value).lower())
    return text[:128]


def _clean_aliases(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = _clean_label(value)
        if text and text not in result:
            result.append(text)
    return result[:12]


def _normalize_label(value: str) -> str:
    return _compact(value).lower()[:255]


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _decimal(value: float | None) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(max(0.0, min(1.0, float(value))))).quantize(Decimal("0.001"))


def _max_decimal(current: Decimal | None, candidate: float) -> Decimal:
    next_value = _decimal(candidate) or Decimal("0.000")
    if current is None:
        return next_value
    return max(current, next_value)


def _status(value: str) -> str:
    return value if value in {"candidate", "hypothesis", "confirmed"} else "hypothesis"


def _lexicon_status(value: str) -> str:
    return "hypothesis" if _status(value) == "confirmed" else _status(value)


def _stronger_status(current: str, candidate: str) -> str:
    rank = {"rejected": 0, "contradicted": 0, "stale": 1, "candidate": 2, "hypothesis": 3, "confirmed": 4}
    return current if rank.get(current, 2) >= rank.get(_status(candidate), 2) else _status(candidate)


def _merge_aliases(current: list[str], new: list[str]) -> list[str]:
    result: list[str] = []
    for value in [*current, *new]:
        text = _clean_label(value)
        if text and text not in result:
            result.append(text)
    return result[:20]


def _prefer_text(current: str | None, new: str | None) -> str | None:
    if not current:
        return new
    if new and len(new) > len(current):
        return new
    return current


def _potential_duplicate_nodes(nodes: list[MemoryNode], *, limit: int = 40) -> list[tuple[MemoryNode, MemoryNode, float, str]]:
    pairs: list[tuple[MemoryNode, MemoryNode, float, str]] = []
    active = [node for node in nodes if node.status in {"candidate", "hypothesis", "confirmed"}]
    for index, left in enumerate(active):
        for right in active[index + 1 :]:
            if left.kind != right.kind and {left.kind, right.kind} != {"concept"}:
                continue
            score, reason = _node_duplicate_score(left, right)
            if score < 0.9:
                continue
            keeper, duplicate = _duplicate_keeper(left, right)
            if keeper.id != duplicate.id:
                pairs.append((keeper, duplicate, score, reason))
    pairs.sort(key=lambda item: item[2], reverse=True)
    seen_duplicates: set[uuid.UUID] = set()
    result: list[tuple[MemoryNode, MemoryNode, float, str]] = []
    for keeper, duplicate, score, reason in pairs:
        if duplicate.id in seen_duplicates:
            continue
        seen_duplicates.add(duplicate.id)
        result.append((keeper, duplicate, score, reason))
        if len(result) >= limit:
            break
    return result


def _node_duplicate_score(left: MemoryNode, right: MemoryNode) -> tuple[float, str]:
    left_labels = _normalized_node_labels(left)
    right_labels = _normalized_node_labels(right)
    if not left_labels or not right_labels:
        return 0.0, "empty"
    if set(left_labels) & set(right_labels):
        return 1.0, "same_label_or_alias"
    best = 0.0
    best_reason = "different"
    for left_label in left_labels:
        for right_label in right_labels:
            score, reason = _label_similarity(left_label, right_label)
            if score > best:
                best = score
                best_reason = reason
    return best, best_reason


def _normalized_node_labels(node: MemoryNode) -> list[str]:
    labels = [node.label, *(node.aliases or [])]
    result: list[str] = []
    for label in labels:
        normalized = _normalize_label(label)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _label_similarity(left: str, right: str) -> tuple[float, str]:
    if left == right:
        return 1.0, "same_label"
    shorter, longer = sorted([left, right], key=len)
    if len(shorter) >= 5 and shorter in longer:
        coverage = len(shorter) / max(1, len(longer))
        return max(0.9, min(0.98, 0.82 + coverage * 0.18)), "substring"
    token_score = _token_jaccard(left, right)
    char_score = _bigram_dice(left, right)
    if token_score >= char_score:
        return token_score, "token_overlap"
    return char_score, "char_overlap"


def _query_label_match_score(label: str, query: str) -> float:
    normalized = _normalize_label(label)
    if not normalized or not query:
        return 0.0
    if normalized in query:
        return float(len(normalized))

    label_tokens = _search_tokens(normalized)
    query_tokens = _search_tokens(query)
    if not label_tokens or not query_tokens:
        return 0.0
    best = 0.0
    for label_token in label_tokens:
        for query_token in query_tokens:
            score = _token_query_match_score(label_token, query_token)
            if score > best:
                best = score
    return best


def _token_query_match_score(label_token: str, query_token: str) -> float:
    if label_token == query_token:
        return float(len(label_token))
    if len(label_token) < 4 or len(query_token) < 4:
        return 0.0
    shorter, longer = sorted([label_token, query_token], key=len)
    if len(shorter) >= 4 and longer.startswith(shorter):
        return float(len(shorter)) * 0.9
    distance = _bounded_edit_distance(label_token, query_token, max_distance=2)
    if distance <= 2:
        similarity = 1 - distance / max(len(label_token), len(query_token))
        if similarity >= 0.62 and label_token[:3] == query_token[:3]:
            return max(len(label_token), len(query_token)) * similarity
    return 0.0


def _bounded_edit_distance(left: str, right: str, *, max_distance: int) -> int:
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        row_min = current[0]
        for right_index, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + cost,
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def _search_tokens(value: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-ЯіїєґІЇЄҐ0-9]+", value.lower())


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-zA-Zа-яА-ЯіїєґІЇЄҐ0-9]+", left))
    right_tokens = set(re.findall(r"[a-zA-Zа-яА-ЯіїєґІЇЄҐ0-9]+", right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _bigram_dice(left: str, right: str) -> float:
    left_bigrams = _bigrams(left)
    right_bigrams = _bigrams(right)
    if not left_bigrams or not right_bigrams:
        return 0.0
    return (2 * len(left_bigrams & right_bigrams)) / (len(left_bigrams) + len(right_bigrams))


def _bigrams(value: str) -> set[str]:
    compact = re.sub(r"\s+", "", value)
    if len(compact) < 2:
        return set()
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def _duplicate_keeper(left: MemoryNode, right: MemoryNode) -> tuple[MemoryNode, MemoryNode]:
    ranked = sorted(
        [left, right],
        key=lambda node: (
            _status_rank(node.status),
            float(node.weight or 0),
            float(node.confidence or 0),
            len(node.summary or ""),
            getattr(node, "last_seen_at", None) or datetime.min.replace(tzinfo=UTC),
        ),
        reverse=True,
    )
    return ranked[0], ranked[1]


def _mark_possible_duplicate(
    node: MemoryNode,
    *,
    other: MemoryNode,
    score: float,
    reason: str,
    checked_at: datetime,
) -> bool:
    meta = node.meta or {}
    existing = list(meta.get("possible_duplicates") or [])
    other_id = str(other.id)
    candidate = {
        "node_id": other_id,
        "label": other.label,
        "score": round(score, 3),
        "reason": reason,
        "checked_at": checked_at.isoformat(),
    }
    changed = False
    next_items: list[dict[str, Any]] = []
    replaced = False
    for item in existing:
        if not isinstance(item, dict):
            continue
        if item.get("node_id") == other_id:
            next_items.append(candidate)
            replaced = True
            changed = item != candidate
        else:
            next_items.append(item)
    if not replaced:
        next_items.append(candidate)
        changed = True
    next_items.sort(key=lambda item: float(item.get("score") or 0), reverse=True)
    next_items = next_items[:5]
    if changed:
        node.meta = {**meta, "possible_duplicates": next_items, "maintenance": "local_duplicate_candidates"}
    return changed


async def _review_evidence_by_node(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    nodes: list[MemoryNode],
) -> dict[str, list[dict[str, Any]]]:
    evidence = await repo.list_memory_evidence_for_nodes(
        session,
        user_id=user_id,
        node_ids=[node.id for node in nodes],
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for item in evidence:
        if item.node_id is None:
            continue
        node_id = str(item.node_id)
        bucket = result.setdefault(node_id, [])
        if len(bucket) >= 3:
            continue
        bucket.append(
            {
                "target_type": item.target_type,
                "target_id": str(item.target_id),
                "text": _compact(item.evidence_text)[:320],
                "confidence": float(item.confidence or 0),
            }
        )
    return result


async def _embedding_duplicate_pairs(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    nodes: list[MemoryNode],
    evidence_by_node: dict[str, list[dict[str, Any]]],
    limit: int,
    only_node_ids: set[uuid.UUID] | None,
) -> list[tuple[str, MemoryNode, MemoryNode, dict[str, Any]]]:
    """Use existing entry vectors as a candidate signal, never as a merge decision."""
    entry_ids = {
        uuid.UUID(str(item["target_id"]))
        for evidence in evidence_by_node.values()
        for item in evidence
        if item.get("target_type") == "entry" and item.get("target_id")
    }
    if len(entry_ids) < 2:
        return []
    records = await repo.list_embeddings_for_targets(
        session,
        user_id=user_id,
        target_type="entry",
        target_ids=list(entry_ids),
    )
    latest_vectors: dict[str, list[float]] = {}
    for record in records:
        entry_id = str(record.target_id)
        if entry_id in latest_vectors:
            continue
        vector = [float(value) for value in (record.embedding or [])]
        if vector:
            latest_vectors[entry_id] = vector

    node_vectors: dict[str, list[list[float]]] = {}
    for node in nodes:
        vectors = [
            latest_vectors[str(item["target_id"])]
            for item in evidence_by_node.get(str(node.id), [])
            if item.get("target_type") == "entry" and str(item.get("target_id")) in latest_vectors
        ]
        if vectors:
            node_vectors[str(node.id)] = vectors[:3]

    candidates: list[tuple[float, str, MemoryNode, MemoryNode, dict[str, Any]]] = []
    for index, left in enumerate(nodes):
        left_vectors = node_vectors.get(str(left.id))
        if not left_vectors:
            continue
        for right in nodes[index + 1 :]:
            right_vectors = node_vectors.get(str(right.id))
            if not right_vectors or not _compatible_duplicate_kinds(left, right):
                continue
            if only_node_ids is not None and left.id not in only_node_ids and right.id not in only_node_ids:
                continue
            if _already_reviewed_separate(left, right):
                continue
            similarity = _max_cosine_similarity(left_vectors, right_vectors)
            if similarity < 0.88:
                continue
            pair_key = tuple(sorted([str(left.id), str(right.id)]))
            pair_id = f"{pair_key[0]}:{pair_key[1]}"
            candidates.append(
                (
                    similarity,
                    pair_id,
                    left,
                    right,
                    {
                        "score": round(similarity, 4),
                        "reason": "embedding_evidence_similarity",
                        "signal_sources": ["embedding"],
                    },
                )
            )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [(pair_id, left, right, candidate) for _, pair_id, left, right, candidate in candidates[:limit]]


def _compatible_duplicate_kinds(left: MemoryNode, right: MemoryNode) -> bool:
    return left.kind == right.kind or "concept" in {left.kind, right.kind}


def _already_reviewed_separate(left: MemoryNode, right: MemoryNode) -> bool:
    target_ids = {str(left.id), str(right.id)}
    for node in (left, right):
        reviewed = list((node.meta or {}).get("reviewed_separate_from") or [])
        if any(str(item.get("node_id") or "") in target_ids - {str(node.id)} for item in reviewed if isinstance(item, dict)):
            return True
    return False


def _max_cosine_similarity(left_vectors: list[list[float]], right_vectors: list[list[float]]) -> float:
    best = 0.0
    for left in left_vectors:
        left_norm = sum(value * value for value in left) ** 0.5
        if left_norm == 0:
            continue
        for right in right_vectors:
            if len(left) != len(right):
                continue
            right_norm = sum(value * value for value in right) ** 0.5
            if right_norm == 0:
                continue
            similarity = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
            best = max(best, similarity)
    return best


def _combine_review_pairs(
    pairs: list[tuple[str, MemoryNode, MemoryNode, dict[str, Any]]],
    *,
    limit: int,
) -> list[tuple[str, MemoryNode, MemoryNode, dict[str, Any]]]:
    combined: dict[str, tuple[str, MemoryNode, MemoryNode, dict[str, Any]]] = {}
    for pair_id, left, right, candidate in pairs:
        existing = combined.get(pair_id)
        if existing is None:
            combined[pair_id] = (pair_id, left, right, dict(candidate))
            continue
        _, _, _, previous = existing
        sources = list(dict.fromkeys([*(previous.get("signal_sources") or []), *(candidate.get("signal_sources") or [])]))
        if previous.get("reason"):
            sources.append(str(previous["reason"]))
        if candidate.get("reason"):
            sources.append(str(candidate["reason"]))
        merged = {
            **previous,
            "score": max(float(previous.get("score") or 0), float(candidate.get("score") or 0)),
            "reason": " + ".join(dict.fromkeys(sources))[:180],
            "signal_sources": list(dict.fromkeys(sources))[:4],
        }
        combined[pair_id] = (pair_id, left, right, merged)
    ranked = sorted(combined.values(), key=lambda item: float(item[3].get("score") or 0), reverse=True)
    return ranked[:limit]


def _review_candidate_pairs(
    nodes: list[MemoryNode],
    *,
    limit: int,
    only_node_ids: set[uuid.UUID] | None = None,
) -> list[tuple[str, MemoryNode, MemoryNode, dict[str, Any]]]:
    by_id = {str(node.id): node for node in nodes}
    pairs: list[tuple[float, str, MemoryNode, MemoryNode, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    for node in nodes:
        for candidate in node.meta.get("possible_duplicates", []) if isinstance(node.meta, dict) else []:
            if not isinstance(candidate, dict):
                continue
            other = by_id.get(str(candidate.get("node_id") or ""))
            if other is None or other.id == node.id:
                continue
            if only_node_ids is not None and node.id not in only_node_ids and other.id not in only_node_ids:
                continue
            pair_key = tuple(sorted([str(node.id), str(other.id)]))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            score = float(candidate.get("score") or 0)
            pair_id = f"{pair_key[0]}:{pair_key[1]}"
            pairs.append((score, pair_id, node, other, candidate))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return [(pair_id, left, right, candidate) for _, pair_id, left, right, candidate in pairs[:limit]]


def _review_pair_payload(
    pair: tuple[str, MemoryNode, MemoryNode, dict[str, Any]],
    *,
    evidence_by_node: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    pair_id, left, right, candidate = pair
    return {
        "pair_id": pair_id,
        "local_candidate": {
            "score": candidate.get("score"),
            "reason": candidate.get("reason"),
        },
        "nodes": [
            _review_node_payload(left, evidence=(evidence_by_node or {}).get(str(left.id), [])),
            _review_node_payload(right, evidence=(evidence_by_node or {}).get(str(right.id), [])),
        ],
    }


def _review_node_payload(node: MemoryNode, *, evidence: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": str(node.id),
        "label": node.label,
        "aliases": list(node.aliases or [])[:8],
        "kind": node.kind,
        "summary": node.summary,
        "status": node.status,
        "weight": float(node.weight or 0),
        "confidence": float(node.confidence or 0),
        "possible_duplicates": [
            {
                "node_id": item.get("node_id"),
                "label": item.get("label"),
                "score": item.get("score"),
                "reason": item.get("reason"),
            }
            for item in list((node.meta or {}).get("possible_duplicates") or [])[:3]
            if isinstance(item, dict)
        ],
        "evidence": list(evidence or [])[:3],
    }


def _apply_memory_graph_review_decision(
    left: MemoryNode,
    right: MemoryNode,
    decision: MemoryGraphReviewDecision,
    *,
    nodes_by_id: dict[str, MemoryNode],
    reviewed_at: datetime,
) -> dict[str, int]:
    result = {
        "aliases_added": 0,
        "nodes_staled": 0,
        "pairs_marked_separate": 0,
        "pairs_needing_confirmation": 0,
    }
    if decision.decision == "separate":
        _mark_review_separate(left, other=right, decision=decision, reviewed_at=reviewed_at)
        _mark_review_separate(right, other=left, decision=decision, reviewed_at=reviewed_at)
        result["pairs_marked_separate"] = 1
        return result
    if decision.decision == "unsure" or decision.confidence < 0.82 or decision.needs_user_confirmation:
        _mark_review_needs_confirmation(left, other=right, decision=decision, reviewed_at=reviewed_at)
        _mark_review_needs_confirmation(right, other=left, decision=decision, reviewed_at=reviewed_at)
        result["pairs_needing_confirmation"] = 1
        return result
    if decision.decision not in {"same", "alias"}:
        return result

    canonical = nodes_by_id.get(decision.canonical_node_id or "") or _duplicate_keeper(left, right)[0]
    duplicate = right if canonical.id == left.id else left
    if canonical.status == "confirmed" and duplicate.status == "confirmed":
        _mark_review_needs_confirmation(left, other=right, decision=decision, reviewed_at=reviewed_at)
        _mark_review_needs_confirmation(right, other=left, decision=decision, reviewed_at=reviewed_at)
        result["pairs_needing_confirmation"] = 1
        return result

    aliases_before = set(canonical.aliases or [])
    aliases_to_add = [duplicate.label, *(duplicate.aliases or [])]
    if decision.alias_text:
        aliases_to_add.append(decision.alias_text)
    canonical.aliases = _merge_aliases(canonical.aliases or [], aliases_to_add)
    result["aliases_added"] = max(0, len(set(canonical.aliases or []) - aliases_before))
    canonical.meta = {
        **(canonical.meta or {}),
        "graph_review": {
            "last_decision": decision.decision,
            "reviewed_with": str(duplicate.id),
            "confidence": round(decision.confidence, 3),
            "reviewed_at": reviewed_at.isoformat(),
            "reason": decision.reason,
        },
    }

    if duplicate.status != "confirmed":
        duplicate.status = "stale"
        duplicate.weight = min(duplicate.weight or Decimal("0.250"), Decimal("0.250"))
        duplicate.meta = {
            **(duplicate.meta or {}),
            "stale_reason": "ai_review_duplicate",
            "duplicate_of": str(canonical.id),
            "duplicate_of_label": canonical.label,
            "duplicate_review_confidence": round(decision.confidence, 3),
            "duplicate_reviewed_at": reviewed_at.isoformat(),
            "duplicate_review_reason": decision.reason,
        }
        result["nodes_staled"] = 1
    else:
        _mark_review_needs_confirmation(duplicate, other=canonical, decision=decision, reviewed_at=reviewed_at)
        result["pairs_needing_confirmation"] = 1
    return result


def _confirmation_candidate_for_decision(
    left: MemoryNode,
    right: MemoryNode,
    decision: MemoryGraphReviewDecision,
    *,
    result: dict[str, int],
) -> MemoryGraphConfirmationCandidate | None:
    if not result["pairs_needing_confirmation"]:
        return None
    question = _compact(decision.confirmation_question or "")
    options = [
        {"label": _compact(option.label)[:80], "outcome": option.outcome}
        for option in decision.confirmation_options
        if _compact(option.label)
    ]
    if not question or len(options) < 2:
        return None
    return MemoryGraphConfirmationCandidate(
        left_node_id=left.id,
        right_node_id=right.id,
        question=question[:600],
        options=tuple(options[:4]),
        reason=_compact(decision.reason or "")[:240] or None,
    )


def _mark_review_separate(
    node: MemoryNode,
    *,
    other: MemoryNode,
    decision: MemoryGraphReviewDecision,
    reviewed_at: datetime,
) -> None:
    node.meta = {
        **(node.meta or {}),
        "reviewed_separate_from": _append_limited_meta_item(
            (node.meta or {}).get("reviewed_separate_from"),
            {
                "node_id": str(other.id),
                "label": other.label,
                "confidence": round(decision.confidence, 3),
                "reason": decision.reason,
                "reviewed_at": reviewed_at.isoformat(),
            },
        ),
        "possible_duplicates": _remove_possible_duplicate((node.meta or {}).get("possible_duplicates"), str(other.id)),
    }


def _mark_review_needs_confirmation(
    node: MemoryNode,
    *,
    other: MemoryNode,
    decision: MemoryGraphReviewDecision,
    reviewed_at: datetime,
) -> None:
    node.meta = {
        **(node.meta or {}),
        "duplicate_review_needs_confirmation": _append_limited_meta_item(
            (node.meta or {}).get("duplicate_review_needs_confirmation"),
            {
                "node_id": str(other.id),
                "label": other.label,
                "decision": decision.decision,
                "confidence": round(decision.confidence, 3),
                "reason": decision.reason,
                "reviewed_at": reviewed_at.isoformat(),
            },
        ),
    }


def _append_limited_meta_item(raw_items: Any, item: dict[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    items = [existing for existing in list(raw_items or []) if isinstance(existing, dict)]
    items = [existing for existing in items if existing.get("node_id") != item.get("node_id")]
    items.append(item)
    return items[-limit:]


def _remove_possible_duplicate(raw_items: Any, node_id: str) -> list[dict[str, Any]]:
    return [
        item
        for item in list(raw_items or [])
        if isinstance(item, dict) and str(item.get("node_id") or "") != node_id
    ][:5]


def _status_rank(status: str) -> int:
    return {"rejected": 0, "contradicted": 0, "stale": 1, "candidate": 2, "hypothesis": 3, "confirmed": 4}.get(status, 2)


def _node_payload(node: MemoryNode) -> dict[str, Any]:
    return {
        "id": str(node.id),
        "label": node.label,
        "kind": node.kind,
        "summary": node.summary,
        "confidence": float(node.confidence or 0),
        "weight": float(node.weight or 0),
        "status": node.status,
        "aliases": node.aliases,
    }


def _edge_payload(edge: MemoryEdge) -> dict[str, Any]:
    return {
        "id": str(edge.id),
        "source_node_id": str(edge.source_node_id),
        "target_node_id": str(edge.target_node_id),
        "relation_label": edge.relation_label,
        "summary": edge.summary,
        "confidence": float(edge.confidence or 0),
        "weight": float(edge.weight or 0),
        "status": edge.status,
        "evidence_count": edge.evidence_count,
    }
