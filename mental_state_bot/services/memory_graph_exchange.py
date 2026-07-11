from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.db import repositories as repo
from mental_state_bot.time_utils import utc_now

MEMORY_GRAPH_EXPORT_SCHEMA = "mental-state-bot.memory-graph.v1"
MAX_GRAPH_IMPORT_BYTES = 2_000_000
MAX_GRAPH_IMPORT_NODES = 1_200
MAX_GRAPH_IMPORT_EDGES = 3_000
MAX_GRAPH_IMPORT_EVIDENCE = 6_000
_STATUSES = {"candidate", "hypothesis", "confirmed", "stale", "contradicted", "rejected"}
_EVIDENCE_TARGET_TYPES = {"entry", "life_context"}


@dataclass(frozen=True)
class MemoryGraphImportPreview:
    nodes: int
    edges: int
    evidence: int
    skipped: int


async def export_memory_graph_json(session: AsyncSession, *, user_id: uuid.UUID) -> bytes:
    nodes = list(await repo.list_memory_nodes_for_export(session, user_id=user_id))
    edges = list(await repo.list_memory_edges_for_export(session, user_id=user_id))
    evidence = list(await repo.list_memory_evidence_for_export(session, user_id=user_id))
    payload = {
        "schema_version": MEMORY_GRAPH_EXPORT_SCHEMA,
        "exported_at": utc_now().isoformat(),
        "nodes": [
            {
                "key": str(node.id),
                "label": node.label,
                "kind": node.kind,
                "aliases": node.aliases or [],
                "summary": node.summary,
                "confidence": float(node.confidence or 0),
                "weight": float(node.weight or 0),
                "status": node.status,
                "last_seen_at": node.last_seen_at.isoformat() if node.last_seen_at else None,
            }
            for node in nodes
        ],
        "edges": [
            {
                "key": str(edge.id),
                "source_key": str(edge.source_node_id),
                "target_key": str(edge.target_node_id),
                "relation_label": edge.relation_label,
                "summary": edge.summary,
                "confidence": float(edge.confidence or 0),
                "weight": float(edge.weight or 0),
                "status": edge.status,
                "last_seen_at": edge.last_seen_at.isoformat() if edge.last_seen_at else None,
            }
            for edge in edges
        ],
        "evidence": [
            {
                "node_key": str(item.node_id) if item.node_id else None,
                "edge_key": str(item.edge_id) if item.edge_id else None,
                "target_type": item.target_type,
                "target_id": str(item.target_id),
                "evidence_text": item.evidence_text,
                "confidence": float(item.confidence or 0),
            }
            for item in evidence
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def parse_memory_graph_import(raw: bytes) -> tuple[dict[str, Any], MemoryGraphImportPreview]:
    if len(raw) > MAX_GRAPH_IMPORT_BYTES:
        raise ValueError("Файл завеликий для безпечного імпорту графа.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Не зміг прочитати JSON графа.") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != MEMORY_GRAPH_EXPORT_SCHEMA:
        raise ValueError("Це не JSON-експорт графа з цього бота або його схема не підтримується.")

    nodes, skipped_nodes = _sanitize_nodes(payload.get("nodes"))
    if not nodes:
        raise ValueError("У файлі немає жодного коректного вузла. Не замінюю поточний граф.")
    edges, skipped_edges = _sanitize_edges(payload.get("edges"), node_keys={item["key"] for item in nodes})
    evidence, skipped_evidence = _sanitize_evidence(
        payload.get("evidence"),
        node_keys={item["key"] for item in nodes},
        edge_keys={item["key"] for item in edges},
    )
    cleaned = {
        "schema_version": MEMORY_GRAPH_EXPORT_SCHEMA,
        "nodes": nodes,
        "edges": edges,
        "evidence": evidence,
    }
    return cleaned, MemoryGraphImportPreview(
        nodes=len(nodes),
        edges=len(edges),
        evidence=len(evidence),
        skipped=skipped_nodes + skipped_edges + skipped_evidence,
    )


async def replace_memory_graph_from_import(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    payload: dict[str, Any],
) -> MemoryGraphImportPreview:
    cleaned, preview = parse_memory_graph_import(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    await repo.delete_memory_graph(session, user_id=user_id)
    now = utc_now()
    nodes_by_key: dict[str, Any] = {}
    for item in cleaned["nodes"]:
        node = await repo.add_memory_node(
            session,
            user_id=user_id,
            label=item["label"],
            normalized_label=_normalized_label(item["label"]),
            kind=item["kind"],
            aliases=item["aliases"],
            summary=item.get("summary"),
            confidence=_decimal(item["confidence"]),
            weight=_decimal(item["weight"]),
            status=_import_status(
                item["status"],
                kind=item["kind"],
                allow_confirmed=item["kind"] == "life_context",
            ),
            last_seen_at=_parse_datetime(item.get("last_seen_at")) or now,
            meta=_import_node_meta(item["kind"]),
        )
        nodes_by_key[item["key"]] = node

    edges_by_key: dict[str, Any] = {}
    for item in cleaned["edges"]:
        edge = await repo.add_memory_edge(
            session,
            user_id=user_id,
            source_node_id=nodes_by_key[item["source_key"]].id,
            target_node_id=nodes_by_key[item["target_key"]].id,
            relation_label=item["relation_label"],
            summary=item.get("summary"),
            confidence=_decimal(item["confidence"]),
            weight=_decimal(item["weight"]),
            status=_import_status(item["status"], allow_confirmed=False),
            evidence_count=0,
            last_seen_at=_parse_datetime(item.get("last_seen_at")) or now,
            meta={"source": "manual_graph_import"},
        )
        edges_by_key[item["key"]] = edge

    for item in cleaned["evidence"]:
        node = nodes_by_key.get(item.get("node_key"))
        edge = edges_by_key.get(item.get("edge_key"))
        if node is None and edge is None:
            continue
        await repo.add_memory_evidence(
            session,
            user_id=user_id,
            node_id=node.id if node else None,
            edge_id=edge.id if edge else None,
            target_type=item["target_type"],
            target_id=uuid.UUID(item["target_id"]),
            evidence_text=item["evidence_text"],
            confidence=_decimal(item["confidence"]),
            meta={"source": "manual_graph_import"},
        )
        if node is not None and node.kind == "lexicon":
            lexicon = dict((node.meta or {}).get("personal_lexicon") or {})
            lexicon["evidence_count"] = int(lexicon.get("evidence_count") or 0) + 1
            node.meta = {**(node.meta or {}), "personal_lexicon": lexicon}
        if edge is not None:
            edge.evidence_count += 1
    return preview


def _sanitize_nodes(value: object) -> tuple[list[dict[str, Any]], int]:
    cleaned: list[dict[str, Any]] = []
    skipped = 0
    seen_keys: set[str] = set()
    seen_labels: set[str] = set()
    for raw in list(value or [])[:MAX_GRAPH_IMPORT_NODES]:
        if not isinstance(raw, dict):
            skipped += 1
            continue
        key = _short_text(raw.get("key"), limit=80)
        label = _short_text(raw.get("label"), limit=120)
        normalized = _normalized_label(label)
        if not key or not label or key in seen_keys or normalized in seen_labels:
            skipped += 1
            continue
        kind = _safe_kind(raw.get("kind"))
        cleaned.append(
            {
                "key": key,
                "label": label,
                "kind": kind,
                "aliases": _safe_aliases(raw.get("aliases")),
                "summary": _nullable_text(raw.get("summary"), limit=800),
                "confidence": _float(raw.get("confidence")),
                "weight": _float(raw.get("weight")),
                "status": _import_status(
                    str(raw.get("status") or "hypothesis"),
                    kind=kind,
                    allow_confirmed=kind == "life_context",
                ),
                "last_seen_at": _nullable_text(raw.get("last_seen_at"), limit=64),
            }
        )
        seen_keys.add(key)
        seen_labels.add(normalized)
    return cleaned, skipped


def _sanitize_edges(value: object, *, node_keys: set[str]) -> tuple[list[dict[str, Any]], int]:
    cleaned: list[dict[str, Any]] = []
    skipped = 0
    seen: set[tuple[str, str, str]] = set()
    seen_keys: set[str] = set()
    for raw in list(value or [])[:MAX_GRAPH_IMPORT_EDGES]:
        if not isinstance(raw, dict):
            skipped += 1
            continue
        key = _short_text(raw.get("key"), limit=80)
        source_key = _short_text(raw.get("source_key"), limit=80)
        target_key = _short_text(raw.get("target_key"), limit=80)
        relation = _safe_relation(raw.get("relation_label"))
        identity = (source_key, target_key, relation)
        if (
            not key
            or source_key not in node_keys
            or target_key not in node_keys
            or source_key == target_key
            or not relation
            or identity in seen
            or key in seen_keys
        ):
            skipped += 1
            continue
        cleaned.append(
            {
                "key": key,
                "source_key": source_key,
                "target_key": target_key,
                "relation_label": relation,
                "summary": _nullable_text(raw.get("summary"), limit=800),
                "confidence": _float(raw.get("confidence")),
                "weight": _float(raw.get("weight")),
                "status": _import_status(str(raw.get("status") or "hypothesis"), allow_confirmed=False),
                "last_seen_at": _nullable_text(raw.get("last_seen_at"), limit=64),
            }
        )
        seen.add(identity)
        seen_keys.add(key)
    return cleaned, skipped


def _sanitize_evidence(
    value: object,
    *,
    node_keys: set[str],
    edge_keys: set[str],
) -> tuple[list[dict[str, Any]], int]:
    cleaned: list[dict[str, Any]] = []
    skipped = 0
    for raw in list(value or [])[:MAX_GRAPH_IMPORT_EVIDENCE]:
        if not isinstance(raw, dict):
            skipped += 1
            continue
        node_key = _short_text(raw.get("node_key"), limit=80) or None
        edge_key = _short_text(raw.get("edge_key"), limit=80) or None
        target_type = _short_text(raw.get("target_type"), limit=64)
        target_id = _short_text(raw.get("target_id"), limit=64)
        evidence_text = _short_text(raw.get("evidence_text"), limit=1200)
        if (
            (node_key is None and edge_key is None)
            or (node_key is not None and node_key not in node_keys)
            or (edge_key is not None and edge_key not in edge_keys)
            or target_type not in _EVIDENCE_TARGET_TYPES
            or not _is_uuid(target_id)
            or not evidence_text
        ):
            skipped += 1
            continue
        cleaned.append(
            {
                "node_key": node_key,
                "edge_key": edge_key,
                "target_type": target_type,
                "target_id": target_id,
                "evidence_text": evidence_text,
                "confidence": _float(raw.get("confidence")),
            }
        )
    return cleaned, skipped


def _short_text(value: object, *, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _nullable_text(value: object, *, limit: int) -> str | None:
    return _short_text(value, limit=limit) or None


def _normalized_label(value: str) -> str:
    return " ".join(value.lower().split())[:255]


def _safe_kind(value: object) -> str:
    raw = _short_text(value, limit=64).lower()
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in raw) or "concept"


def _safe_relation(value: object) -> str:
    raw = _short_text(value, limit=128).lower()
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in raw).strip("_")


def _safe_aliases(value: object) -> list[str]:
    aliases: list[str] = []
    for item in list(value or [])[:20] if isinstance(value, list) else []:
        alias = _short_text(item, limit=120)
        if alias and alias not in aliases:
            aliases.append(alias)
    return aliases


def _float(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _import_status(value: str, *, kind: str | None = None, allow_confirmed: bool = False) -> str:
    status = value if value in _STATUSES else "hypothesis"
    if status == "confirmed" and (kind == "lexicon" or not allow_confirmed):
        return "hypothesis"
    return status


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _decimal(value: float) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.001"))


def _import_node_meta(kind: str) -> dict[str, Any]:
    meta: dict[str, Any] = {"source": "manual_graph_import"}
    if kind == "lexicon":
        meta["personal_lexicon"] = {"conditional": True, "evidence_count": 0}
    return meta


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True
