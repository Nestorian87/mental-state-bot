"""add memory graph tables

Revision ID: 20260709_0002
Revises: 20260629_0001
Create Date: 2026-07-09
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260709_0002"
down_revision = "20260629_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_nodes",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("normalized_label", sa.String(length=255), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("aliases", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("weight", sa.Numeric(precision=6, scale=3), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "normalized_label", name="uq_memory_nodes_user_label"),
    )
    op.create_index("ix_memory_nodes_user_kind", "memory_nodes", ["user_id", "kind"])
    op.create_index("ix_memory_nodes_user_status", "memory_nodes", ["user_id", "status"])

    op.create_table(
        "memory_edges",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_node_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target_node_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("relation_label", sa.String(length=128), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("weight", sa.Numeric(precision=6, scale=3), nullable=True),
        sa.Column("confidence", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("evidence_count", sa.Integer(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["source_node_id"], ["memory_nodes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_node_id"], ["memory_nodes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "source_node_id",
            "target_node_id",
            "relation_label",
            name="uq_memory_edges_user_relation",
        ),
    )
    op.create_index("ix_memory_edges_source", "memory_edges", ["source_node_id"])
    op.create_index("ix_memory_edges_target", "memory_edges", ["target_node_id"])
    op.create_index("ix_memory_edges_user_status", "memory_edges", ["user_id", "status"])

    op.create_table(
        "memory_evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("node_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("edge_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("target_type", sa.String(length=64), nullable=False),
        sa.Column("target_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("evidence_text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["edge_id"], ["memory_edges.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["node_id"], ["memory_nodes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_evidence_node", "memory_evidence", ["node_id"])
    op.create_index("ix_memory_evidence_edge", "memory_evidence", ["edge_id"])
    op.create_index("ix_memory_evidence_target", "memory_evidence", ["target_type", "target_id"])


def downgrade() -> None:
    op.drop_index("ix_memory_evidence_target", table_name="memory_evidence")
    op.drop_index("ix_memory_evidence_edge", table_name="memory_evidence")
    op.drop_index("ix_memory_evidence_node", table_name="memory_evidence")
    op.drop_table("memory_evidence")
    op.drop_index("ix_memory_edges_user_status", table_name="memory_edges")
    op.drop_index("ix_memory_edges_target", table_name="memory_edges")
    op.drop_index("ix_memory_edges_source", table_name="memory_edges")
    op.drop_table("memory_edges")
    op.drop_index("ix_memory_nodes_user_status", table_name="memory_nodes")
    op.drop_index("ix_memory_nodes_user_kind", table_name="memory_nodes")
    op.drop_table("memory_nodes")
