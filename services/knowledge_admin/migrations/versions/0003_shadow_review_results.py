"""Persist privacy-scoped Shadow generation results for human review."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_shadow_review_results"
down_revision: str | None = "0002_knowledge_item_versions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "shadow_review_results",
        sa.Column("shadow_id", sa.String(length=36), primary_key=True),
        sa.Column("result_key", sa.String(length=64), nullable=False),
        sa.Column("turn_id", sa.String(length=36), nullable=False),
        sa.Column(
            "knowledge_id",
            sa.String(length=80),
            sa.ForeignKey("knowledge_items.knowledge_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("knowledge_version", sa.String(length=40), nullable=False),
        sa.Column(
            "source_id",
            sa.String(length=80),
            sa.ForeignKey("knowledge_sources.source_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("standard_answer", sa.Text(), nullable=False),
        sa.Column("prohibited_extensions_json", sa.Text(), nullable=False),
        sa.Column("generated_answer", sa.Text()),
        sa.Column("generation_model_id", sa.String(length=200)),
        sa.Column("prompt_version", sa.String(length=100)),
        sa.Column("prompt_hash", sa.String(length=64)),
        sa.Column("generation_latency_ms", sa.Float()),
        sa.Column("output_guard_safe", sa.Boolean()),
        sa.Column("fallback_reason", sa.String(length=200), nullable=False),
        sa.Column("review_status", sa.String(length=30), nullable=False),
        sa.Column("review_label", sa.String(length=40)),
        sa.Column("reviewer_id", sa.String(length=200)),
        sa.Column("reviewer_note", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True)),
        sa.Column("row_version", sa.Integer(), nullable=False),
    )
    op.create_index(
        "ix_shadow_review_results_result_key",
        "shadow_review_results",
        ["result_key"],
        unique=True,
    )
    op.create_index("ix_shadow_review_results_turn_id", "shadow_review_results", ["turn_id"])
    op.create_index(
        "ix_shadow_review_results_knowledge_id",
        "shadow_review_results",
        ["knowledge_id"],
    )
    op.create_index(
        "ix_shadow_review_results_review_status",
        "shadow_review_results",
        ["review_status"],
    )
    op.create_index(
        "ix_shadow_review_results_created_at",
        "shadow_review_results",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_table("shadow_review_results")
