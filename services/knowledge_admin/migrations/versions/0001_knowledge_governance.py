"""Create knowledge governance tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_knowledge_governance"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "knowledge_sources",
        sa.Column("source_id", sa.String(length=80), primary_key=True),
        sa.Column("supplied_url", sa.Text(), nullable=False),
        sa.Column("canonical_url", sa.Text(), nullable=False, unique=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("publisher", sa.String(length=200), nullable=False),
        sa.Column("source_type", sa.String(length=40), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("topics", json_type, nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("notes", sa.Text()),
    )
    op.create_index("ix_knowledge_sources_status", "knowledge_sources", ["status"])

    op.create_table(
        "knowledge_items",
        sa.Column("knowledge_id", sa.String(length=80), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("standard_answer", sa.Text(), nullable=False),
        sa.Column(
            "source_id",
            sa.String(length=80),
            sa.ForeignKey("knowledge_sources.source_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("source_locator", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=40), nullable=False),
        sa.Column("products", json_type, nullable=False),
        sa.Column("platforms", json_type, nullable=False),
        sa.Column("app_versions", json_type, nullable=False),
        sa.Column("effective_at", sa.DateTime(timezone=True)),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("review_at", sa.DateTime(timezone=True)),
        sa.Column("owner_unit", sa.String(length=200)),
        sa.Column("author", sa.String(length=200), nullable=False),
        sa.Column("reviewer", sa.String(length=200)),
        sa.Column("approver", sa.String(length=200)),
        sa.Column("approved_at", sa.DateTime(timezone=True)),
        sa.Column("version", sa.String(length=40), nullable=False),
        sa.Column("change_summary", sa.Text(), nullable=False),
        sa.Column("previous_version", sa.String(length=40)),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("public_answer_allowed", sa.Boolean(), nullable=False),
        sa.Column("allowed_intents", json_type, nullable=False),
        sa.Column("prohibited_extensions", json_type, nullable=False),
        sa.Column("row_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_knowledge_items_source_id", "knowledge_items", ["source_id"])
    op.create_index("ix_knowledge_items_status", "knowledge_items", ["status"])

    op.create_table(
        "knowledge_governance_events",
        sa.Column("event_id", sa.String(length=36), primary_key=True),
        sa.Column(
            "knowledge_id",
            sa.String(length=80),
            sa.ForeignKey("knowledge_items.knowledge_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("action", sa.String(length=40), nullable=False),
        sa.Column("from_status", sa.String(length=30), nullable=False),
        sa.Column("to_status", sa.String(length=30), nullable=False),
        sa.Column("actor_id", sa.String(length=200), nullable=False),
        sa.Column("reason", sa.Text()),
        sa.Column("row_version", sa.Integer(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_knowledge_governance_events_knowledge_id",
        "knowledge_governance_events",
        ["knowledge_id"],
    )
    op.create_index(
        "ix_knowledge_governance_events_occurred_at",
        "knowledge_governance_events",
        ["occurred_at"],
    )


def downgrade() -> None:
    op.drop_table("knowledge_governance_events")
    op.drop_table("knowledge_items")
    op.drop_table("knowledge_sources")
