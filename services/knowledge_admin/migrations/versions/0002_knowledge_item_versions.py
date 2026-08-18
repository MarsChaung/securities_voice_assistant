"""Preserve immutable published knowledge versions."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_knowledge_item_versions"
down_revision: str | None = "0001_knowledge_governance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "knowledge_item_versions",
        sa.Column("version_id", sa.String(length=36), primary_key=True),
        sa.Column(
            "knowledge_id",
            sa.String(length=80),
            sa.ForeignKey("knowledge_items.knowledge_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("version", sa.String(length=40), nullable=False),
        sa.Column("item_snapshot", json_type, nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("archived_by", sa.String(length=200), nullable=False),
        sa.UniqueConstraint(
            "knowledge_id",
            "version",
            name="uq_knowledge_item_version",
        ),
    )
    op.create_index(
        "ix_knowledge_item_versions_knowledge_id",
        "knowledge_item_versions",
        ["knowledge_id"],
    )
    op.create_index(
        "ix_knowledge_item_versions_archived_at",
        "knowledge_item_versions",
        ["archived_at"],
    )


def downgrade() -> None:
    op.drop_table("knowledge_item_versions")
