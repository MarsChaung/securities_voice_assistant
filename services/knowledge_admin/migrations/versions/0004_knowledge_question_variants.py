"""Add governed question variants to knowledge items."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_knowledge_question_variants"
down_revision: str | None = "0003_shadow_review_results"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_question_variants",
        sa.Column("variant_id", sa.String(length=36), primary_key=True),
        sa.Column(
            "knowledge_id",
            sa.String(length=80),
            sa.ForeignKey("knowledge_items.knowledge_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("usage", sa.String(length=30), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "knowledge_id",
            "normalized_text",
            name="uq_knowledge_question_variant_normalized",
        ),
    )
    op.create_index(
        "ix_knowledge_question_variants_knowledge_id",
        "knowledge_question_variants",
        ["knowledge_id"],
    )
    op.create_index(
        "ix_knowledge_question_variants_usage",
        "knowledge_question_variants",
        ["usage"],
    )


def downgrade() -> None:
    op.drop_table("knowledge_question_variants")
