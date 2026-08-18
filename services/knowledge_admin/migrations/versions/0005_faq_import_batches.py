"""Add governed FAQ import preview batches."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_faq_import_batches"
down_revision: str | None = "0004_knowledge_question_variants"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "faq_import_batches",
        sa.Column("batch_id", sa.String(length=36), primary_key=True),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("file_sha256", sa.String(length=64), nullable=False),
        sa.Column("dataset_title", sa.String(length=200), nullable=False),
        sa.Column("publisher", sa.String(length=200), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("source_id", sa.String(length=80), nullable=False),
        sa.Column("uploaded_by", sa.String(length=200), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("sheet_name", sa.String(length=200), nullable=False),
        sa.Column("rows", sa.JSON(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("valid_row_count", sa.Integer(), nullable=False),
        sa.Column("imported_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("row_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "file_sha256",
            "source_url",
            name="uq_faq_import_batch_file_source",
        ),
    )
    op.create_index(
        "ix_faq_import_batches_file_sha256",
        "faq_import_batches",
        ["file_sha256"],
    )
    op.create_index(
        "ix_faq_import_batches_status",
        "faq_import_batches",
        ["status"],
    )
    op.create_index(
        "ix_faq_import_batches_created_at",
        "faq_import_batches",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_table("faq_import_batches")
