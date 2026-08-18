"""Allow local import sources without formal URLs."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_local_import_sources"
down_revision: str | None = "0005_faq_import_batches"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("knowledge_sources") as batch:
        batch.alter_column("supplied_url", existing_type=sa.Text(), nullable=True)
        batch.alter_column("canonical_url", existing_type=sa.Text(), nullable=True)
    with op.batch_alter_table("knowledge_items") as batch:
        batch.alter_column("source_uri", existing_type=sa.Text(), nullable=True)
    with op.batch_alter_table("faq_import_batches") as batch:
        batch.add_column(
            sa.Column(
                "source_type",
                sa.String(length=40),
                server_default="approved_internal_faq",
                nullable=False,
            )
        )
        batch.alter_column("source_url", existing_type=sa.Text(), nullable=True)
        batch.create_unique_constraint(
            "uq_faq_import_batch_file_source_id",
            ["file_sha256", "source_id"],
        )

    op.execute(
        """
        UPDATE knowledge_items
        SET source_type = 'local_import', source_uri = NULL
        WHERE source_id IN (
            SELECT source_id
            FROM knowledge_sources
            WHERE canonical_url LIKE 'https://example.invalid/local-test/%'
        )
        """
    )
    op.execute(
        """
        UPDATE faq_import_batches
        SET source_type = 'local_import', source_url = NULL
        WHERE source_id IN (
            SELECT source_id
            FROM knowledge_sources
            WHERE canonical_url LIKE 'https://example.invalid/local-test/%'
        )
        """
    )
    op.execute(
        """
        UPDATE knowledge_sources
        SET source_type = 'local_import',
            supplied_url = NULL,
            canonical_url = NULL,
            status = 'active',
            notes = '本機匯入資料；以批次、檔案 SHA-256 與 Excel 列號追溯。'
        WHERE canonical_url LIKE 'https://example.invalid/local-test/%'
        """
    )

    with op.batch_alter_table("faq_import_batches") as batch:
        batch.alter_column(
            "source_type",
            existing_type=sa.String(length=40),
            server_default=None,
        )


def downgrade() -> None:
    op.execute(
        """
        UPDATE knowledge_sources
        SET supplied_url = 'https://example.invalid/downgrade/' || source_id,
            canonical_url = 'https://example.invalid/downgrade/' || source_id,
            source_type = 'approved_internal_faq',
            status = 'inaccessible'
        WHERE source_type = 'local_import'
        """
    )
    op.execute(
        """
        UPDATE knowledge_items
        SET source_uri = 'https://example.invalid/downgrade/' || source_id,
            source_type = 'approved_internal_faq'
        WHERE source_type = 'local_import'
        """
    )
    op.execute(
        """
        UPDATE faq_import_batches
        SET source_url = 'https://example.invalid/downgrade/' || source_id
        WHERE source_type = 'local_import'
        """
    )

    with op.batch_alter_table("faq_import_batches") as batch:
        batch.drop_constraint(
            "uq_faq_import_batch_file_source_id",
            type_="unique",
        )
        batch.alter_column("source_url", existing_type=sa.Text(), nullable=False)
        batch.drop_column("source_type")
    with op.batch_alter_table("knowledge_items") as batch:
        batch.alter_column("source_uri", existing_type=sa.Text(), nullable=False)
    with op.batch_alter_table("knowledge_sources") as batch:
        batch.alter_column("canonical_url", existing_type=sa.Text(), nullable=False)
        batch.alter_column("supplied_url", existing_type=sa.Text(), nullable=False)
