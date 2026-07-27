"""Add governed ASR terms and observed recognition aliases."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_knowledge_asr_terms"
down_revision: str | None = "0006_local_import_sources"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("knowledge_items") as batch:
        batch.add_column(
            sa.Column(
                "asr_terms",
                sa.JSON(),
                server_default="[]",
                nullable=False,
            )
        )
        batch.alter_column("asr_terms", server_default=None)


def downgrade() -> None:
    with op.batch_alter_table("knowledge_items") as batch:
        batch.drop_column("asr_terms")
