from datetime import datetime

from pydantic import ValidationError
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    Text,
    and_,
    create_engine,
    or_,
    select,
)
from sqlalchemy.engine import Engine, RowMapping
from sqlalchemy.exc import SQLAlchemyError

from .models import KnowledgeDocument, KnowledgeItem, KnowledgeSource

_metadata = MetaData()
_sources = Table(
    "knowledge_sources",
    _metadata,
    Column("source_id", String),
    Column("supplied_url", Text),
    Column("canonical_url", Text),
    Column("title", Text),
    Column("publisher", String),
    Column("source_type", String),
    Column("retrieved_at", DateTime(timezone=True)),
    Column("topics", JSON),
    Column("status", String),
    Column("notes", Text),
)
_items = Table(
    "knowledge_items",
    _metadata,
    Column("knowledge_id", String),
    Column("title", Text),
    Column("standard_answer", Text),
    Column("source_id", String),
    Column("source_uri", Text),
    Column("source_locator", Text),
    Column("source_type", String),
    Column("products", JSON),
    Column("platforms", JSON),
    Column("app_versions", JSON),
    Column("effective_at", DateTime(timezone=True)),
    Column("expires_at", DateTime(timezone=True)),
    Column("review_at", DateTime(timezone=True)),
    Column("owner_unit", String),
    Column("author", String),
    Column("reviewer", String),
    Column("approver", String),
    Column("approved_at", DateTime(timezone=True)),
    Column("version", String),
    Column("change_summary", Text),
    Column("previous_version", String),
    Column("status", String),
    Column("public_answer_allowed", Boolean),
    Column("allowed_intents", JSON),
    Column("prohibited_extensions", JSON),
)


class KnowledgeRepositoryError(RuntimeError):
    pass


class SqlKnowledgeRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    @classmethod
    def from_url(cls, database_url: str) -> "SqlKnowledgeRepository":
        return cls(create_engine(database_url, pool_pre_ping=True))

    def check_connection(self) -> None:
        try:
            with self.engine.connect() as connection:
                connection.execute(select(_sources.c.source_id).limit(1))
        except (SQLAlchemyError, ValidationError) as exc:
            raise KnowledgeRepositoryError("knowledge database unavailable") from exc

    def eligible_documents(self, *, at: datetime) -> tuple[KnowledgeDocument, ...]:
        if at.tzinfo is None:
            raise ValueError("eligible_documents 的時間必須包含時區")
        statement = (
            select(
                *(_items.c[column] for column in _ITEM_FIELDS),
                *(_sources.c[column].label(f"source_{column}") for column in _SOURCE_FIELDS),
            )
            .select_from(_items.join(_sources, _items.c.source_id == _sources.c.source_id))
            .where(
                and_(
                    _items.c.status == "published",
                    _items.c.public_answer_allowed.is_(True),
                    _sources.c.status == "active",
                    _items.c.effective_at.is_not(None),
                    _items.c.effective_at <= at,
                    or_(_items.c.expires_at.is_(None), _items.c.expires_at > at),
                    _items.c.review_at.is_not(None),
                    _items.c.review_at >= at,
                )
            )
            .order_by(_items.c.knowledge_id)
        )
        try:
            with self.engine.connect() as connection:
                rows = connection.execute(statement).mappings()
                return tuple(_row_to_document(row) for row in rows)
        except (SQLAlchemyError, ValidationError) as exc:
            raise KnowledgeRepositoryError("knowledge database unavailable") from exc


_ITEM_FIELDS = (
    "knowledge_id",
    "title",
    "standard_answer",
    "source_id",
    "source_uri",
    "source_locator",
    "source_type",
    "products",
    "platforms",
    "app_versions",
    "effective_at",
    "expires_at",
    "review_at",
    "owner_unit",
    "author",
    "reviewer",
    "approver",
    "approved_at",
    "version",
    "change_summary",
    "previous_version",
    "status",
    "public_answer_allowed",
    "allowed_intents",
    "prohibited_extensions",
)
_SOURCE_FIELDS = (
    "source_id",
    "supplied_url",
    "canonical_url",
    "title",
    "publisher",
    "source_type",
    "retrieved_at",
    "topics",
    "status",
    "notes",
)


def _row_to_document(row: RowMapping) -> KnowledgeDocument:
    item = KnowledgeItem.model_validate({field: row[field] for field in _ITEM_FIELDS})
    source = KnowledgeSource.model_validate(
        {field: row[f"source_{field}"] for field in _SOURCE_FIELDS}
    )
    return KnowledgeDocument(item=item, source=source)
