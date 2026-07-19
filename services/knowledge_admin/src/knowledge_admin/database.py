from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

json_type = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class KnowledgeSourceRecord(Base):
    __tablename__ = "knowledge_sources"

    source_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    supplied_url: Mapped[str] = mapped_column(Text)
    canonical_url: Mapped[str] = mapped_column(Text, unique=True)
    title: Mapped[str] = mapped_column(Text)
    publisher: Mapped[str] = mapped_column(String(200))
    source_type: Mapped[str] = mapped_column(String(40))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    topics: Mapped[list[str]] = mapped_column(json_type)
    status: Mapped[str] = mapped_column(String(30), index=True)
    notes: Mapped[str | None] = mapped_column(Text)


class KnowledgeItemRecord(Base):
    __tablename__ = "knowledge_items"

    knowledge_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    standard_answer: Mapped[str] = mapped_column(Text)
    source_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_sources.source_id", ondelete="RESTRICT"), index=True
    )
    source_uri: Mapped[str] = mapped_column(Text)
    source_locator: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(40))
    products: Mapped[list[str]] = mapped_column(json_type)
    platforms: Mapped[list[str]] = mapped_column(json_type)
    app_versions: Mapped[list[str]] = mapped_column(json_type)
    effective_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    owner_unit: Mapped[str | None] = mapped_column(String(200))
    author: Mapped[str] = mapped_column(String(200))
    reviewer: Mapped[str | None] = mapped_column(String(200))
    approver: Mapped[str | None] = mapped_column(String(200))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    version: Mapped[str] = mapped_column(String(40))
    change_summary: Mapped[str] = mapped_column(Text)
    previous_version: Mapped[str | None] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(30), index=True)
    public_answer_allowed: Mapped[bool] = mapped_column(Boolean, default=False)
    allowed_intents: Mapped[list[str]] = mapped_column(json_type)
    prohibited_extensions: Mapped[list[str]] = mapped_column(json_type)
    row_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class KnowledgeItemVersionRecord(Base):
    __tablename__ = "knowledge_item_versions"
    __table_args__ = (
        UniqueConstraint("knowledge_id", "version", name="uq_knowledge_item_version"),
    )

    version_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    knowledge_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_items.knowledge_id", ondelete="RESTRICT"), index=True
    )
    version: Mapped[str] = mapped_column(String(40))
    item_snapshot: Mapped[dict[str, object]] = mapped_column(json_type)
    archived_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    archived_by: Mapped[str] = mapped_column(String(200))


class GovernanceEventRecord(Base):
    __tablename__ = "knowledge_governance_events"

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    knowledge_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_items.knowledge_id", ondelete="RESTRICT"), index=True
    )
    action: Mapped[str] = mapped_column(String(40))
    from_status: Mapped[str] = mapped_column(String(30))
    to_status: Mapped[str] = mapped_column(String(30))
    actor_id: Mapped[str] = mapped_column(String(200))
    reason: Mapped[str | None] = mapped_column(Text)
    row_version: Mapped[int] = mapped_column(Integer)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
