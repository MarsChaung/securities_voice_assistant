from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

json_type = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class KnowledgeSourceRecord(Base):
    __tablename__ = "knowledge_sources"

    source_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    supplied_url: Mapped[str | None] = mapped_column(Text)
    canonical_url: Mapped[str | None] = mapped_column(Text, unique=True)
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
    source_uri: Mapped[str | None] = mapped_column(Text)
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
    asr_terms: Mapped[list[dict[str, object]]] = mapped_column(json_type, default=list)
    row_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    question_variants: Mapped[list["KnowledgeQuestionVariantRecord"]] = relationship(
        back_populates="knowledge_item",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="KnowledgeQuestionVariantRecord.position",
    )


class KnowledgeQuestionVariantRecord(Base):
    __tablename__ = "knowledge_question_variants"
    __table_args__ = (
        UniqueConstraint(
            "knowledge_id",
            "normalized_text",
            name="uq_knowledge_question_variant_normalized",
        ),
    )

    variant_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    knowledge_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_items.knowledge_id", ondelete="CASCADE"), index=True
    )
    question_text: Mapped[str] = mapped_column(Text)
    normalized_text: Mapped[str] = mapped_column(Text)
    usage: Mapped[str] = mapped_column(String(30), index=True)
    position: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    knowledge_item: Mapped[KnowledgeItemRecord] = relationship(back_populates="question_variants")


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


class FaqImportBatchRecord(Base):
    __tablename__ = "faq_import_batches"
    __table_args__ = (
        UniqueConstraint(
            "file_sha256",
            "source_url",
            name="uq_faq_import_batch_file_source",
        ),
        UniqueConstraint(
            "file_sha256",
            "source_id",
            name="uq_faq_import_batch_file_source_id",
        ),
    )

    batch_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    file_sha256: Mapped[str] = mapped_column(String(64), index=True)
    dataset_title: Mapped[str] = mapped_column(String(200))
    publisher: Mapped[str] = mapped_column(String(200))
    source_url: Mapped[str | None] = mapped_column(Text)
    source_id: Mapped[str] = mapped_column(String(80))
    source_type: Mapped[str] = mapped_column(String(40))
    uploaded_by: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(30), index=True)
    sheet_name: Mapped[str] = mapped_column(String(200))
    rows: Mapped[list[dict[str, object]]] = mapped_column(json_type)
    row_count: Mapped[int] = mapped_column(Integer)
    valid_row_count: Mapped[int] = mapped_column(Integer)
    imported_count: Mapped[int] = mapped_column(Integer, default=0)
    row_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    imported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
