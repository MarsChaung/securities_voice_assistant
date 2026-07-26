import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import ceil
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Engine,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class ShadowReviewStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NOT_REVIEWABLE = "not_reviewable"


class ShadowReviewLabel(StrEnum):
    ACCEPTABLE = "acceptable"
    INCORRECT = "incorrect"
    UNSUPPORTED_EXTENSION = "unsupported_extension"
    MISSING_QUALIFIER = "missing_qualifier"
    UNSAFE = "unsafe"
    TONE = "tone"
    OTHER = "other"


@dataclass(frozen=True)
class ShadowReviewInput:
    turn_id: str
    knowledge_id: str
    knowledge_version: str
    source_id: str
    standard_answer: str
    prohibited_extensions: tuple[str, ...]
    generated_answer: str | None
    generation_model_id: str | None
    prompt_version: str | None
    prompt_hash: str | None
    generation_latency_ms: float | None
    output_guard_safe: bool | None
    fallback_reason: str


@dataclass(frozen=True)
class ShadowReviewEntry:
    shadow_id: str
    result_key: str
    turn_id: str
    knowledge_id: str
    knowledge_version: str
    source_id: str
    standard_answer: str
    prohibited_extensions: tuple[str, ...]
    generated_answer: str | None
    generation_model_id: str | None
    prompt_version: str | None
    prompt_hash: str | None
    generation_latency_ms: float | None
    output_guard_safe: bool | None
    fallback_reason: str
    review_status: ShadowReviewStatus
    review_label: ShadowReviewLabel | None
    reviewer_id: str | None
    reviewer_note: str | None
    created_at: datetime
    reviewed_at: datetime | None
    row_version: int


@dataclass(frozen=True)
class ShadowReviewMetrics:
    total: int
    pending: int
    accepted: int
    rejected: int
    not_reviewable: int
    output_guard_safe: int
    output_guard_blocked: int
    acceptance_rate: float | None
    average_latency_ms: float | None
    p95_latency_ms: float | None


class ShadowReviewNotFoundError(LookupError):
    pass


class ShadowReviewConcurrentUpdateError(RuntimeError):
    pass


class ShadowReviewStateError(RuntimeError):
    pass


class ShadowReviewBase(DeclarativeBase):
    pass


class ShadowReviewRecord(ShadowReviewBase):
    __tablename__ = "shadow_review_results"

    shadow_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    result_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    turn_id: Mapped[str] = mapped_column(String(36), index=True)
    knowledge_id: Mapped[str] = mapped_column(String(80), index=True)
    knowledge_version: Mapped[str] = mapped_column(String(40))
    source_id: Mapped[str] = mapped_column(String(80))
    standard_answer: Mapped[str] = mapped_column(Text)
    prohibited_extensions_json: Mapped[str] = mapped_column(Text)
    generated_answer: Mapped[str | None] = mapped_column(Text)
    generation_model_id: Mapped[str | None] = mapped_column(String(200))
    prompt_version: Mapped[str | None] = mapped_column(String(100))
    prompt_hash: Mapped[str | None] = mapped_column(String(64))
    generation_latency_ms: Mapped[float | None] = mapped_column(Float)
    output_guard_safe: Mapped[bool | None] = mapped_column(Boolean)
    fallback_reason: Mapped[str] = mapped_column(String(200))
    review_status: Mapped[str] = mapped_column(String(30), index=True)
    review_label: Mapped[str | None] = mapped_column(String(40))
    reviewer_id: Mapped[str | None] = mapped_column(String(200))
    reviewer_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    row_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class DatabaseShadowReviewRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    @classmethod
    def from_url(cls, database_url: str) -> "DatabaseShadowReviewRepository":
        return cls(create_engine(database_url, pool_pre_ping=True))

    def check_connection(self) -> None:
        with self._sessions() as session:
            session.scalar(select(func.count()).select_from(ShadowReviewRecord))

    def record(self, item: ShadowReviewInput, *, now: datetime | None = None) -> ShadowReviewEntry:
        created_at = _as_utc(now or datetime.now(UTC))
        result_key = _result_key(item)
        status = (
            ShadowReviewStatus.PENDING
            if item.generated_answer is not None
            else ShadowReviewStatus.NOT_REVIEWABLE
        )
        try:
            with self._sessions.begin() as session:
                record = ShadowReviewRecord(
                    shadow_id=str(uuid4()),
                    result_key=result_key,
                    turn_id=item.turn_id,
                    knowledge_id=item.knowledge_id,
                    knowledge_version=item.knowledge_version,
                    source_id=item.source_id,
                    standard_answer=item.standard_answer,
                    prohibited_extensions_json=json.dumps(
                        item.prohibited_extensions, ensure_ascii=False
                    ),
                    generated_answer=item.generated_answer,
                    generation_model_id=item.generation_model_id,
                    prompt_version=item.prompt_version,
                    prompt_hash=item.prompt_hash,
                    generation_latency_ms=item.generation_latency_ms,
                    output_guard_safe=item.output_guard_safe,
                    fallback_reason=item.fallback_reason,
                    review_status=status.value,
                    review_label=None,
                    reviewer_id=None,
                    reviewer_note=None,
                    created_at=created_at,
                    reviewed_at=None,
                    row_version=1,
                )
                session.add(record)
                session.flush()
                return _to_domain(record)
        except IntegrityError:
            with self._sessions() as session:
                existing = session.scalar(
                    select(ShadowReviewRecord).where(
                        ShadowReviewRecord.result_key == result_key
                    )
                )
                if existing is None:
                    raise
                return _to_domain(existing)

    def list_results(
        self,
        *,
        status: ShadowReviewStatus | None = None,
        knowledge_id: str | None = None,
    ) -> tuple[ShadowReviewEntry, ...]:
        statement = select(ShadowReviewRecord)
        if status is not None:
            statement = statement.where(ShadowReviewRecord.review_status == status.value)
        if knowledge_id:
            statement = statement.where(ShadowReviewRecord.knowledge_id == knowledge_id)
        statement = statement.order_by(
            ShadowReviewRecord.created_at.desc(), ShadowReviewRecord.shadow_id
        )
        with self._sessions() as session:
            return tuple(_to_domain(record) for record in session.scalars(statement))

    def get_result(self, shadow_id: str) -> ShadowReviewEntry:
        with self._sessions() as session:
            record = session.get(ShadowReviewRecord, shadow_id)
            if record is None:
                raise ShadowReviewNotFoundError(shadow_id)
            return _to_domain(record)

    def review(
        self,
        *,
        shadow_id: str,
        label: ShadowReviewLabel,
        reviewer_id: str,
        reviewer_note: str | None,
        expected_version: int,
        now: datetime | None = None,
    ) -> ShadowReviewEntry:
        reviewed_at = _as_utc(now or datetime.now(UTC))
        normalized_note = reviewer_note.strip() if reviewer_note else None
        if label is ShadowReviewLabel.OTHER and not normalized_note:
            raise ValueError("選擇其他問題時必須填寫複核說明")
        with self._sessions.begin() as session:
            record = session.scalar(
                select(ShadowReviewRecord)
                .where(ShadowReviewRecord.shadow_id == shadow_id)
                .with_for_update()
            )
            if record is None:
                raise ShadowReviewNotFoundError(shadow_id)
            if record.row_version != expected_version:
                raise ShadowReviewConcurrentUpdateError("資料已被其他複核操作更新")
            if record.review_status != ShadowReviewStatus.PENDING.value:
                raise ShadowReviewStateError("此 Shadow 結果已完成複核或不可複核")

            record.review_status = (
                ShadowReviewStatus.ACCEPTED.value
                if label is ShadowReviewLabel.ACCEPTABLE
                else ShadowReviewStatus.REJECTED.value
            )
            record.review_label = label.value
            record.reviewer_id = reviewer_id
            record.reviewer_note = normalized_note
            record.reviewed_at = reviewed_at
            record.row_version += 1
            session.flush()
            return _to_domain(record)

    def metrics(self) -> ShadowReviewMetrics:
        results = self.list_results()
        accepted = sum(item.review_status is ShadowReviewStatus.ACCEPTED for item in results)
        rejected = sum(item.review_status is ShadowReviewStatus.REJECTED for item in results)
        reviewed = accepted + rejected
        latencies = sorted(
            item.generation_latency_ms
            for item in results
            if item.generation_latency_ms is not None
        )
        return ShadowReviewMetrics(
            total=len(results),
            pending=sum(item.review_status is ShadowReviewStatus.PENDING for item in results),
            accepted=accepted,
            rejected=rejected,
            not_reviewable=sum(
                item.review_status is ShadowReviewStatus.NOT_REVIEWABLE for item in results
            ),
            output_guard_safe=sum(item.output_guard_safe is True for item in results),
            output_guard_blocked=sum(item.output_guard_safe is False for item in results),
            acceptance_rate=accepted / reviewed if reviewed else None,
            average_latency_ms=(sum(latencies) / len(latencies) if latencies else None),
            p95_latency_ms=(
                latencies[max(0, ceil(len(latencies) * 0.95) - 1)] if latencies else None
            ),
        )


def _result_key(item: ShadowReviewInput) -> str:
    values = (
        item.knowledge_id,
        item.knowledge_version,
        item.source_id,
        item.generation_model_id or "generation-error",
        item.prompt_hash or item.prompt_version or item.fallback_reason,
    )
    return hashlib.sha256("\x1f".join(values).encode()).hexdigest()


def _to_domain(record: ShadowReviewRecord) -> ShadowReviewEntry:
    return ShadowReviewEntry(
        shadow_id=record.shadow_id,
        result_key=record.result_key,
        turn_id=record.turn_id,
        knowledge_id=record.knowledge_id,
        knowledge_version=record.knowledge_version,
        source_id=record.source_id,
        standard_answer=record.standard_answer,
        prohibited_extensions=tuple(json.loads(record.prohibited_extensions_json)),
        generated_answer=record.generated_answer,
        generation_model_id=record.generation_model_id,
        prompt_version=record.prompt_version,
        prompt_hash=record.prompt_hash,
        generation_latency_ms=record.generation_latency_ms,
        output_guard_safe=record.output_guard_safe,
        fallback_reason=record.fallback_reason,
        review_status=ShadowReviewStatus(record.review_status),
        review_label=(
            ShadowReviewLabel(record.review_label) if record.review_label is not None else None
        ),
        reviewer_id=record.reviewer_id,
        reviewer_note=record.reviewer_note,
        created_at=_as_utc(record.created_at),
        reviewed_at=_as_utc(record.reviewed_at) if record.reviewed_at is not None else None,
        row_version=record.row_version,
    )


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
