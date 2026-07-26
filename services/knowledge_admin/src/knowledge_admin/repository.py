import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.orm import sessionmaker

from retrieval import (
    KnowledgeItem,
    KnowledgeSource,
    KnowledgeStatus,
    LocalKnowledgeRepository,
    QuestionVariant,
    QuestionVariantUsage,
)

from .database import (
    GovernanceEventRecord,
    KnowledgeItemRecord,
    KnowledgeItemVersionRecord,
    KnowledgeQuestionVariantRecord,
    KnowledgeSourceRecord,
)
from .governance import GovernanceAction, GovernanceActor, GovernancePolicy


class KnowledgeNotFoundError(LookupError):
    pass


class ConcurrentUpdateError(RuntimeError):
    pass


@dataclass(frozen=True)
class GovernancePayload:
    effective_at: datetime | None = None
    review_at: datetime | None = None
    owner_unit: str | None = None
    app_versions: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class QuestionVariantInput:
    variant_id: str | None
    question_text: str
    usage: QuestionVariantUsage


@dataclass(frozen=True)
class GovernanceEvent:
    event_id: str
    knowledge_id: str
    action: GovernanceAction
    from_status: KnowledgeStatus
    to_status: KnowledgeStatus
    actor_id: str
    reason: str | None
    row_version: int
    occurred_at: datetime


@dataclass(frozen=True)
class KnowledgeRecord:
    item: KnowledgeItem
    row_version: int


@dataclass(frozen=True)
class KnowledgeVersionSnapshot:
    item: KnowledgeItem
    archived_at: datetime
    archived_by: str


class DatabaseKnowledgeRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    @classmethod
    def from_url(cls, database_url: str) -> "DatabaseKnowledgeRepository":
        return cls(create_engine(database_url, pool_pre_ping=True))

    def check_connection(self) -> None:
        with self._sessions() as session:
            session.scalar(select(func.count()).select_from(KnowledgeSourceRecord))

    def list_sources(self) -> tuple[KnowledgeSource, ...]:
        with self._sessions() as session:
            records = session.scalars(
                select(KnowledgeSourceRecord).order_by(KnowledgeSourceRecord.source_id)
            )
            return tuple(_source_to_domain(record) for record in records)

    def list_items(self) -> tuple[KnowledgeRecord, ...]:
        with self._sessions() as session:
            records = session.scalars(
                select(KnowledgeItemRecord).order_by(KnowledgeItemRecord.knowledge_id)
            )
            return tuple(
                KnowledgeRecord(item=_item_to_domain(record), row_version=record.row_version)
                for record in records
            )

    def get_item(self, knowledge_id: str) -> KnowledgeRecord:
        with self._sessions() as session:
            record = session.get(KnowledgeItemRecord, knowledge_id)
            if record is None:
                raise KnowledgeNotFoundError(knowledge_id)
            return KnowledgeRecord(item=_item_to_domain(record), row_version=record.row_version)

    def get_source(self, source_id: str) -> KnowledgeSource:
        with self._sessions() as session:
            record = session.get(KnowledgeSourceRecord, source_id)
            if record is None:
                raise KnowledgeNotFoundError(source_id)
            return _source_to_domain(record)

    def list_events(self, knowledge_id: str) -> tuple[GovernanceEvent, ...]:
        with self._sessions() as session:
            records = session.scalars(
                select(GovernanceEventRecord)
                .where(GovernanceEventRecord.knowledge_id == knowledge_id)
                .order_by(GovernanceEventRecord.row_version, GovernanceEventRecord.event_id)
            )
            return tuple(_event_to_domain(record) for record in records)

    def list_versions(self, knowledge_id: str) -> tuple[KnowledgeVersionSnapshot, ...]:
        with self._sessions() as session:
            records = session.scalars(
                select(KnowledgeItemVersionRecord)
                .where(KnowledgeItemVersionRecord.knowledge_id == knowledge_id)
                .order_by(
                    KnowledgeItemVersionRecord.archived_at.desc(),
                    KnowledgeItemVersionRecord.version_id,
                )
            )
            return tuple(_version_to_domain(record) for record in records)

    def perform_action(
        self,
        *,
        knowledge_id: str,
        action: GovernanceAction,
        actor: GovernanceActor,
        expected_version: int,
        payload: GovernancePayload | None = None,
        now: datetime | None = None,
    ) -> KnowledgeRecord:
        if action in {
            GovernanceAction.UPDATE_CONTENT,
            GovernanceAction.UPDATE_QUESTION_VARIANTS,
        }:
            raise ValueError("草稿內容必須使用專用編輯操作")
        resolved_payload = payload or GovernancePayload()
        occurred_at = now or datetime.now(UTC)
        if occurred_at.tzinfo is None:
            raise ValueError("治理事件時間必須包含時區")

        with self._sessions.begin() as session:
            record = session.scalar(
                select(KnowledgeItemRecord)
                .where(KnowledgeItemRecord.knowledge_id == knowledge_id)
                .with_for_update()
            )
            if record is None:
                raise KnowledgeNotFoundError(knowledge_id)
            if record.row_version != expected_version:
                raise ConcurrentUpdateError("資料已被其他操作更新，請重新載入")

            current_item = _item_to_domain(record)
            next_status = GovernancePolicy.next_status(current_item, action, actor)
            previous_status = current_item.status

            if action is GovernanceAction.START_REVISION:
                session.add(
                    KnowledgeItemVersionRecord(
                        version_id=str(uuid4()),
                        knowledge_id=knowledge_id,
                        version=current_item.version,
                        item_snapshot=current_item.model_dump(mode="json"),
                        archived_at=occurred_at,
                        archived_by=actor.actor_id,
                    )
                )

            self._apply_action(record, action, actor, resolved_payload, occurred_at)
            record.status = next_status.value
            record.row_version += 1
            record.updated_at = occurred_at

            updated_item = _item_to_domain(record)
            session.add(
                GovernanceEventRecord(
                    event_id=str(uuid4()),
                    knowledge_id=knowledge_id,
                    action=action.value,
                    from_status=previous_status.value,
                    to_status=updated_item.status.value,
                    actor_id=actor.actor_id,
                    reason=resolved_payload.reason,
                    row_version=record.row_version,
                    occurred_at=occurred_at,
                )
            )
            session.flush()
            return KnowledgeRecord(item=updated_item, row_version=record.row_version)

    def update_content(
        self,
        *,
        knowledge_id: str,
        title: str,
        standard_answer: str,
        actor: GovernanceActor,
        expected_version: int,
        now: datetime | None = None,
    ) -> KnowledgeRecord:
        occurred_at = now or datetime.now(UTC)
        if occurred_at.tzinfo is None:
            raise ValueError("治理事件時間必須包含時區")

        with self._sessions.begin() as session:
            record = session.scalar(
                select(KnowledgeItemRecord)
                .where(KnowledgeItemRecord.knowledge_id == knowledge_id)
                .with_for_update()
            )
            if record is None:
                raise KnowledgeNotFoundError(knowledge_id)
            if record.row_version != expected_version:
                raise ConcurrentUpdateError("資料已被其他操作更新，請重新載入")

            current_item = _item_to_domain(record)
            GovernancePolicy.next_status(
                current_item,
                GovernanceAction.UPDATE_CONTENT,
                actor,
            )
            validated = KnowledgeItem.model_validate(
                current_item.model_dump(mode="json")
                | {
                    "title": title.strip(),
                    "standard_answer": standard_answer.strip(),
                }
            )
            changed_fields = sum(
                (
                    record.title != validated.title,
                    record.standard_answer != validated.standard_answer,
                )
            )
            if changed_fields == 0:
                raise ValueError("知識內容沒有變更")

            record.title = validated.title
            record.standard_answer = validated.standard_answer
            record.row_version += 1
            record.updated_at = occurred_at
            session.add(
                GovernanceEventRecord(
                    event_id=str(uuid4()),
                    knowledge_id=knowledge_id,
                    action=GovernanceAction.UPDATE_CONTENT.value,
                    from_status=KnowledgeStatus.DRAFT.value,
                    to_status=KnowledgeStatus.DRAFT.value,
                    actor_id=actor.actor_id,
                    reason=f"更新 {changed_fields} 個知識內容欄位",
                    row_version=record.row_version,
                    occurred_at=occurred_at,
                )
            )
            session.flush()
            return KnowledgeRecord(
                item=_item_to_domain(record),
                row_version=record.row_version,
            )

    def update_question_variants(
        self,
        *,
        knowledge_id: str,
        variants: tuple[QuestionVariantInput, ...],
        actor: GovernanceActor,
        expected_version: int,
        now: datetime | None = None,
    ) -> KnowledgeRecord:
        occurred_at = now or datetime.now(UTC)
        if occurred_at.tzinfo is None:
            raise ValueError("治理事件時間必須包含時區")

        with self._sessions.begin() as session:
            record = session.scalar(
                select(KnowledgeItemRecord)
                .where(KnowledgeItemRecord.knowledge_id == knowledge_id)
                .with_for_update()
            )
            if record is None:
                raise KnowledgeNotFoundError(knowledge_id)
            if record.row_version != expected_version:
                raise ConcurrentUpdateError("資料已被其他操作更新，請重新載入")

            current_item = _item_to_domain(record)
            GovernancePolicy.next_status(
                current_item,
                GovernanceAction.UPDATE_QUESTION_VARIANTS,
                actor,
            )
            resolved_variants = _validate_variant_inputs(
                variants,
                existing=record.question_variants,
            )
            retrieval_normalized = {
                normalized
                for _, _, usage, normalized in resolved_variants
                if usage is QuestionVariantUsage.RETRIEVAL
            }
            if retrieval_normalized:
                conflict = session.scalar(
                    select(KnowledgeQuestionVariantRecord)
                    .where(
                        KnowledgeQuestionVariantRecord.knowledge_id != knowledge_id,
                        KnowledgeQuestionVariantRecord.usage
                        == QuestionVariantUsage.RETRIEVAL.value,
                        KnowledgeQuestionVariantRecord.normalized_text.in_(
                            retrieval_normalized
                        ),
                    )
                    .limit(1)
                )
                if conflict is not None:
                    raise ValueError(
                        f"問句變體與 {conflict.knowledge_id} 的正式檢索問句重複"
                    )

            existing_by_id = {
                variant.variant_id: variant for variant in record.question_variants
            }
            next_records: list[KnowledgeQuestionVariantRecord] = []
            added = 0
            updated = 0
            retained_ids: set[str] = set()
            for position, (variant_id, question_text, usage, normalized) in enumerate(
                resolved_variants
            ):
                if variant_id is None:
                    variant_record = KnowledgeQuestionVariantRecord(
                        variant_id=str(uuid4()),
                        knowledge_id=knowledge_id,
                        question_text=question_text,
                        normalized_text=normalized,
                        usage=usage.value,
                        position=position,
                        created_at=occurred_at,
                        updated_at=occurred_at,
                    )
                    added += 1
                else:
                    variant_record = existing_by_id[variant_id]
                    retained_ids.add(variant_id)
                    if (
                        variant_record.question_text != question_text
                        or variant_record.usage != usage.value
                        or variant_record.position != position
                    ):
                        updated += 1
                    variant_record.question_text = question_text
                    variant_record.normalized_text = normalized
                    variant_record.usage = usage.value
                    variant_record.position = position
                    variant_record.updated_at = occurred_at
                next_records.append(variant_record)

            deleted = len(existing_by_id.keys() - retained_ids)
            record.question_variants = next_records
            record.row_version += 1
            record.updated_at = occurred_at
            session.add(
                GovernanceEventRecord(
                    event_id=str(uuid4()),
                    knowledge_id=knowledge_id,
                    action=GovernanceAction.UPDATE_QUESTION_VARIANTS.value,
                    from_status=KnowledgeStatus.DRAFT.value,
                    to_status=KnowledgeStatus.DRAFT.value,
                    actor_id=actor.actor_id,
                    reason=f"問句變體：新增 {added}、修改 {updated}、刪除 {deleted}",
                    row_version=record.row_version,
                    occurred_at=occurred_at,
                )
            )
            session.flush()
            return KnowledgeRecord(
                item=_item_to_domain(record),
                row_version=record.row_version,
            )

    @staticmethod
    def _apply_action(
        record: KnowledgeItemRecord,
        action: GovernanceAction,
        actor: GovernanceActor,
        payload: GovernancePayload,
        occurred_at: datetime,
    ) -> None:
        if action is GovernanceAction.START_REVISION:
            if not payload.reason or not payload.reason.strip():
                raise ValueError("建立複審新版時必須填寫版本說明")
            if record.review_at is None:
                raise ValueError("只有具備複審到期時間的發布知識可以建立新版")
            review_deadline = _as_utc(record.review_at)
            if review_deadline > occurred_at.astimezone(UTC):
                raise ValueError("複審尚未到期，不得提前將線上版本轉為草稿")
            record.previous_version = record.version
            record.version = _next_draft_version(record.version)
            record.change_summary = payload.reason.strip()
            record.public_answer_allowed = False
            record.reviewer = None
            record.approver = None
            record.approved_at = None
            record.effective_at = None
            record.expires_at = None
            record.review_at = None
            record.owner_unit = None
            record.app_versions = []
        elif action is GovernanceAction.COMPLETE_REVIEW:
            record.reviewer = actor.actor_id
        elif action is GovernanceAction.APPROVE:
            if payload.effective_at is None or payload.review_at is None or not payload.owner_unit:
                raise ValueError("核准時必須指定生效時間、複審到期時間與權責單位")
            if payload.effective_at.tzinfo is None or payload.review_at.tzinfo is None:
                raise ValueError("生效與複審到期時間必須包含時區")
            if payload.review_at <= payload.effective_at:
                raise ValueError("複審到期時間必須晚於生效時間")
            if payload.review_at <= occurred_at:
                raise ValueError("複審到期時間必須晚於目前時間")
            record.approver = actor.actor_id
            record.approved_at = occurred_at
            record.effective_at = payload.effective_at
            record.review_at = payload.review_at
            record.owner_unit = payload.owner_unit
            record.app_versions = list(payload.app_versions)
        elif action is GovernanceAction.PUBLISH:
            record.public_answer_allowed = True
            if record.version.endswith("-draft"):
                if record.previous_version is None:
                    record.previous_version = record.version
                    record.version = "1.0"
                else:
                    record.version = record.version.removesuffix("-draft")
        elif action is GovernanceAction.RETURN_DRAFT:
            if not payload.reason or not payload.reason.strip():
                raise ValueError("退回草稿時必須填寫原因")
            record.public_answer_allowed = False
            record.reviewer = None
            record.approver = None
            record.approved_at = None
            record.effective_at = None
            record.review_at = None
            record.owner_unit = None
            record.app_versions = []
        elif action is GovernanceAction.REVOKE:
            if not payload.reason or not payload.reason.strip():
                raise ValueError("撤銷時必須填寫原因")
            record.public_answer_allowed = False

    def seed_from_files(self, knowledge_root: Path) -> tuple[int, int]:
        local_repository = LocalKnowledgeRepository.load(knowledge_root)
        inserted_sources = 0
        inserted_items = 0
        now = datetime.now(UTC)

        with self._sessions.begin() as session:
            for source in local_repository.sources:
                if session.get(KnowledgeSourceRecord, source.source_id) is None:
                    session.add(_source_from_domain(source))
                    inserted_sources += 1
            session.flush()
            for item in local_repository.items:
                if session.get(KnowledgeItemRecord, item.knowledge_id) is None:
                    session.add(_item_from_domain(item, now=now))
                    inserted_items += 1

        return inserted_sources, inserted_items


def _source_to_domain(record: KnowledgeSourceRecord) -> KnowledgeSource:
    return KnowledgeSource.model_validate(
        {
            "source_id": record.source_id,
            "supplied_url": record.supplied_url,
            "canonical_url": record.canonical_url,
            "title": record.title,
            "publisher": record.publisher,
            "source_type": record.source_type,
            "retrieved_at": record.retrieved_at,
            "topics": record.topics,
            "status": record.status,
            "notes": record.notes,
        }
    )


def _item_to_domain(record: KnowledgeItemRecord) -> KnowledgeItem:
    return KnowledgeItem.model_validate(
        {
            "knowledge_id": record.knowledge_id,
            "title": record.title,
            "standard_answer": record.standard_answer,
            "source_id": record.source_id,
            "source_uri": record.source_uri,
            "source_locator": record.source_locator,
            "source_type": record.source_type,
            "products": record.products,
            "platforms": record.platforms,
            "app_versions": record.app_versions,
            "effective_at": _optional_utc(record.effective_at),
            "expires_at": _optional_utc(record.expires_at),
            "review_at": _optional_utc(record.review_at),
            "owner_unit": record.owner_unit,
            "author": record.author,
            "reviewer": record.reviewer,
            "approver": record.approver,
            "approved_at": _optional_utc(record.approved_at),
            "version": record.version,
            "change_summary": record.change_summary,
            "previous_version": record.previous_version,
            "status": record.status,
            "public_answer_allowed": record.public_answer_allowed,
            "allowed_intents": record.allowed_intents,
            "prohibited_extensions": record.prohibited_extensions,
            "question_variants": [
                {
                    "variant_id": variant.variant_id,
                    "question_text": variant.question_text,
                    "usage": variant.usage,
                }
                for variant in record.question_variants
            ],
        }
    )


def _event_to_domain(record: GovernanceEventRecord) -> GovernanceEvent:
    occurred_at = (
        record.occurred_at
        if record.occurred_at.tzinfo is not None
        else record.occurred_at.replace(tzinfo=UTC)
    )
    return GovernanceEvent(
        event_id=record.event_id,
        knowledge_id=record.knowledge_id,
        action=GovernanceAction(record.action),
        from_status=KnowledgeStatus(record.from_status),
        to_status=KnowledgeStatus(record.to_status),
        actor_id=record.actor_id,
        reason=record.reason,
        row_version=record.row_version,
        occurred_at=occurred_at,
    )


def _version_to_domain(record: KnowledgeItemVersionRecord) -> KnowledgeVersionSnapshot:
    archived_at = _as_utc(record.archived_at)
    return KnowledgeVersionSnapshot(
        item=KnowledgeItem.model_validate(record.item_snapshot),
        archived_at=archived_at,
        archived_by=record.archived_by,
    )


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _optional_utc(value: datetime | None) -> datetime | None:
    return _as_utc(value) if value is not None else None


def _next_draft_version(version: str) -> str:
    match = re.fullmatch(r"(\d+)\.(\d+)", version)
    if match is None:
        raise ValueError("目前版本格式不支援建立複審新版")
    major, minor = (int(part) for part in match.groups())
    return f"{major}.{minor + 1}-draft"


def _source_from_domain(source: KnowledgeSource) -> KnowledgeSourceRecord:
    return KnowledgeSourceRecord(
        source_id=source.source_id,
        supplied_url=source.supplied_url,
        canonical_url=source.canonical_url,
        title=source.title,
        publisher=source.publisher,
        source_type=source.source_type,
        retrieved_at=source.retrieved_at,
        topics=source.topics,
        status=source.status.value,
        notes=source.notes,
    )


def _item_from_domain(item: KnowledgeItem, *, now: datetime) -> KnowledgeItemRecord:
    record = KnowledgeItemRecord(
        knowledge_id=item.knowledge_id,
        title=item.title,
        standard_answer=item.standard_answer,
        source_id=item.source_id,
        source_uri=item.source_uri,
        source_locator=item.source_locator,
        source_type=item.source_type,
        products=item.products,
        platforms=item.platforms,
        app_versions=item.app_versions,
        effective_at=item.effective_at,
        expires_at=item.expires_at,
        review_at=item.review_at,
        owner_unit=item.owner_unit,
        author=item.author,
        reviewer=item.reviewer,
        approver=item.approver,
        approved_at=item.approved_at,
        version=item.version,
        change_summary=item.change_summary,
        previous_version=item.previous_version,
        status=item.status.value,
        public_answer_allowed=item.public_answer_allowed,
        allowed_intents=item.allowed_intents,
        prohibited_extensions=item.prohibited_extensions,
        row_version=1,
        created_at=now,
        updated_at=now,
    )
    record.question_variants = [
        KnowledgeQuestionVariantRecord(
            variant_id=variant.variant_id,
            knowledge_id=item.knowledge_id,
            question_text=variant.question_text,
            normalized_text=_normalize_question(variant.question_text),
            usage=variant.usage.value,
            position=position,
            created_at=now,
            updated_at=now,
        )
        for position, variant in enumerate(item.question_variants)
    ]
    return record


def _validate_variant_inputs(
    variants: tuple[QuestionVariantInput, ...],
    *,
    existing: list[KnowledgeQuestionVariantRecord],
) -> tuple[tuple[str | None, str, QuestionVariantUsage, str], ...]:
    existing_ids = {variant.variant_id for variant in existing}
    provided_ids = [variant.variant_id for variant in variants if variant.variant_id]
    if len(provided_ids) != len(set(provided_ids)):
        raise ValueError("同一問句變體不得重複提交")
    unknown_ids = set(provided_ids) - existing_ids
    if unknown_ids:
        raise ValueError("提交了不屬於此知識項目的問句變體")

    resolved: list[tuple[str | None, str, QuestionVariantUsage, str]] = []
    normalized_values: set[str] = set()
    for variant in variants:
        validated = QuestionVariant(
            variant_id=variant.variant_id or "new",
            question_text=variant.question_text,
            usage=variant.usage,
        )
        normalized = _normalize_question(validated.question_text)
        if not normalized:
            raise ValueError("問句變體必須包含文字或數字")
        if normalized in normalized_values:
            raise ValueError("問句變體正規化後不得重複")
        normalized_values.add(normalized)
        resolved.append(
            (
                variant.variant_id,
                validated.question_text,
                validated.usage,
                normalized,
            )
        )
    return tuple(resolved)


def _normalize_question(value: str) -> str:
    return re.sub(r"[^0-9a-z\u3400-\u9fff]", "", value.casefold())
