from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect

from knowledge_admin.governance import GovernanceAction, GovernanceActor, KnowledgeRole
from knowledge_admin.repository import (
    ASRTermInput,
    ConcurrentUpdateError,
    DatabaseKnowledgeRepository,
    GovernancePayload,
    QuestionVariantInput,
)
from retrieval import KnowledgeStatus, QuestionVariantUsage

ROOT = Path(__file__).parents[1]


def actor(actor_id: str, role: KnowledgeRole) -> GovernanceActor:
    return GovernanceActor(actor_id=actor_id, roles=frozenset({role}))


def publish_for_revision(
    repository: DatabaseKnowledgeRepository,
    *,
    knowledge_id: str = "K-CATHAY-DCA-001",
) -> None:
    current = repository.get_item(knowledge_id)
    submitted = repository.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.SUBMIT_REVIEW,
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=current.row_version,
        now=datetime(2026, 7, 18, tzinfo=UTC),
    )
    reviewed = repository.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.COMPLETE_REVIEW,
        actor=actor("reviewer.dev", KnowledgeRole.REVIEWER),
        expected_version=submitted.row_version,
        now=datetime(2026, 7, 18, tzinfo=UTC),
    )
    approved = repository.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.APPROVE,
        actor=actor("approver.dev", KnowledgeRole.APPROVER),
        expected_version=reviewed.row_version,
        payload=GovernancePayload(
            effective_at=datetime(2026, 7, 18, tzinfo=UTC),
            review_at=datetime(2026, 7, 20, tzinfo=UTC),
            owner_unit="數位通路處",
        ),
        now=datetime(2026, 7, 18, tzinfo=UTC),
    )
    repository.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.PUBLISH,
        actor=actor("publisher.dev", KnowledgeRole.PUBLISHER),
        expected_version=approved.row_version,
        now=datetime(2026, 7, 18, tzinfo=UTC),
    )


def test_seed_is_idempotent(knowledge_store: DatabaseKnowledgeRepository) -> None:
    assert len(knowledge_store.list_sources()) == 4
    assert len(knowledge_store.list_items()) == 15


def test_author_can_add_edit_and_delete_question_variants_in_draft(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    knowledge_id = "K-CATHAY-DCA-001"
    created = knowledge_store.update_question_variants(
        knowledge_id=knowledge_id,
        variants=(
            QuestionVariantInput(
                variant_id=None,
                question_text="什麼是台股定期定額？",
                usage=QuestionVariantUsage.RETRIEVAL,
            ),
            QuestionVariantInput(
                variant_id=None,
                question_text="每個月固定買台股是什麼？",
                usage=QuestionVariantUsage.EVALUATION_ONLY,
            ),
        ),
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=1,
        now=datetime(2026, 7, 19, tzinfo=UTC),
    )

    assert created.row_version == 2
    assert [variant.question_text for variant in created.item.question_variants] == [
        "什麼是台股定期定額？",
        "每個月固定買台股是什麼？",
    ]
    assert created.item.question_variants[1].usage is QuestionVariantUsage.EVALUATION_ONLY
    first_id = created.item.question_variants[0].variant_id

    revised = knowledge_store.update_question_variants(
        knowledge_id=knowledge_id,
        variants=(
            QuestionVariantInput(
                variant_id=first_id,
                question_text="台股定期定額是什麼？",
                usage=QuestionVariantUsage.EXCLUDED,
            ),
            QuestionVariantInput(
                variant_id=None,
                question_text="固定每月投入台股的方式是什麼？",
                usage=QuestionVariantUsage.RETRIEVAL,
            ),
        ),
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=created.row_version,
        now=datetime(2026, 7, 20, tzinfo=UTC),
    )

    assert revised.row_version == 3
    assert revised.item.question_variants[0].variant_id == first_id
    assert revised.item.question_variants[0].usage is QuestionVariantUsage.EXCLUDED
    assert len(revised.item.question_variants) == 2
    assert knowledge_store.list_events(knowledge_id)[-1].reason == (
        "問句變體：新增 1、修改 1、刪除 1"
    )


def test_normalized_duplicate_question_variants_are_rejected(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    with pytest.raises(ValueError, match="正規化後不得重複"):
        knowledge_store.update_question_variants(
            knowledge_id="K-CATHAY-DCA-001",
            variants=(
                QuestionVariantInput(
                    variant_id=None,
                    question_text="線上開戶？",
                    usage=QuestionVariantUsage.RETRIEVAL,
                ),
                QuestionVariantInput(
                    variant_id=None,
                    question_text="線上 開戶",
                    usage=QuestionVariantUsage.RETRIEVAL,
                ),
            ),
            actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
            expected_version=1,
        )

    inserted = knowledge_store.seed_from_files(ROOT / "knowledge")

    assert inserted == (0, 0)
    assert len(knowledge_store.list_items()) == 15


def test_author_can_manage_versioned_asr_terms(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    knowledge_id = "K-CATHAY-DCA-001"
    created = knowledge_store.update_asr_terms(
        knowledge_id=knowledge_id,
        terms=(
            ASRTermInput(
                term_id=None,
                canonical_term="假除權息",
                aliases=("甲竹全席", "甲雛全息"),
            ),
        ),
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=1,
        now=datetime(2026, 7, 19, tzinfo=UTC),
    )

    assert created.row_version == 2
    assert created.item.asr_terms[0].canonical_term == "假除權息"
    assert created.item.asr_terms[0].aliases == ["甲竹全席", "甲雛全息"]
    assert knowledge_store.list_events(knowledge_id)[-1].reason == (
        "ASR 詞彙：新增 1、修改 0、刪除 0"
    )

    term_id = created.item.asr_terms[0].term_id
    revised = knowledge_store.update_asr_terms(
        knowledge_id=knowledge_id,
        terms=(
            ASRTermInput(
                term_id=term_id,
                canonical_term="假除權息",
                aliases=("甲竹全席",),
            ),
            ASRTermInput(
                term_id=None,
                canonical_term="複委託",
                aliases=("副委託",),
            ),
        ),
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=created.row_version,
    )

    assert [term.canonical_term for term in revised.item.asr_terms] == [
        "假除權息",
        "複委託",
    ]
    assert knowledge_store.list_events(knowledge_id)[-1].reason == (
        "ASR 詞彙：新增 1、修改 1、刪除 0"
    )


def test_asr_alias_conflict_is_blocked_when_publishing(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    author = actor("Codex-assisted draft import", KnowledgeRole.AUTHOR)
    first_id = "K-CATHAY-DCA-001"
    second_id = "K-CATHAY-DCA-002"
    knowledge_store.update_asr_terms(
        knowledge_id=first_id,
        terms=(
            ASRTermInput(
                term_id=None,
                canonical_term="假除權息",
                aliases=("甲竹全席",),
            ),
        ),
        actor=author,
        expected_version=1,
    )
    knowledge_store.update_asr_terms(
        knowledge_id=second_id,
        terms=(
            ASRTermInput(
                term_id=None,
                canonical_term="除權息參考價",
                aliases=("甲竹全席",),
            ),
        ),
        actor=author,
        expected_version=1,
    )
    publish_for_revision(knowledge_store, knowledge_id=first_id)

    with pytest.raises(ValueError, match=r"甲竹全席.*K-CATHAY-DCA-001"):
        publish_for_revision(knowledge_store, knowledge_id=second_id)

    assert knowledge_store.get_item(second_id).item.status is KnowledgeStatus.APPROVED


def test_complete_governance_flow_is_persisted(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    knowledge_id = "K-CATHAY-DCA-001"

    submitted = knowledge_store.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.SUBMIT_REVIEW,
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=1,
    )
    reviewed = knowledge_store.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.COMPLETE_REVIEW,
        actor=actor("reviewer.dev", KnowledgeRole.REVIEWER),
        expected_version=submitted.row_version,
    )
    approved = knowledge_store.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.APPROVE,
        actor=actor("approver.dev", KnowledgeRole.APPROVER),
        expected_version=reviewed.row_version,
        payload=GovernancePayload(
            effective_at=datetime(2026, 7, 20, tzinfo=UTC),
            review_at=datetime(2026, 10, 20, tzinfo=UTC),
            owner_unit="數位通路處",
        ),
        now=datetime(2026, 7, 19, tzinfo=UTC),
    )
    published = knowledge_store.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.PUBLISH,
        actor=actor("publisher.dev", KnowledgeRole.PUBLISHER),
        expected_version=approved.row_version,
    )

    assert published.item.status is KnowledgeStatus.PUBLISHED
    assert published.item.public_answer_allowed is True
    assert published.item.version == "1.0"
    assert published.item.previous_version == "0.1-draft"
    assert published.row_version == 5
    events = knowledge_store.list_events(knowledge_id)
    assert [event.action for event in events] == [
        GovernanceAction.SUBMIT_REVIEW,
        GovernanceAction.COMPLETE_REVIEW,
        GovernanceAction.APPROVE,
        GovernanceAction.PUBLISH,
    ]
    assert all(event.occurred_at.tzinfo is not None for event in events)


def test_app_knowledge_requires_app_versions_before_approval(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    knowledge_id = "K-CATHAY-DCA-004"
    submitted = knowledge_store.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.SUBMIT_REVIEW,
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=1,
    )
    reviewed = knowledge_store.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.COMPLETE_REVIEW,
        actor=actor("reviewer.dev", KnowledgeRole.REVIEWER),
        expected_version=submitted.row_version,
    )
    payload = GovernancePayload(
        effective_at=datetime(2026, 7, 20, tzinfo=UTC),
        review_at=datetime(2026, 10, 20, tzinfo=UTC),
        owner_unit="數位通路處",
    )

    with pytest.raises(ValueError, match="App knowledge"):
        knowledge_store.perform_action(
            knowledge_id=knowledge_id,
            action=GovernanceAction.APPROVE,
            actor=actor("approver.dev", KnowledgeRole.APPROVER),
            expected_version=reviewed.row_version,
            payload=payload,
        )

    current = knowledge_store.get_item(knowledge_id)
    assert current.item.status is KnowledgeStatus.REVIEW
    assert current.row_version == reviewed.row_version


def test_approval_requires_future_review_deadline(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    knowledge_id = "K-CATHAY-DCA-001"
    submitted = knowledge_store.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.SUBMIT_REVIEW,
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=1,
        now=datetime(2026, 7, 19, tzinfo=UTC),
    )
    reviewed = knowledge_store.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.COMPLETE_REVIEW,
        actor=actor("reviewer.dev", KnowledgeRole.REVIEWER),
        expected_version=submitted.row_version,
        now=datetime(2026, 7, 19, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="晚於目前時間"):
        knowledge_store.perform_action(
            knowledge_id=knowledge_id,
            action=GovernanceAction.APPROVE,
            actor=actor("approver.dev", KnowledgeRole.APPROVER),
            expected_version=reviewed.row_version,
            payload=GovernancePayload(
                effective_at=datetime(2026, 7, 19, tzinfo=UTC),
                review_at=datetime(2026, 7, 20, tzinfo=UTC),
                owner_unit="數位通路處",
            ),
            now=datetime(2026, 7, 21, tzinfo=UTC),
        )


def test_stale_row_version_is_rejected(knowledge_store: DatabaseKnowledgeRepository) -> None:
    knowledge_id = "K-CATHAY-DCA-001"
    knowledge_store.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.SUBMIT_REVIEW,
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=1,
    )

    with pytest.raises(ConcurrentUpdateError, match="重新載入"):
        knowledge_store.perform_action(
            knowledge_id=knowledge_id,
            action=GovernanceAction.SUBMIT_REVIEW,
            actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
            expected_version=1,
        )


def test_expired_published_item_can_start_governed_revision(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    knowledge_id = "K-CATHAY-DCA-001"
    knowledge_store.update_question_variants(
        knowledge_id=knowledge_id,
        variants=(
            QuestionVariantInput(
                variant_id=None,
                question_text="台股定期定額是什麼？",
                usage=QuestionVariantUsage.RETRIEVAL,
            ),
        ),
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=1,
        now=datetime(2026, 7, 18, tzinfo=UTC),
    )
    publish_for_revision(knowledge_store, knowledge_id=knowledge_id)
    published = knowledge_store.get_item(knowledge_id)

    revision = knowledge_store.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.START_REVISION,
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=published.row_version,
        payload=GovernancePayload(reason="例行複審並展延複審到期時間"),
        now=datetime(2026, 7, 21, tzinfo=UTC),
    )

    assert revision.item.status is KnowledgeStatus.DRAFT
    assert revision.item.version == "1.1-draft"
    assert revision.item.previous_version == "1.0"
    assert revision.item.change_summary == "例行複審並展延複審到期時間"
    assert revision.item.public_answer_allowed is False
    assert revision.item.review_at is None
    assert revision.item.reviewer is None
    assert revision.item.approver is None

    versions = knowledge_store.list_versions(knowledge_id)
    assert len(versions) == 1
    assert versions[0].item.version == "1.0"
    assert versions[0].item.status is KnowledgeStatus.PUBLISHED
    assert versions[0].item.review_at == datetime(2026, 7, 20, tzinfo=UTC)
    assert versions[0].item.question_variants[0].question_text == "台股定期定額是什麼？"
    assert versions[0].archived_by == "Codex-assisted draft import"
    assert knowledge_store.list_events(knowledge_id)[-1].action is GovernanceAction.START_REVISION

    submitted = knowledge_store.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.SUBMIT_REVIEW,
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=revision.row_version,
        now=datetime(2026, 7, 21, tzinfo=UTC),
    )
    reviewed = knowledge_store.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.COMPLETE_REVIEW,
        actor=actor("reviewer.dev", KnowledgeRole.REVIEWER),
        expected_version=submitted.row_version,
        now=datetime(2026, 7, 21, tzinfo=UTC),
    )
    approved = knowledge_store.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.APPROVE,
        actor=actor("approver.dev", KnowledgeRole.APPROVER),
        expected_version=reviewed.row_version,
        payload=GovernancePayload(
            effective_at=datetime(2026, 7, 21, tzinfo=UTC),
            review_at=datetime(2026, 10, 21, tzinfo=UTC),
            owner_unit="數位通路處",
        ),
        now=datetime(2026, 7, 21, tzinfo=UTC),
    )
    republished = knowledge_store.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.PUBLISH,
        actor=actor("publisher.dev", KnowledgeRole.PUBLISHER),
        expected_version=approved.row_version,
        now=datetime(2026, 7, 21, tzinfo=UTC),
    )

    assert republished.item.version == "1.1"
    assert republished.item.previous_version == "1.0"


def test_published_item_cannot_start_revision_before_review_deadline(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    knowledge_id = "K-CATHAY-DCA-001"
    publish_for_revision(knowledge_store, knowledge_id=knowledge_id)
    published = knowledge_store.get_item(knowledge_id)

    with pytest.raises(ValueError, match="尚未到期"):
        knowledge_store.perform_action(
            knowledge_id=knowledge_id,
            action=GovernanceAction.START_REVISION,
            actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
            expected_version=published.row_version,
            payload=GovernancePayload(reason="提前建立新版"),
            now=datetime(2026, 7, 19, tzinfo=UTC),
        )

    assert knowledge_store.get_item(knowledge_id).item.status is KnowledgeStatus.PUBLISHED
    assert knowledge_store.list_versions(knowledge_id) == ()


def test_revoked_item_can_start_a_governed_revision(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    knowledge_id = "K-CATHAY-DCA-001"
    publish_for_revision(knowledge_store, knowledge_id=knowledge_id)
    published = knowledge_store.get_item(knowledge_id)
    revoked = knowledge_store.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.REVOKE,
        actor=actor("revoker.dev", KnowledgeRole.REVOKER),
        expected_version=published.row_version,
        payload=GovernancePayload(reason="需要修訂 ASR 詞彙"),
        now=datetime(2026, 7, 19, tzinfo=UTC),
    )

    revised = knowledge_store.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.START_REVOKED_REVISION,
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=revoked.row_version,
        payload=GovernancePayload(reason="新增語音辨識詞彙與 ASR 別名"),
        now=datetime(2026, 7, 19, tzinfo=UTC),
    )

    assert revised.item.status is KnowledgeStatus.DRAFT
    assert revised.item.version == "1.1-draft"
    assert revised.item.previous_version == "1.0"
    assert revised.item.public_answer_allowed is False
    assert revised.item.reviewer is None
    assert revised.item.approver is None
    versions = knowledge_store.list_versions(knowledge_id)
    assert len(versions) == 1
    assert versions[0].item.status is KnowledgeStatus.REVOKED
    assert versions[0].item.version == "1.0"
    assert knowledge_store.list_events(knowledge_id)[-1].action is (
        GovernanceAction.START_REVOKED_REVISION
    )


def test_alembic_migration_creates_governance_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "migration.db"
    monkeypatch.setenv("SVA_DATABASE_URL", f"sqlite+pysqlite:///{database_path}")
    config = Config(ROOT / "alembic.ini")

    command.upgrade(config, "head")

    table_names = set(inspect(knowledge_store_engine(database_path)).get_table_names())
    assert {
        "alembic_version",
        "knowledge_sources",
        "knowledge_items",
        "knowledge_item_versions",
        "knowledge_governance_events",
        "knowledge_question_variants",
        "faq_import_batches",
    } <= table_names
    columns = {
        column["name"]
        for column in inspect(knowledge_store_engine(database_path)).get_columns("knowledge_items")
    }
    assert "asr_terms" in columns


def knowledge_store_engine(database_path: Path) -> Engine:
    return create_engine(f"sqlite+pysqlite:///{database_path}")
