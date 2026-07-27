from datetime import UTC, datetime

import pytest
from sqlalchemy import update

from knowledge_admin.database import KnowledgeSourceRecord
from knowledge_admin.governance import GovernanceAction, GovernanceActor, KnowledgeRole
from knowledge_admin.repository import (
    ASRTermInput,
    DatabaseKnowledgeRepository,
    GovernancePayload,
    QuestionVariantInput,
)
from retrieval import (
    KnowledgeRepositoryError,
    QuestionVariantUsage,
    SqlKnowledgeRepository,
)


def actor(actor_id: str, role: KnowledgeRole) -> GovernanceActor:
    return GovernanceActor(actor_id=actor_id, roles=frozenset({role}))


def test_sql_repository_only_returns_runtime_eligible_documents(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    runtime_repository = SqlKnowledgeRepository(knowledge_store.engine)
    assert runtime_repository.eligible_documents(at=datetime(2026, 7, 20, tzinfo=UTC)) == ()

    current = knowledge_store.get_item("K-CATHAY-DCA-001")
    current = knowledge_store.update_question_variants(
        knowledge_id=current.item.knowledge_id,
        variants=(
            QuestionVariantInput(
                variant_id=None,
                question_text="每月固定投入台股的方式是什麼？",
                usage=QuestionVariantUsage.RETRIEVAL,
            ),
        ),
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=current.row_version,
    )
    current = knowledge_store.update_asr_terms(
        knowledge_id=current.item.knowledge_id,
        terms=(
            ASRTermInput(
                term_id=None,
                canonical_term="台股定期定額",
                aliases=("台股定期定餓",),
            ),
        ),
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=current.row_version,
    )
    submitted = knowledge_store.perform_action(
        knowledge_id=current.item.knowledge_id,
        action=GovernanceAction.SUBMIT_REVIEW,
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=current.row_version,
    )
    reviewed = knowledge_store.perform_action(
        knowledge_id=current.item.knowledge_id,
        action=GovernanceAction.COMPLETE_REVIEW,
        actor=actor("reviewer.dev", KnowledgeRole.REVIEWER),
        expected_version=submitted.row_version,
    )
    approved = knowledge_store.perform_action(
        knowledge_id=current.item.knowledge_id,
        action=GovernanceAction.APPROVE,
        actor=actor("approver.dev", KnowledgeRole.APPROVER),
        expected_version=reviewed.row_version,
        payload=GovernancePayload(
            effective_at=datetime(2026, 7, 20, tzinfo=UTC),
            review_at=datetime(2026, 10, 20, tzinfo=UTC),
            owner_unit="數位通路處",
        ),
    )
    knowledge_store.perform_action(
        knowledge_id=current.item.knowledge_id,
        action=GovernanceAction.PUBLISH,
        actor=actor("publisher.dev", KnowledgeRole.PUBLISHER),
        expected_version=approved.row_version,
    )

    eligible = runtime_repository.eligible_documents(at=datetime(2026, 7, 21, tzinfo=UTC))
    overdue = runtime_repository.eligible_documents(at=datetime(2026, 10, 21, tzinfo=UTC))

    assert [document.item.knowledge_id for document in eligible] == ["K-CATHAY-DCA-001"]
    assert eligible[0].source.source_id == "SRC-CATHAY-DCA-001"
    assert eligible[0].item.question_variants[0].question_text == (
        "每月固定投入台股的方式是什麼？"
    )
    assert eligible[0].item.asr_terms[0].canonical_term == "台股定期定額"
    assert eligible[0].item.asr_terms[0].aliases == ["台股定期定餓"]
    assert overdue == ()

    with knowledge_store.engine.begin() as connection:
        connection.execute(
            update(KnowledgeSourceRecord)
            .where(KnowledgeSourceRecord.source_id == "SRC-CATHAY-DCA-001")
            .values(canonical_url="http://invalid.example")
        )

    with pytest.raises(KnowledgeRepositoryError):
        runtime_repository.eligible_documents(at=datetime(2026, 7, 21, tzinfo=UTC))
