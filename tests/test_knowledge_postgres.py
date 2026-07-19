import os
from datetime import UTC, datetime

import pytest

from answer_contract import TurnRequest
from knowledge_admin.governance import GovernanceAction, GovernanceActor, KnowledgeRole
from knowledge_admin.repository import DatabaseKnowledgeRepository, GovernancePayload
from orchestrator.service import TurnService
from retrieval import KnowledgeStatus, SqlKnowledgeRepository


def actor(actor_id: str, role: KnowledgeRole) -> GovernanceActor:
    return GovernanceActor(actor_id=actor_id, roles=frozenset({role}))


def test_postgres_complete_governance_flow() -> None:
    database_url = os.environ.get("SVA_POSTGRES_TEST_URL")
    if database_url is None:
        pytest.skip("SVA_POSTGRES_TEST_URL 未設定")

    repository = DatabaseKnowledgeRepository.from_url(database_url)
    current = repository.get_item("K-CATHAY-DCA-001")
    assert current.item.status is KnowledgeStatus.DRAFT

    submitted = repository.perform_action(
        knowledge_id=current.item.knowledge_id,
        action=GovernanceAction.SUBMIT_REVIEW,
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=current.row_version,
    )
    reviewed = repository.perform_action(
        knowledge_id=current.item.knowledge_id,
        action=GovernanceAction.COMPLETE_REVIEW,
        actor=actor("reviewer.dev", KnowledgeRole.REVIEWER),
        expected_version=submitted.row_version,
    )
    approved = repository.perform_action(
        knowledge_id=current.item.knowledge_id,
        action=GovernanceAction.APPROVE,
        actor=actor("approver.dev", KnowledgeRole.APPROVER),
        expected_version=reviewed.row_version,
        payload=GovernancePayload(
            effective_at=datetime(2026, 7, 20, tzinfo=UTC),
            review_at=datetime(2026, 10, 20, tzinfo=UTC),
            owner_unit="PostgreSQL integration test",
        ),
    )
    published = repository.perform_action(
        knowledge_id=current.item.knowledge_id,
        action=GovernanceAction.PUBLISH,
        actor=actor("publisher.dev", KnowledgeRole.PUBLISHER),
        expected_version=approved.row_version,
    )

    assert published.item.status is KnowledgeStatus.PUBLISHED
    assert published.item.version == "1.0"
    assert published.item.public_answer_allowed is True
    assert [event.action for event in repository.list_events(current.item.knowledge_id)] == [
        GovernanceAction.SUBMIT_REVIEW,
        GovernanceAction.COMPLETE_REVIEW,
        GovernanceAction.APPROVE,
        GovernanceAction.PUBLISH,
    ]

    runtime_repository = SqlKnowledgeRepository.from_url(database_url)
    service = TurnService(
        knowledge_repository=runtime_repository,
        clock=lambda: datetime(2026, 7, 21, tzinfo=UTC),
    )
    response = service.evaluate(TurnRequest(transcript="什麼是台股定期定額？", channel="web"))

    assert response.result.decision.value == "answer"
    assert response.result.answer_id == current.item.knowledge_id
    assert response.result.citations[0].source_id == "SRC-CATHAY-DCA-001"

    revision = repository.perform_action(
        knowledge_id=current.item.knowledge_id,
        action=GovernanceAction.START_REVISION,
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=published.row_version,
        payload=GovernancePayload(reason="PostgreSQL 複審新版測試"),
        now=datetime(2026, 10, 21, tzinfo=UTC),
    )

    assert revision.item.status is KnowledgeStatus.DRAFT
    assert revision.item.version == "1.1-draft"
    assert repository.list_versions(current.item.knowledge_id)[0].item.version == "1.0"
    assert runtime_repository.eligible_documents(at=datetime(2026, 10, 21, tzinfo=UTC)) == ()
