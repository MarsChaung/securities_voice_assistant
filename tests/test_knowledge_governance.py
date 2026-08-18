from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from knowledge_admin.governance import (
    GovernanceAction,
    GovernanceActor,
    GovernanceError,
    GovernancePolicy,
    KnowledgeRole,
)
from retrieval import KnowledgeItem, KnowledgeSource, KnowledgeStatus


def make_item(**changes: object) -> KnowledgeItem:
    values: dict[str, object] = {
        "knowledge_id": "K-TEST-001",
        "title": "測試知識",
        "standard_answer": "測試答案",
        "source_id": "SRC-TEST-001",
        "source_uri": "https://example.com/source",
        "source_locator": "測試段落",
        "source_type": "official_web",
        "author": "author-1",
        "version": "0.1-draft",
        "status": "draft",
        "public_answer_allowed": False,
        "allowed_intents": ["general_securities_knowledge"],
        "prohibited_extensions": ["不得延伸為個人化投資建議"],
    }
    values.update(changes)
    return KnowledgeItem.model_validate(values)


def actor(actor_id: str, *roles: KnowledgeRole) -> GovernanceActor:
    return GovernanceActor(actor_id=actor_id, roles=frozenset(roles))


def test_local_import_source_and_item_allow_empty_formal_url() -> None:
    source = KnowledgeSource(
        source_id="SRC-LOCAL-001",
        supplied_url=None,
        canonical_url=None,
        title="本機 FAQ 匯入",
        publisher="客戶服務處",
        source_type="local_import",
        retrieved_at=datetime(2026, 7, 24, tzinfo=UTC),
        topics=["內部 FAQ"],
    )
    item = make_item(
        source_id=source.source_id,
        source_uri=None,
        source_type="local_import",
    )

    assert source.canonical_url is None
    assert item.source_uri is None


def test_non_local_source_still_requires_formal_url() -> None:
    with pytest.raises(ValidationError, match="必須提供 HTTPS URL"):
        KnowledgeSource(
            source_id="SRC-FAQ-001",
            supplied_url=None,
            canonical_url=None,
            title="核准內部 FAQ",
            publisher="客戶服務處",
            source_type="approved_internal_faq",
            retrieved_at=datetime(2026, 7, 24, tzinfo=UTC),
            topics=["內部 FAQ"],
        )


def test_author_can_submit_own_draft_for_review() -> None:
    item = make_item()

    next_status = GovernancePolicy.next_status(
        item,
        GovernanceAction.SUBMIT_REVIEW,
        actor("author-1", KnowledgeRole.AUTHOR),
    )

    assert next_status is KnowledgeStatus.REVIEW


def test_other_author_cannot_submit_draft() -> None:
    item = make_item()

    with pytest.raises(GovernanceError, match="原作者"):
        GovernancePolicy.next_status(
            item,
            GovernanceAction.SUBMIT_REVIEW,
            actor("author-2", KnowledgeRole.AUTHOR),
        )


@pytest.mark.parametrize("actor_id", ["author-1", "reviewer-1"])
def test_author_or_reviewer_cannot_approve(actor_id: str) -> None:
    item = make_item(status="review", reviewer="reviewer-1")

    with pytest.raises(GovernanceError, match="不得兼任"):
        GovernancePolicy.next_status(
            item,
            GovernanceAction.APPROVE,
            actor(actor_id, KnowledgeRole.APPROVER),
        )


def test_distinct_approver_can_approve_reviewed_item() -> None:
    item = make_item(status="review", reviewer="reviewer-1")

    next_status = GovernancePolicy.next_status(
        item,
        GovernanceAction.APPROVE,
        actor("approver-1", KnowledgeRole.APPROVER),
    )

    assert next_status is KnowledgeStatus.APPROVED


def test_distinct_reviewer_can_complete_review() -> None:
    item = make_item(status="review")

    next_status = GovernancePolicy.next_status(
        item,
        GovernanceAction.COMPLETE_REVIEW,
        actor("reviewer-1", KnowledgeRole.REVIEWER),
    )

    assert next_status is KnowledgeStatus.REVIEW


def test_approval_requires_assigned_reviewer() -> None:
    item = make_item(status="review")

    with pytest.raises(GovernanceError, match="審核人"):
        GovernancePolicy.next_status(
            item,
            GovernanceAction.APPROVE,
            actor("approver-1", KnowledgeRole.APPROVER),
        )


def test_publisher_must_be_independent() -> None:
    item = make_item(
        status="approved",
        reviewer="reviewer-1",
        approver="approver-1",
        effective_at="2026-07-18T00:00:00+08:00",
        review_at="2026-10-18T00:00:00+08:00",
        owner_unit="數位通路處",
        approved_at="2026-07-18T00:00:00+08:00",
    )

    with pytest.raises(GovernanceError, match="獨立"):
        GovernancePolicy.next_status(
            item,
            GovernanceAction.PUBLISH,
            actor("approver-1", KnowledgeRole.PUBLISHER),
        )


def test_invalid_transition_is_rejected() -> None:
    item = make_item()

    with pytest.raises(GovernanceError, match="不允許"):
        GovernancePolicy.next_status(
            item,
            GovernanceAction.PUBLISH,
            actor("publisher-1", KnowledgeRole.PUBLISHER),
        )


def test_original_author_can_start_revision_for_published_item() -> None:
    item = make_item(
        status="published",
        public_answer_allowed=True,
        reviewer="reviewer-1",
        approver="approver-1",
        effective_at="2026-07-18T00:00:00+08:00",
        review_at="2026-10-18T00:00:00+08:00",
        owner_unit="數位通路處",
        approved_at="2026-07-18T00:00:00+08:00",
        version="1.0",
    )

    next_status = GovernancePolicy.next_status(
        item,
        GovernanceAction.START_REVISION,
        actor("author-1", KnowledgeRole.AUTHOR),
    )

    assert next_status is KnowledgeStatus.DRAFT


def test_other_author_cannot_start_revision() -> None:
    item = make_item(
        status="published",
        public_answer_allowed=True,
        reviewer="reviewer-1",
        approver="approver-1",
        effective_at="2026-07-18T00:00:00+08:00",
        review_at="2026-10-18T00:00:00+08:00",
        owner_unit="數位通路處",
        approved_at="2026-07-18T00:00:00+08:00",
        version="1.0",
    )

    with pytest.raises(GovernanceError, match="原作者"):
        GovernancePolicy.next_status(
            item,
            GovernanceAction.START_REVISION,
            actor("author-2", KnowledgeRole.AUTHOR),
        )
