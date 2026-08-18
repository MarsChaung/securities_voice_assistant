from datetime import UTC, datetime

import pytest

from knowledge_admin.repository import DatabaseKnowledgeRepository
from observability import (
    DatabaseShadowReviewRepository,
    ShadowReviewConcurrentUpdateError,
    ShadowReviewInput,
    ShadowReviewLabel,
    ShadowReviewStateError,
    ShadowReviewStatus,
)


def shadow_input(
    *,
    generated_answer: str | None = "簡單說，美股交易時段會依夏令與冬令時間調整。",
    fallback_reason: str = "shadow_only",
) -> ShadowReviewInput:
    return ShadowReviewInput(
        turn_id="turn-shadow-1",
        knowledge_id="K-CATHAY-US-003",
        knowledge_version="1.1",
        source_id="SRC-CATHAY-USSTOCK-001",
        standard_answer="美股交易時間會因夏令與冬令時間不同而改變。",
        prohibited_extensions=("不得保證固定成交時間",),
        generated_answer=generated_answer,
        generation_model_id="Qwen3.6-35B-A3B-oQ4" if generated_answer else None,
        prompt_version="controlled-answer-v4" if generated_answer else None,
        prompt_hash="a" * 64 if generated_answer else None,
        generation_latency_ms=1250.0 if generated_answer else None,
        output_guard_safe=True if generated_answer else None,
        fallback_reason=fallback_reason,
    )


def repository(
    knowledge_store: DatabaseKnowledgeRepository,
) -> DatabaseShadowReviewRepository:
    return DatabaseShadowReviewRepository(knowledge_store.engine)


def test_record_deduplicates_same_knowledge_model_and_prompt(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    reviews = repository(knowledge_store)

    first = reviews.record(shadow_input(), now=datetime(2026, 7, 22, 1, 0, tzinfo=UTC))
    duplicate = reviews.record(shadow_input(), now=datetime(2026, 7, 22, 2, 0, tzinfo=UTC))

    assert duplicate.shadow_id == first.shadow_id
    assert len(reviews.list_results()) == 1
    assert first.review_status is ShadowReviewStatus.PENDING
    assert first.generated_answer is not None
    assert first.prohibited_extensions == ("不得保證固定成交時間",)
    metrics = reviews.metrics()
    assert metrics.pending == 1
    assert metrics.output_guard_safe == 1
    assert metrics.average_latency_ms == 1250.0


def test_review_is_immutable_and_updates_metrics(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    reviews = repository(knowledge_store)
    result = reviews.record(shadow_input())

    accepted = reviews.review(
        shadow_id=result.shadow_id,
        label=ShadowReviewLabel.ACCEPTABLE,
        reviewer_id="reviewer.dev",
        reviewer_note="語意完整且未新增內容",
        expected_version=1,
        now=datetime(2026, 7, 22, 3, 0, tzinfo=UTC),
    )

    assert accepted.review_status is ShadowReviewStatus.ACCEPTED
    assert accepted.row_version == 2
    assert accepted.reviewer_id == "reviewer.dev"
    assert reviews.metrics().acceptance_rate == 1.0
    with pytest.raises(ShadowReviewStateError):
        reviews.review(
            shadow_id=result.shadow_id,
            label=ShadowReviewLabel.TONE,
            reviewer_id="reviewer.dev",
            reviewer_note=None,
            expected_version=2,
        )


def test_review_validates_concurrency_and_other_note(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    reviews = repository(knowledge_store)
    result = reviews.record(shadow_input())

    with pytest.raises(ShadowReviewConcurrentUpdateError):
        reviews.review(
            shadow_id=result.shadow_id,
            label=ShadowReviewLabel.INCORRECT,
            reviewer_id="reviewer.dev",
            reviewer_note=None,
            expected_version=99,
        )
    with pytest.raises(ValueError, match="必須填寫複核說明"):
        reviews.review(
            shadow_id=result.shadow_id,
            label=ShadowReviewLabel.OTHER,
            reviewer_id="reviewer.dev",
            reviewer_note=None,
            expected_version=1,
        )


def test_generation_error_is_reported_but_not_reviewable(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    reviews = repository(knowledge_store)

    result = reviews.record(shadow_input(generated_answer=None, fallback_reason="generation_error"))

    assert result.review_status is ShadowReviewStatus.NOT_REVIEWABLE
    assert result.generated_answer is None
    assert reviews.list_results(status=ShadowReviewStatus.PENDING) == ()
    assert reviews.metrics().not_reviewable == 1
