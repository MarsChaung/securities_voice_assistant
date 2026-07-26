from threading import Event

from observability import SafeAuditLogger, ShadowReviewInput
from orchestrator.answering import AnswerEvidence, AnswerGenerationError, GeneratedAnswer
from orchestrator.shadow import (
    ShadowAnswerTask,
    ShadowSubmitStatus,
    ThreadedShadowAnswerRunner,
)


def evidence() -> AnswerEvidence:
    return AnswerEvidence(
        standard_answer="美股交易可依官方規則使用新臺幣或美元交割。",
        prohibited_extensions=("不得查詢個人帳戶餘額",),
        knowledge_id="K-CATHAY-US-002",
        knowledge_version="1.1",
        source_id="SRC-CATHAY-USSTOCK-001",
    )


class CapturingAuditLogger(SafeAuditLogger):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, object]] = []

    def shadow_generation(self, **fields: object) -> None:
        self.events.append(fields)


class BlockingComposer:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def compose(self, item: AnswerEvidence) -> GeneratedAnswer:
        self.started.set()
        if not self.release.wait(timeout=2):
            raise AssertionError("test did not release background composer")
        return GeneratedAnswer(
            answer="簡單說，美股可使用新臺幣或美元交割。",
            model_id="synthetic-model",
            prompt_version="controlled-answer-v4",
            prompt_hash="a" * 64,
            latency_ms=12.5,
        )


class CapturingReviewWriter:
    def __init__(self) -> None:
        self.items: list[ShadowReviewInput] = []

    def record(self, item: ShadowReviewInput) -> object:
        self.items.append(item)
        return object()


class FailingComposer:
    def compose(self, item: AnswerEvidence) -> GeneratedAnswer:
        raise AnswerGenerationError("synthetic failure")


def test_shadow_runner_returns_before_generation_and_deduplicates_evidence() -> None:
    composer = BlockingComposer()
    audit = CapturingAuditLogger()
    reviews = CapturingReviewWriter()
    runner = ThreadedShadowAnswerRunner(
        composer=composer,
        audit_logger=audit,
        review_writer=reviews,
    )
    task = ShadowAnswerTask(turn_id="turn-1", evidence=evidence())

    assert runner.submit(task) is ShadowSubmitStatus.QUEUED
    assert composer.started.wait(timeout=1)
    assert runner.submit(task) is ShadowSubmitStatus.CACHED
    assert audit.events == []

    composer.release.set()
    runner.close()

    assert len(audit.events) == 1
    assert audit.events[0]["turn_id"] == "turn-1"
    assert audit.events[0]["output_guard_safe"] is True
    assert audit.events[0]["fallback_reason"] == "shadow_only"
    assert "generated_answer" not in audit.events[0]
    assert len(reviews.items) == 1
    assert reviews.items[0].knowledge_id == "K-CATHAY-US-002"
    assert reviews.items[0].generated_answer == (
        "簡單說，美股可使用新臺幣或美元交割。"
    )


def test_shadow_runner_records_generation_error_without_question_text() -> None:
    audit = CapturingAuditLogger()
    reviews = CapturingReviewWriter()
    runner = ThreadedShadowAnswerRunner(
        composer=FailingComposer(),
        audit_logger=audit,
        review_writer=reviews,
    )

    assert runner.submit(ShadowAnswerTask(turn_id="turn-error", evidence=evidence())) is (
        ShadowSubmitStatus.QUEUED
    )
    runner.close()

    assert len(reviews.items) == 1
    assert reviews.items[0].generated_answer is None
    assert reviews.items[0].fallback_reason == "generation_error"
    assert reviews.items[0].standard_answer == evidence().standard_answer
    assert audit.events[0]["fallback_reason"] == "generation_error"
