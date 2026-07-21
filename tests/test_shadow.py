from threading import Event

from observability import SafeAuditLogger
from orchestrator.answering import AnswerEvidence, GeneratedAnswer
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


def test_shadow_runner_returns_before_generation_and_deduplicates_evidence() -> None:
    composer = BlockingComposer()
    audit = CapturingAuditLogger()
    runner = ThreadedShadowAnswerRunner(composer=composer, audit_logger=audit)
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
