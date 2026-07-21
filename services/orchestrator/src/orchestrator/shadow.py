from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from threading import BoundedSemaphore, Lock
from typing import Protocol

from observability import SafeAuditLogger

from .answering import (
    AnswerComposer,
    AnswerEvidence,
    AnswerGenerationError,
    ControlledOutputGuard,
)


@dataclass(frozen=True)
class ShadowAnswerTask:
    turn_id: str
    evidence: AnswerEvidence


class ShadowSubmitStatus(StrEnum):
    QUEUED = "shadow_queued"
    CACHED = "shadow_cached"
    QUEUE_FULL = "shadow_queue_full"


class ShadowAnswerRunner(Protocol):
    def submit(self, task: ShadowAnswerTask) -> ShadowSubmitStatus: ...

    def close(self) -> None: ...


class ThreadedShadowAnswerRunner:
    """Run privacy-safe Shadow generation outside the request path."""

    def __init__(
        self,
        *,
        composer: AnswerComposer,
        audit_logger: SafeAuditLogger | None = None,
        output_guard: ControlledOutputGuard | None = None,
        max_pending: int = 8,
    ) -> None:
        self._composer = composer
        self._audit_logger = audit_logger or SafeAuditLogger()
        self._output_guard = output_guard or ControlledOutputGuard()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shadow-answer")
        self._capacity = BoundedSemaphore(max_pending)
        self._known_evidence: set[tuple[str, str, str]] = set()
        self._known_lock = Lock()

    def submit(self, task: ShadowAnswerTask) -> ShadowSubmitStatus:
        evidence_key = self._evidence_key(task.evidence)
        with self._known_lock:
            if evidence_key in self._known_evidence:
                return ShadowSubmitStatus.CACHED
            if not self._capacity.acquire(blocking=False):
                return ShadowSubmitStatus.QUEUE_FULL
            self._known_evidence.add(evidence_key)

        future = self._executor.submit(self._run, task)
        future.add_done_callback(
            lambda completed: self._release(completed, evidence_key=evidence_key)
        )
        return ShadowSubmitStatus.QUEUED

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _run(self, task: ShadowAnswerTask) -> bool:
        evidence = task.evidence
        try:
            generated = self._composer.compose(evidence)
        except AnswerGenerationError:
            self._audit_logger.shadow_generation(
                turn_id=task.turn_id,
                answer_id=evidence.knowledge_id,
                knowledge_version=evidence.knowledge_version,
                source_id=evidence.source_id,
                output_guard_safe=False,
                fallback_reason="generation_error",
            )
            return False

        guard_result = self._output_guard.validate(
            generated_answer=generated.answer,
            standard_answer=evidence.standard_answer,
            prohibited_extensions=evidence.prohibited_extensions,
        )
        self._audit_logger.shadow_generation(
            turn_id=task.turn_id,
            answer_id=evidence.knowledge_id,
            knowledge_version=evidence.knowledge_version,
            source_id=evidence.source_id,
            generation_model_id=generated.model_id,
            prompt_version=generated.prompt_version,
            prompt_hash=generated.prompt_hash,
            generation_latency_ms=generated.latency_ms,
            output_guard_safe=guard_result.safe,
            fallback_reason=(
                "shadow_only"
                if guard_result.safe
                else f"output_guard:{guard_result.reason}"
            ),
        )
        return True

    def _release(
        self,
        future: Future[bool],
        *,
        evidence_key: tuple[str, str, str],
    ) -> None:
        try:
            completed = future.result()
        except Exception:
            completed = False
        if not completed:
            with self._known_lock:
                self._known_evidence.discard(evidence_key)
        self._capacity.release()

    @staticmethod
    def _evidence_key(evidence: AnswerEvidence) -> tuple[str, str, str]:
        return evidence.knowledge_id, evidence.knowledge_version, evidence.source_id
