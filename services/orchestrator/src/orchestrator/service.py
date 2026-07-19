from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from answer_contract import AnswerContract, Citation, Decision, TurnRequest, TurnResponse
from observability import SafeAuditLogger
from policy import DomainPolicyEngine, PolicyAction, PolicyResult, SensitiveDataGuard
from retrieval import (
    KnowledgeDocument,
    KnowledgeRepositoryError,
    LexicalKnowledgeRetriever,
    RetrievalMatch,
)

from .answering import (
    AnswerComposer,
    AnswerEvidence,
    AnswerGenerationError,
    AnswerMode,
    ControlledOutputGuard,
)
from .intent_routing import (
    IntentRouter,
    IntentRouterMode,
    IntentRoutingError,
)


class RuntimeKnowledgeRepository(Protocol):
    def eligible_documents(self, *, at: datetime) -> tuple[KnowledgeDocument, ...]: ...


class RuntimeKnowledgeRetriever(Protocol):
    def search(
        self,
        *,
        query: str,
        intent: str,
        documents: Sequence[KnowledgeDocument],
    ) -> RetrievalMatch | None: ...


@dataclass(frozen=True)
class KnowledgeAvailability:
    status: str
    eligible_document_count: int


@dataclass(frozen=True)
class GenerationTrace:
    answer_mode: str
    model_id: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
    latency_ms: float | None = None
    fallback_reason: str | None = None


@dataclass(frozen=True)
class KnowledgeAnswerOutcome:
    result: AnswerContract
    generation: GenerationTrace


@dataclass(frozen=True)
class IntentRoutingTrace:
    mode: str
    model_id: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
    latency_ms: float | None = None
    candidate_intents: tuple[str, ...] = ()
    confidence: float | None = None
    risk_flags: tuple[str, ...] = ()
    applied: bool = False
    fallback_reason: str | None = None


class TurnService:
    def __init__(
        self,
        sensitive_data_guard: SensitiveDataGuard | None = None,
        policy_engine: DomainPolicyEngine | None = None,
        audit_logger: SafeAuditLogger | None = None,
        knowledge_repository: RuntimeKnowledgeRepository | None = None,
        knowledge_retriever: RuntimeKnowledgeRetriever | None = None,
        answer_mode: AnswerMode | str = AnswerMode.EXACT,
        answer_composer: AnswerComposer | None = None,
        output_guard: ControlledOutputGuard | None = None,
        intent_router_mode: IntentRouterMode | str = IntentRouterMode.DISABLED,
        intent_router: IntentRouter | None = None,
        intent_router_minimum_confidence: float = 0.8,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._sensitive_data_guard = sensitive_data_guard or SensitiveDataGuard()
        self._policy_engine = policy_engine or DomainPolicyEngine()
        self._audit_logger = audit_logger or SafeAuditLogger()
        self._knowledge_repository = knowledge_repository
        self._knowledge_retriever = knowledge_retriever or LexicalKnowledgeRetriever()
        self._answer_mode = AnswerMode(answer_mode)
        self._answer_composer = answer_composer
        self._output_guard = output_guard or ControlledOutputGuard()
        self._intent_router_mode = IntentRouterMode(intent_router_mode)
        self._intent_router = intent_router
        self._intent_router_minimum_confidence = intent_router_minimum_confidence
        self._clock = clock or (lambda: datetime.now(UTC))

        if self._answer_mode is AnswerMode.CONTROLLED_LLM and self._answer_composer is None:
            raise ValueError("controlled_llm answer mode requires an answer composer")
        if (
            self._intent_router_mode is not IntentRouterMode.DISABLED
            and self._intent_router is None
        ):
            raise ValueError("enabled intent router mode requires an intent router")
        if not 0 <= self._intent_router_minimum_confidence <= 1:
            raise ValueError("intent router minimum confidence must be between 0 and 1")

    def knowledge_status(self) -> str:
        return self.knowledge_availability().status

    def knowledge_availability(self) -> KnowledgeAvailability:
        if self._knowledge_repository is None:
            return KnowledgeAvailability(status="disabled", eligible_document_count=0)
        try:
            documents = self._knowledge_repository.eligible_documents(at=self._clock())
        except KnowledgeRepositoryError:
            return KnowledgeAvailability(status="unavailable", eligible_document_count=0)
        return KnowledgeAvailability(
            status="connected",
            eligible_document_count=len(documents),
        )

    def record_feedback(self, *, turn_id: str, rating: str) -> None:
        self._audit_logger.turn_feedback(turn_id=turn_id, rating=rating)

    def evaluate(self, request: TurnRequest) -> TurnResponse:
        started_at = perf_counter()
        turn_id = str(uuid4())
        generation = GenerationTrace(answer_mode=self._answer_mode.value)
        intent_routing = IntentRoutingTrace(mode=self._intent_router_mode.value)
        guard_result = self._sensitive_data_guard.scan(request.transcript)

        if guard_result.has_sensitive_data:
            result = AnswerContract(
                decision=Decision.REFUSE,
                intent="sensitive_data_detected",
                policy_rule_id="PII-001",
                answer=(
                    "為保護您的資料安全，請不要提供帳號、密碼、驗證碼或個人資料。"
                    "如需處理個人帳務，請使用官方客服管道。"
                ),
                confidence=1.0,
            )
            self._log_decision(
                turn_id=turn_id,
                request=request,
                result=result,
                started_at=started_at,
                sensitive_data_types=guard_result.detected_types,
                generation=generation,
                intent_routing=intent_routing,
            )
            return TurnResponse(turn_id=turn_id, result=result)

        policy_result = self._policy_engine.classify(request.transcript)
        if self._can_use_intent_router(policy_result):
            policy_result, intent_routing = self._route_intent(
                question=request.transcript,
                deterministic_result=policy_result,
            )
        error_type: str | None = None

        if policy_result.action is PolicyAction.HANDOFF:
            result = AnswerContract(
                decision=Decision.HANDOFF,
                intent=policy_result.intent,
                policy_rule_id=policy_result.policy_rule_id,
                answer="這項需求需要由人工協助，請透過證券公司的官方客服管道辦理。",
                confidence=policy_result.confidence,
            )
        elif policy_result.action is PolicyAction.REFUSE:
            result = AnswerContract(
                decision=Decision.REFUSE,
                intent=policy_result.intent,
                policy_rule_id=policy_result.policy_rule_id,
                answer=(
                    "這項需求不在本服務可回答的範圍內。"
                    "本服務不處理交易、個人帳務或投資建議，請使用官方管道。"
                ),
                confidence=policy_result.confidence,
            )
        else:
            try:
                outcome = self._answer_from_knowledge(
                    request=request,
                    intent=policy_result.intent,
                    policy_rule_id=policy_result.policy_rule_id,
                )
                result = outcome.result
                generation = outcome.generation
            except KnowledgeRepositoryError:
                result = AnswerContract(
                    decision=Decision.REFUSE,
                    intent=policy_result.intent,
                    policy_rule_id="KNO-002",
                    answer="知識服務目前暫時無法使用，請稍後再試或改用官方客服管道。",
                    confidence=1.0,
                )
                error_type = "knowledge_repository_unavailable"

        self._log_decision(
            turn_id=turn_id,
            request=request,
            result=result,
            started_at=started_at,
            error_type=error_type,
            generation=generation,
            intent_routing=intent_routing,
        )
        return TurnResponse(turn_id=turn_id, result=result)

    def _can_use_intent_router(self, policy_result: PolicyResult) -> bool:
        if self._intent_router_mode is IntentRouterMode.DISABLED:
            return False
        if policy_result.action is PolicyAction.HANDOFF:
            return False
        return not policy_result.policy_rule_id.startswith("POL-REFUSE-")

    def _route_intent(
        self,
        *,
        question: str,
        deterministic_result: PolicyResult,
    ) -> tuple[PolicyResult, IntentRoutingTrace]:
        if self._intent_router is None:
            return deterministic_result, IntentRoutingTrace(
                mode=self._intent_router_mode.value,
                fallback_reason="router_unavailable",
            )
        try:
            route = self._intent_router.route(question)
        except IntentRoutingError:
            return deterministic_result, IntentRoutingTrace(
                mode=self._intent_router_mode.value,
                fallback_reason="routing_error",
            )

        classification = route.classification
        trace = IntentRoutingTrace(
            mode=self._intent_router_mode.value,
            model_id=route.model_id,
            prompt_version=route.prompt_version,
            prompt_hash=route.prompt_hash,
            latency_ms=route.latency_ms,
            candidate_intents=tuple(classification.candidate_intents),
            confidence=classification.confidence,
            risk_flags=tuple(classification.risk_flags),
        )
        if self._intent_router_mode is IntentRouterMode.SHADOW:
            return deterministic_result, trace
        if classification.risk_flags:
            is_complaint = "complaint_or_dispute" in classification.risk_flags
            return (
                PolicyResult(
                    action=(PolicyAction.HANDOFF if is_complaint else PolicyAction.REFUSE),
                    intent=("complaint_or_dispute" if is_complaint else "intent_risk_detected"),
                    policy_rule_id=("LLM-HANDOFF-001" if is_complaint else "LLM-RISK-001"),
                    confidence=classification.confidence,
                ),
                replace(trace, applied=True),
            )
        if (
            classification.needs_clarification
            or classification.confidence < self._intent_router_minimum_confidence
        ):
            return deterministic_result, replace(
                trace,
                fallback_reason="low_confidence_or_clarification",
            )

        intent = classification.candidate_intents[0]
        if intent == "unknown":
            return (
                PolicyResult(
                    action=PolicyAction.REFUSE,
                    intent="unknown_or_ambiguous",
                    policy_rule_id="LLM-DEFAULT-DENY",
                    confidence=classification.confidence,
                ),
                replace(trace, applied=True),
            )
        return (
            PolicyResult(
                action=PolicyAction.ALLOW,
                intent=intent,
                policy_rule_id="LLM-ALLOW-001",
                confidence=classification.confidence,
            ),
            replace(trace, applied=True),
        )

    def _answer_from_knowledge(
        self,
        *,
        request: TurnRequest,
        intent: str,
        policy_rule_id: str,
    ) -> KnowledgeAnswerOutcome:
        generation = GenerationTrace(answer_mode=self._answer_mode.value)
        if self._answer_mode is AnswerMode.FIXED_MESSAGE:
            return KnowledgeAnswerOutcome(
                result=AnswerContract(
                    decision=Decision.REFUSE,
                    intent=intent,
                    policy_rule_id="SYS-FIXED-001",
                    answer="目前僅提供固定安全訊息，請改用證券公司的官方客服管道。",
                    confidence=1.0,
                ),
                generation=generation,
            )

        if self._knowledge_repository is None:
            match = None
        else:
            documents = self._knowledge_repository.eligible_documents(at=self._clock())
            match = self._knowledge_retriever.search(
                query=request.transcript,
                intent=intent,
                documents=documents,
            )

        if match is None:
            return KnowledgeAnswerOutcome(
                result=AnswerContract(
                    decision=Decision.REFUSE,
                    intent=intent,
                    policy_rule_id="KNO-001",
                    answer="目前沒有足夠且已核准的知識來源可回答，請改用官方客服管道。",
                    confidence=1.0,
                ),
                generation=generation,
            )

        item = match.document.item
        source = match.document.source
        exact_answer = AnswerContract(
            decision=Decision.ANSWER,
            intent=intent,
            policy_rule_id=policy_rule_id,
            answer_id=item.knowledge_id,
            source_ids=[source.source_id],
            knowledge_versions=[item.version],
            citations=[
                Citation(
                    source_id=source.source_id,
                    source_uri=source.canonical_url,
                    source_title=source.title,
                    source_locator=item.source_locator,
                )
            ],
            answer=item.standard_answer,
            confidence=match.score,
        )
        if self._answer_mode is AnswerMode.EXACT:
            return KnowledgeAnswerOutcome(result=exact_answer, generation=generation)

        if self._answer_composer is None:
            return KnowledgeAnswerOutcome(
                result=exact_answer,
                generation=GenerationTrace(
                    answer_mode=self._answer_mode.value,
                    fallback_reason="composer_unavailable",
                ),
            )

        try:
            generated = self._answer_composer.compose(
                AnswerEvidence(
                    question=request.transcript,
                    standard_answer=item.standard_answer,
                    prohibited_extensions=tuple(item.prohibited_extensions),
                    knowledge_id=item.knowledge_id,
                    knowledge_version=item.version,
                    source_id=source.source_id,
                )
            )
        except AnswerGenerationError:
            return KnowledgeAnswerOutcome(
                result=exact_answer,
                generation=GenerationTrace(
                    answer_mode=self._answer_mode.value,
                    fallback_reason="generation_error",
                ),
            )

        output_guard_result = self._output_guard.validate(
            generated_answer=generated.answer,
            standard_answer=item.standard_answer,
            prohibited_extensions=tuple(item.prohibited_extensions),
        )
        generation = GenerationTrace(
            answer_mode=self._answer_mode.value,
            model_id=generated.model_id,
            prompt_version=generated.prompt_version,
            prompt_hash=generated.prompt_hash,
            latency_ms=generated.latency_ms,
            fallback_reason=(
                None
                if output_guard_result.safe
                else f"output_guard:{output_guard_result.reason}"
            ),
        )
        if not output_guard_result.safe:
            return KnowledgeAnswerOutcome(result=exact_answer, generation=generation)

        return KnowledgeAnswerOutcome(
            result=exact_answer.model_copy(update={"answer": generated.answer}),
            generation=generation,
        )

    def _log_decision(
        self,
        *,
        turn_id: str,
        request: TurnRequest,
        result: AnswerContract,
        started_at: float,
        sensitive_data_types: list[str] | None = None,
        error_type: str | None = None,
        generation: GenerationTrace,
        intent_routing: IntentRoutingTrace,
    ) -> None:
        self._audit_logger.turn_decision(
            turn_id=turn_id,
            decision=result.decision.value,
            intent=result.intent,
            policy_rule_id=result.policy_rule_id,
            source_ids=result.source_ids,
            knowledge_versions=result.knowledge_versions,
            sensitive_data_types=sensitive_data_types,
            input_character_count=len(request.transcript),
            output_character_count=len(result.answer),
            total_latency_ms=(perf_counter() - started_at) * 1_000,
            answer_id=result.answer_id,
            answer_confidence=result.confidence,
            error_type=error_type,
            answer_mode=generation.answer_mode,
            generation_model_id=generation.model_id,
            prompt_version=generation.prompt_version,
            prompt_hash=generation.prompt_hash,
            generation_latency_ms=generation.latency_ms,
            generation_fallback_reason=generation.fallback_reason,
            intent_router_mode=intent_routing.mode,
            intent_router_model_id=intent_routing.model_id,
            intent_prompt_version=intent_routing.prompt_version,
            intent_prompt_hash=intent_routing.prompt_hash,
            intent_router_latency_ms=intent_routing.latency_ms,
            intent_candidate_intents=list(intent_routing.candidate_intents),
            intent_router_confidence=intent_routing.confidence,
            intent_risk_flags=list(intent_routing.risk_flags),
            intent_router_applied=intent_routing.applied,
            intent_router_fallback_reason=intent_routing.fallback_reason,
        )
