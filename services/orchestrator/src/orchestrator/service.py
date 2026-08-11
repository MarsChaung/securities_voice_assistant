import logging
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
    NaturalAnswerComposer,
    focus_approved_answer,
    select_approved_answer_segments,
)
from .asr import MandarinPhoneticResolver, build_asr_context
from .conversation import ConversationResolution, FollowUpKind, ReplyMode
from .intent_routing import (
    IntentRouter,
    IntentRouterMode,
    IntentRoutingError,
)
from .shadow import ShadowAnswerRunner, ShadowAnswerTask


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
    applied: bool = False
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


@dataclass
class TurnTimingTrace:
    conversation_resolution_latency_ms: float | None = None
    conversation_semantic_latency_ms: float | None = None
    policy_guard_latency_ms: float | None = None
    retrieval_latency_ms: float | None = None


class TurnService:
    _NEW_QUESTION_CONTEXT_BONUS = 0.05

    def __init__(
        self,
        sensitive_data_guard: SensitiveDataGuard | None = None,
        policy_engine: DomainPolicyEngine | None = None,
        audit_logger: SafeAuditLogger | None = None,
        knowledge_repository: RuntimeKnowledgeRepository | None = None,
        knowledge_retriever: RuntimeKnowledgeRetriever | None = None,
        answer_mode: AnswerMode | str = AnswerMode.EXACT,
        answer_composer: AnswerComposer | None = None,
        natural_answer_composer: NaturalAnswerComposer | None = None,
        output_guard: ControlledOutputGuard | None = None,
        shadow_runner: ShadowAnswerRunner | None = None,
        intent_router_mode: IntentRouterMode | str = IntentRouterMode.DISABLED,
        intent_router: IntentRouter | None = None,
        intent_router_minimum_confidence: float = 0.8,
        phonetic_resolver: MandarinPhoneticResolver | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._sensitive_data_guard = sensitive_data_guard or SensitiveDataGuard()
        self._policy_engine = policy_engine or DomainPolicyEngine()
        self._audit_logger = audit_logger or SafeAuditLogger()
        self._knowledge_repository = knowledge_repository
        self._knowledge_retriever = knowledge_retriever or LexicalKnowledgeRetriever()
        self._answer_mode = AnswerMode(answer_mode)
        self._answer_composer = answer_composer
        self._natural_answer_composer = natural_answer_composer
        self._output_guard = output_guard or ControlledOutputGuard()
        self._shadow_runner = shadow_runner
        self._intent_router_mode = IntentRouterMode(intent_router_mode)
        self._intent_router = intent_router
        self._intent_router_minimum_confidence = intent_router_minimum_confidence
        self._phonetic_resolver = phonetic_resolver or MandarinPhoneticResolver()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._retrieval_logger = logging.getLogger("sva.retrieval")

        if self._answer_mode is AnswerMode.CONTROLLED_LLM and self._answer_composer is None:
            raise ValueError("controlled LLM answer mode requires an answer composer")
        if self._answer_mode is AnswerMode.SHADOW_LLM and self._shadow_runner is None:
            raise ValueError("shadow LLM answer mode requires a background runner")
        if (
            self._intent_router_mode is not IntentRouterMode.DISABLED
            and self._intent_router is None
        ):
            raise ValueError("enabled intent router mode requires an intent router")
        if not 0 <= self._intent_router_minimum_confidence <= 1:
            raise ValueError("intent router minimum confidence must be between 0 and 1")

    def knowledge_status(self) -> str:
        return self.knowledge_availability().status

    @property
    def natural_answer_available(self) -> bool:
        return (
            self._natural_answer_composer is not None
            and self._answer_mode is not AnswerMode.FIXED_MESSAGE
        )

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

    def record_voice_playback(
        self,
        *,
        turn_id: str,
        chunk_count: int,
        audio_duration_ms: float,
        initial_buffered_ms: float,
        first_playback_delay_ms: float | None,
        buffer_target_ms: float,
        crossfade_ms: float,
        underrun_count: int,
        underrun_total_ms: float,
        underrun_max_ms: float,
        interrupted: bool,
        interruption_reason: str | None,
        barge_in_mode: str | None,
        barge_in_duck_latency_ms: float | None,
        barge_in_confirm_latency_ms: float | None,
        barge_in_false_trigger_count: int,
        chunk_timings: list[dict[str, float | None]],
    ) -> None:
        self._audit_logger.voice_playback(
            turn_id=turn_id,
            chunk_count=chunk_count,
            audio_duration_ms=audio_duration_ms,
            initial_buffered_ms=initial_buffered_ms,
            first_playback_delay_ms=first_playback_delay_ms,
            buffer_target_ms=buffer_target_ms,
            crossfade_ms=crossfade_ms,
            underrun_count=underrun_count,
            underrun_total_ms=underrun_total_ms,
            underrun_max_ms=underrun_max_ms,
            interrupted=interrupted,
            interruption_reason=interruption_reason,
            barge_in_mode=barge_in_mode,
            barge_in_duck_latency_ms=barge_in_duck_latency_ms,
            barge_in_confirm_latency_ms=barge_in_confirm_latency_ms,
            barge_in_false_trigger_count=barge_in_false_trigger_count,
            chunk_timings=chunk_timings,
        )

    def voice_asr_context(self) -> str:
        if self._knowledge_repository is None:
            return ""
        try:
            documents = self._knowledge_repository.eligible_documents(at=self._clock())
        except KnowledgeRepositoryError:
            return ""
        return build_asr_context(documents)

    def evaluate(
        self,
        request: TurnRequest,
        *,
        conversation: ConversationResolution | None = None,
    ) -> TurnResponse:
        started_at = perf_counter()
        turn_id = str(uuid4())
        generation = GenerationTrace(
            answer_mode=(
                ReplyMode.NATURAL.value if conversation is not None else self._answer_mode.value
            )
        )
        intent_routing = IntentRoutingTrace(mode=self._intent_router_mode.value)
        timing = TurnTimingTrace(
            conversation_resolution_latency_ms=(
                conversation.resolution_latency_ms if conversation is not None else None
            ),
            conversation_semantic_latency_ms=(
                conversation.semantic_latency_ms if conversation is not None else None
            ),
        )
        guard_result = self._sensitive_data_guard.scan(request.transcript)

        if guard_result.has_sensitive_data:
            timing.policy_guard_latency_ms = (perf_counter() - started_at) * 1_000
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
                timing=timing,
            )
            return TurnResponse(turn_id=turn_id, result=result)

        policy_question = request.transcript
        policy_result = self._policy_engine.classify(policy_question)
        if (
            conversation is not None
            and conversation.kind is not FollowUpKind.NEW_QUESTION
            and self._can_apply_conversation_context(policy_result)
        ):
            contextual_result = self._policy_engine.classify(conversation.retrieval_query)
            if contextual_result.action is PolicyAction.ALLOW:
                policy_result = contextual_result
                policy_question = conversation.retrieval_query
        timing.policy_guard_latency_ms = (perf_counter() - started_at) * 1_000
        if self._can_use_intent_router(policy_result):
            policy_result, intent_routing = self._route_intent(
                question=policy_question,
                deterministic_result=policy_result,
            )
        error_type: str | None = None

        if policy_result.action is PolicyAction.HANDOFF:
            result = AnswerContract(
                decision=Decision.HANDOFF,
                intent=policy_result.intent,
                policy_rule_id=policy_result.policy_rule_id,
                answer="很抱歉，這項需求必須由客服協助處理。",
                confidence=policy_result.confidence,
            )
        elif policy_result.action is PolicyAction.REFUSE:
            result = AnswerContract(
                decision=Decision.REFUSE,
                intent=policy_result.intent,
                policy_rule_id=policy_result.policy_rule_id,
                answer="很抱歉，這項需求不在本服務可回答的範圍內。",
                confidence=policy_result.confidence,
            )
        else:
            try:
                outcome = self._answer_from_knowledge(
                    turn_id=turn_id,
                    request=request,
                    intent=policy_result.intent,
                    policy_rule_id=policy_result.policy_rule_id,
                    conversation=conversation,
                    timing=timing,
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
            timing=timing,
        )
        return TurnResponse(turn_id=turn_id, result=result)

    @staticmethod
    def _can_apply_conversation_context(policy_result: PolicyResult) -> bool:
        if policy_result.action is PolicyAction.HANDOFF:
            return False
        return not policy_result.policy_rule_id.startswith("POL-REFUSE-")

    def _can_use_intent_router(self, policy_result: PolicyResult) -> bool:
        if self._intent_router_mode is IntentRouterMode.DISABLED:
            return False
        if policy_result.action is PolicyAction.HANDOFF:
            return False
        if policy_result.intent in {
            "credential_recovery_guidance",
            "account_authorization_guidance",
            "personal_data_change_guidance",
        }:
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
        turn_id: str,
        request: TurnRequest,
        intent: str,
        policy_rule_id: str,
        conversation: ConversationResolution | None,
        timing: TurnTimingTrace,
    ) -> KnowledgeAnswerOutcome:
        generation = GenerationTrace(
            answer_mode=(
                ReplyMode.NATURAL.value if conversation is not None else self._answer_mode.value
            )
        )
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

        retrieval_started_at = perf_counter()
        try:
            if self._knowledge_repository is None:
                documents: tuple[KnowledgeDocument, ...] = ()
                match = None
            else:
                documents = self._knowledge_repository.eligible_documents(at=self._clock())
                match = self._search_knowledge(
                    original_query=request.transcript,
                    intent=intent,
                    documents=documents,
                    conversation=conversation,
                )
                if (
                    match is None
                    and conversation is not None
                    and conversation.kind is not FollowUpKind.NEW_QUESTION
                    and conversation.reference_knowledge_id is not None
                ):
                    reference_document = next(
                        (
                            document
                            for document in documents
                            if document.item.knowledge_id
                            == conversation.reference_knowledge_id
                        ),
                        None,
                    )
                    if reference_document is not None:
                        match = RetrievalMatch(
                            document=reference_document,
                            score=1.0,
                        )
                        policy_rule_id = "CTX-FOLLOW-UP-001"

            if (
                match is None
                and request.channel == "voice"
                and intent == "general_securities_knowledge"
            ):
                phonetic = self._phonetic_resolver.resolve(
                    query=(
                        conversation.retrieval_query
                        if conversation is not None
                        else request.transcript
                    ),
                    intent=intent,
                    documents=documents,
                )
                if phonetic.ambiguous:
                    titles = "或".join(
                        f"「{candidate.document.item.title}」"
                        for candidate in phonetic.candidates
                    )
                    return KnowledgeAnswerOutcome(
                        result=AnswerContract(
                            decision=Decision.CLARIFY,
                            intent=intent,
                            policy_rule_id=(
                                "ASR-ALIAS-002"
                                if phonetic.strategy == "alias"
                                else "ASR-PHONETIC-002"
                            ),
                            answer=(
                                f"我辨識到的內容可能是在詢問{titles}，請再說一次完整問題。"
                            ),
                            confidence=phonetic.candidates[0].score,
                        ),
                        generation=generation,
                    )
                if phonetic.match is not None:
                    match = phonetic.match
                    policy_rule_id = (
                        "ASR-ALIAS-001"
                        if phonetic.strategy == "alias"
                        else "ASR-PHONETIC-001"
                    )
        finally:
            timing.retrieval_latency_ms = (perf_counter() - retrieval_started_at) * 1_000

        if match is None:
            return KnowledgeAnswerOutcome(
                result=AnswerContract(
                    decision=Decision.REFUSE,
                    intent=intent,
                    policy_rule_id="KNO-001",
                    answer="目前沒有足夠且有效的已發布知識來源可回答，請改用官方客服管道。",
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
        evidence = AnswerEvidence(
            standard_answer=item.standard_answer,
            prohibited_extensions=tuple(item.prohibited_extensions),
            knowledge_id=item.knowledge_id,
            knowledge_version=item.version,
            source_id=source.source_id,
        )
        if conversation is not None:
            return self._compose_natural_answer(
                exact_answer=exact_answer,
                evidence=evidence,
                conversation=conversation,
                current_utterance=request.transcript,
            )
        if self._answer_mode is AnswerMode.EXACT:
            return KnowledgeAnswerOutcome(result=exact_answer, generation=generation)

        if self._answer_mode is AnswerMode.SHADOW_LLM:
            if self._shadow_runner is None:
                return KnowledgeAnswerOutcome(
                    result=exact_answer,
                    generation=GenerationTrace(
                        answer_mode=self._answer_mode.value,
                        fallback_reason="shadow_runner_unavailable",
                    ),
                )
            submit_status = self._shadow_runner.submit(
                ShadowAnswerTask(turn_id=turn_id, evidence=evidence)
            )
            return KnowledgeAnswerOutcome(
                result=exact_answer,
                generation=GenerationTrace(
                    answer_mode=self._answer_mode.value,
                    fallback_reason=submit_status.value,
                ),
            )

        if self._answer_composer is None:
            return KnowledgeAnswerOutcome(
                result=exact_answer,
                generation=GenerationTrace(
                    answer_mode=self._answer_mode.value,
                    fallback_reason="composer_unavailable",
                ),
            )

        try:
            generated = self._answer_composer.compose(evidence)
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
            applied=output_guard_result.safe,
            fallback_reason=(
                None if output_guard_result.safe else f"output_guard:{output_guard_result.reason}"
            ),
        )
        if not output_guard_result.safe:
            return KnowledgeAnswerOutcome(result=exact_answer, generation=generation)

        return KnowledgeAnswerOutcome(
            result=exact_answer.model_copy(update={"answer": generated.answer}),
            generation=generation,
        )

    def _search_knowledge(
        self,
        *,
        original_query: str,
        intent: str,
        documents: Sequence[KnowledgeDocument],
        conversation: ConversationResolution | None,
    ) -> RetrievalMatch | None:
        original_match = self._knowledge_retriever.search(
            query=original_query,
            intent=intent,
            documents=documents,
        )
        if (
            conversation is None
            or (
                conversation.retrieval_query == original_query
                and conversation.kind is not FollowUpKind.NEW_QUESTION
            )
        ):
            return original_match
        if conversation.kind is FollowUpKind.NEW_QUESTION:
            reference_exchange = next(
                (
                    exchange
                    for exchange in reversed(conversation.history)
                    if exchange.decision == "answer"
                    and exchange.knowledge_id is not None
                ),
                None,
            )
            if reference_exchange is None or (
                original_match is not None
                and original_match.document.item.knowledge_id
                == reference_exchange.knowledge_id
            ):
                return original_match
            reference_document = next(
                (
                    document
                    for document in documents
                    if document.item.knowledge_id == reference_exchange.knowledge_id
                ),
                None,
            )
            recent_match = (
                self._knowledge_retriever.search(
                    query=original_query,
                    intent=intent,
                    documents=(reference_document,),
                )
                if reference_document is not None
                else None
            )
            selected_match = original_match
            if recent_match is not None and (
                original_match is None
                or recent_match.score + self._NEW_QUESTION_CONTEXT_BONUS
                >= original_match.score
            ):
                selected_match = recent_match
            self._log_conversation_retrieval(
                original_match=original_match,
                contextual_match=None,
                reference_match=recent_match,
                selected_match=selected_match,
                conversation=conversation,
            )
            return selected_match

        contextual_match = self._knowledge_retriever.search(
            query=conversation.retrieval_query,
            intent=intent,
            documents=documents,
        )
        reference_match: RetrievalMatch | None = None
        if conversation.reference_knowledge_id is not None and (
            contextual_match is None
            or contextual_match.document.item.knowledge_id
            != conversation.reference_knowledge_id
        ):
            reference_document = next(
                (
                    document
                    for document in documents
                    if document.item.knowledge_id == conversation.reference_knowledge_id
                ),
                None,
            )
            if reference_document is not None:
                reference_match = self._knowledge_retriever.search(
                    query=conversation.retrieval_query,
                    intent=intent,
                    documents=(reference_document,),
                )

        candidates = (
            (original_match, 0.0, 0),
            (contextual_match, 0.02, 1),
            (reference_match, 0.05, 2),
        )
        available = [
            (match, min(1.0, match.score + bonus), priority)
            for match, bonus, priority in candidates
            if match is not None
        ]
        if not available:
            self._log_conversation_retrieval(
                original_match=original_match,
                contextual_match=contextual_match,
                reference_match=reference_match,
                selected_match=None,
                conversation=conversation,
            )
            return None
        contextual_matches_reference = (
            contextual_match is not None
            and conversation.reference_knowledge_id is not None
            and contextual_match.document.item.knowledge_id
            == conversation.reference_knowledge_id
        )
        if conversation.semantic_applied and contextual_matches_reference:
            selected_match = contextual_match
        elif conversation.semantic_applied and reference_match is not None:
            selected_match = reference_match
        else:
            selected_match = max(
                available,
                key=lambda candidate: (candidate[1], candidate[2]),
            )[0]
        self._log_conversation_retrieval(
            original_match=original_match,
            contextual_match=contextual_match,
            reference_match=reference_match,
            selected_match=selected_match,
            conversation=conversation,
        )
        return selected_match

    def _log_conversation_retrieval(
        self,
        *,
        original_match: RetrievalMatch | None,
        contextual_match: RetrievalMatch | None,
        reference_match: RetrievalMatch | None,
        selected_match: RetrievalMatch | None,
        conversation: ConversationResolution,
    ) -> None:
        def summary(match: RetrievalMatch | None) -> str:
            if match is None:
                return "none"
            return f"{match.document.item.knowledge_id}:{match.score:.4f}"

        self._retrieval_logger.info(
            "conversation_retrieval semantic_applied=%s original=%s contextual=%s "
            "reference=%s selected=%s",
            conversation.semantic_applied,
            summary(original_match),
            summary(contextual_match),
            summary(reference_match),
            summary(selected_match),
        )

    def _compose_natural_answer(
        self,
        *,
        exact_answer: AnswerContract,
        evidence: AnswerEvidence,
        conversation: ConversationResolution,
        current_utterance: str,
    ) -> KnowledgeAnswerOutcome:
        focused_answer = (
            focus_approved_answer(
                standard_answer=evidence.standard_answer,
                current_utterance=current_utterance,
            )
            if conversation.kind is FollowUpKind.ELABORATE
            else None
        )
        generation_evidence = (
            replace(evidence, standard_answer=focused_answer)
            if focused_answer is not None
            else evidence
        )
        fallback_result = (
            exact_answer.model_copy(update={"answer": focused_answer})
            if focused_answer is not None
            else exact_answer
        )
        if self._natural_answer_composer is None:
            return KnowledgeAnswerOutcome(
                result=fallback_result,
                generation=GenerationTrace(
                    answer_mode=ReplyMode.NATURAL.value,
                    fallback_reason="natural_composer_unavailable",
                ),
            )
        try:
            generated = self._natural_answer_composer.compose(
                generation_evidence,
                current_utterance=current_utterance,
                follow_up_kind=conversation.kind,
                history=(
                    ()
                    if conversation.kind is FollowUpKind.NEW_QUESTION
                    else conversation.history[-4:]
                ),
            )
        except AnswerGenerationError:
            return KnowledgeAnswerOutcome(
                result=fallback_result,
                generation=GenerationTrace(
                    answer_mode=ReplyMode.NATURAL.value,
                    fallback_reason="generation_error",
                ),
            )

        selected_answer = select_approved_answer_segments(
            standard_answer=generation_evidence.standard_answer,
            segment_ids=generated.selected_segment_ids,
        )
        if generated.selected_segment_ids and selected_answer is None:
            return KnowledgeAnswerOutcome(
                result=fallback_result,
                generation=GenerationTrace(
                    answer_mode=ReplyMode.NATURAL.value,
                    model_id=generated.model_id,
                    prompt_version=generated.prompt_version,
                    prompt_hash=generated.prompt_hash,
                    latency_ms=generated.latency_ms,
                    fallback_reason="evidence_selection_invalid",
                ),
            )
        if selected_answer is not None:
            generation_evidence = replace(
                generation_evidence,
                standard_answer=selected_answer,
            )
            fallback_result = exact_answer.model_copy(update={"answer": selected_answer})

        output_guard_result = self._output_guard.validate(
            generated_answer=generated.answer,
            standard_answer=generation_evidence.standard_answer,
            prohibited_extensions=generation_evidence.prohibited_extensions,
        )
        generation = GenerationTrace(
            answer_mode=ReplyMode.NATURAL.value,
            model_id=generated.model_id,
            prompt_version=generated.prompt_version,
            prompt_hash=generated.prompt_hash,
            latency_ms=generated.latency_ms,
            applied=output_guard_result.safe,
            fallback_reason=(
                None if output_guard_result.safe else f"output_guard:{output_guard_result.reason}"
            ),
        )
        if not output_guard_result.safe:
            return KnowledgeAnswerOutcome(result=fallback_result, generation=generation)
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
        timing: TurnTimingTrace,
    ) -> None:
        total_latency_ms = (perf_counter() - started_at) * 1_000
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
            total_latency_ms=total_latency_ms,
            end_to_end_latency_ms=(
                total_latency_ms + (timing.conversation_resolution_latency_ms or 0)
            ),
            conversation_resolution_latency_ms=(
                timing.conversation_resolution_latency_ms
            ),
            conversation_semantic_latency_ms=timing.conversation_semantic_latency_ms,
            policy_guard_latency_ms=timing.policy_guard_latency_ms,
            retrieval_latency_ms=timing.retrieval_latency_ms,
            answer_id=result.answer_id,
            answer_confidence=result.confidence,
            error_type=error_type,
            answer_mode=generation.answer_mode,
            generation_model_id=generation.model_id,
            prompt_version=generation.prompt_version,
            prompt_hash=generation.prompt_hash,
            generation_latency_ms=generation.latency_ms,
            generation_applied=generation.applied,
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
