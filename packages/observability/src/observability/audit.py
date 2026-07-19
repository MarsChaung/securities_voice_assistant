import json
import logging
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TurnDecisionEvent:
    schema_version: str = field(init=False, default="1.3")
    turn_id: str
    decision: str
    intent: str
    policy_rule_id: str
    input_character_count: int
    output_character_count: int
    total_latency_ms: float
    answer_id: str | None = None
    answer_confidence: float | None = None
    source_ids: tuple[str, ...] = field(default_factory=tuple)
    knowledge_versions: tuple[str, ...] = field(default_factory=tuple)
    sensitive_data_types: tuple[str, ...] = field(default_factory=tuple)
    error_type: str | None = None
    answer_mode: str = "exact"
    generation_model_id: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
    generation_latency_ms: float | None = None
    generation_fallback_reason: str | None = None
    intent_router_mode: str = "disabled"
    intent_router_model_id: str | None = None
    intent_prompt_version: str | None = None
    intent_prompt_hash: str | None = None
    intent_router_latency_ms: float | None = None
    intent_candidate_intents: tuple[str, ...] = field(default_factory=tuple)
    intent_router_confidence: float | None = None
    intent_risk_flags: tuple[str, ...] = field(default_factory=tuple)
    intent_router_applied: bool = False
    intent_router_fallback_reason: str | None = None


@dataclass(frozen=True)
class TurnFeedbackEvent:
    schema_version: str = field(init=False, default="1.0")
    turn_id: str
    rating: str


class SafeAuditLogger:
    """只接受政策中繼資料；介面刻意不提供 transcript 或 audio 欄位。"""

    def __init__(self) -> None:
        self._logger = logging.getLogger("sva.audit")

    def turn_decision(
        self,
        *,
        turn_id: str,
        decision: str,
        intent: str,
        policy_rule_id: str,
        input_character_count: int,
        output_character_count: int,
        total_latency_ms: float,
        answer_id: str | None = None,
        answer_confidence: float | None = None,
        source_ids: list[str] | None = None,
        knowledge_versions: list[str] | None = None,
        sensitive_data_types: list[str] | None = None,
        error_type: str | None = None,
        answer_mode: str = "exact",
        generation_model_id: str | None = None,
        prompt_version: str | None = None,
        prompt_hash: str | None = None,
        generation_latency_ms: float | None = None,
        generation_fallback_reason: str | None = None,
        intent_router_mode: str = "disabled",
        intent_router_model_id: str | None = None,
        intent_prompt_version: str | None = None,
        intent_prompt_hash: str | None = None,
        intent_router_latency_ms: float | None = None,
        intent_candidate_intents: list[str] | None = None,
        intent_router_confidence: float | None = None,
        intent_risk_flags: list[str] | None = None,
        intent_router_applied: bool = False,
        intent_router_fallback_reason: str | None = None,
    ) -> None:
        event = TurnDecisionEvent(
            turn_id=turn_id,
            decision=decision,
            intent=intent,
            policy_rule_id=policy_rule_id,
            input_character_count=input_character_count,
            output_character_count=output_character_count,
            total_latency_ms=round(total_latency_ms, 3),
            answer_id=answer_id,
            answer_confidence=answer_confidence,
            source_ids=tuple(source_ids or []),
            knowledge_versions=tuple(knowledge_versions or []),
            sensitive_data_types=tuple(sensitive_data_types or []),
            error_type=error_type,
            answer_mode=answer_mode,
            generation_model_id=generation_model_id,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            generation_latency_ms=(
                round(generation_latency_ms, 3)
                if generation_latency_ms is not None
                else None
            ),
            generation_fallback_reason=generation_fallback_reason,
            intent_router_mode=intent_router_mode,
            intent_router_model_id=intent_router_model_id,
            intent_prompt_version=intent_prompt_version,
            intent_prompt_hash=intent_prompt_hash,
            intent_router_latency_ms=(
                round(intent_router_latency_ms, 3)
                if intent_router_latency_ms is not None
                else None
            ),
            intent_candidate_intents=tuple(intent_candidate_intents or []),
            intent_router_confidence=intent_router_confidence,
            intent_risk_flags=tuple(intent_risk_flags or []),
            intent_router_applied=intent_router_applied,
            intent_router_fallback_reason=intent_router_fallback_reason,
        )
        self._logger.info("turn_decision %s", json.dumps(asdict(event), ensure_ascii=False))

    def turn_feedback(self, *, turn_id: str, rating: str) -> None:
        event = TurnFeedbackEvent(turn_id=turn_id, rating=rating)
        self._logger.info("turn_feedback %s", json.dumps(asdict(event), ensure_ascii=False))
