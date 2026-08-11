import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TurnDecisionEvent:
    schema_version: str = field(init=False, default="1.5")
    turn_id: str
    decision: str
    intent: str
    policy_rule_id: str
    input_character_count: int
    output_character_count: int
    total_latency_ms: float
    end_to_end_latency_ms: float
    conversation_resolution_latency_ms: float | None = None
    conversation_semantic_latency_ms: float | None = None
    policy_guard_latency_ms: float | None = None
    retrieval_latency_ms: float | None = None
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
    generation_applied: bool = False
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


@dataclass(frozen=True)
class ShadowGenerationEvent:
    schema_version: str = field(init=False, default="1.0")
    turn_id: str
    answer_id: str
    knowledge_version: str
    source_id: str
    generation_model_id: str | None
    prompt_version: str | None
    prompt_hash: str | None
    generation_latency_ms: float | None
    output_guard_safe: bool
    fallback_reason: str


@dataclass(frozen=True)
class VoiceSynthesisEvent:
    schema_version: str = field(init=False, default="1.0")
    turn_id: str
    tts_model_id: str
    sentence_count: int
    audio_chunk_count: int
    first_audio_latency_ms: float | None
    total_latency_ms: float
    error_type: str | None


@dataclass(frozen=True)
class VoicePlaybackChunkTiming:
    arrival_offset_ms: float
    duration_ms: float
    scheduled_start_offset_ms: float | None
    gap_before_ms: float


@dataclass(frozen=True)
class VoicePlaybackEvent:
    schema_version: str = field(init=False, default="1.1")
    turn_id: str
    chunk_count: int
    audio_duration_ms: float
    initial_buffered_ms: float
    first_playback_delay_ms: float | None
    buffer_target_ms: float
    crossfade_ms: float
    underrun_count: int
    underrun_total_ms: float
    underrun_max_ms: float
    interrupted: bool
    interruption_reason: str | None
    barge_in_mode: str | None
    barge_in_duck_latency_ms: float | None
    barge_in_confirm_latency_ms: float | None
    barge_in_false_trigger_count: int
    chunk_timings: tuple[VoicePlaybackChunkTiming, ...]


def _required_metric(
    chunk: Mapping[str, float | None],
    key: str,
) -> float:
    value = chunk[key]
    if value is None:
        raise ValueError(f"{key} must not be null")
    return round(value, 3)


def _optional_metric(
    chunk: Mapping[str, float | None],
    key: str,
) -> float | None:
    value = chunk[key]
    return round(value, 3) if value is not None else None


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
        end_to_end_latency_ms: float,
        conversation_resolution_latency_ms: float | None = None,
        conversation_semantic_latency_ms: float | None = None,
        policy_guard_latency_ms: float | None = None,
        retrieval_latency_ms: float | None = None,
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
        generation_applied: bool = False,
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
            end_to_end_latency_ms=round(end_to_end_latency_ms, 3),
            conversation_resolution_latency_ms=(
                round(conversation_resolution_latency_ms, 3)
                if conversation_resolution_latency_ms is not None
                else None
            ),
            conversation_semantic_latency_ms=(
                round(conversation_semantic_latency_ms, 3)
                if conversation_semantic_latency_ms is not None
                else None
            ),
            policy_guard_latency_ms=(
                round(policy_guard_latency_ms, 3)
                if policy_guard_latency_ms is not None
                else None
            ),
            retrieval_latency_ms=(
                round(retrieval_latency_ms, 3)
                if retrieval_latency_ms is not None
                else None
            ),
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
            generation_applied=generation_applied,
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

    def shadow_generation(
        self,
        *,
        turn_id: str,
        answer_id: str,
        knowledge_version: str,
        source_id: str,
        output_guard_safe: bool,
        fallback_reason: str,
        generation_model_id: str | None = None,
        prompt_version: str | None = None,
        prompt_hash: str | None = None,
        generation_latency_ms: float | None = None,
    ) -> None:
        event = ShadowGenerationEvent(
            turn_id=turn_id,
            answer_id=answer_id,
            knowledge_version=knowledge_version,
            source_id=source_id,
            generation_model_id=generation_model_id,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            generation_latency_ms=(
                round(generation_latency_ms, 3)
                if generation_latency_ms is not None
                else None
            ),
            output_guard_safe=output_guard_safe,
            fallback_reason=fallback_reason,
        )
        self._logger.info("shadow_generation %s", json.dumps(asdict(event), ensure_ascii=False))

    def voice_synthesis(
        self,
        *,
        turn_id: str,
        tts_model_id: str,
        sentence_count: int,
        audio_chunk_count: int,
        first_audio_latency_ms: float | None,
        total_latency_ms: float,
        error_type: str | None,
    ) -> None:
        event = VoiceSynthesisEvent(
            turn_id=turn_id,
            tts_model_id=tts_model_id,
            sentence_count=sentence_count,
            audio_chunk_count=audio_chunk_count,
            first_audio_latency_ms=(
                round(first_audio_latency_ms, 3)
                if first_audio_latency_ms is not None
                else None
            ),
            total_latency_ms=round(total_latency_ms, 3),
            error_type=error_type,
        )
        self._logger.info("voice_synthesis %s", json.dumps(asdict(event), ensure_ascii=False))

    def voice_playback(
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
        chunk_timings: Sequence[Mapping[str, float | None]],
    ) -> None:
        event = VoicePlaybackEvent(
            turn_id=turn_id,
            chunk_count=chunk_count,
            audio_duration_ms=round(audio_duration_ms, 3),
            initial_buffered_ms=round(initial_buffered_ms, 3),
            first_playback_delay_ms=(
                round(first_playback_delay_ms, 3)
                if first_playback_delay_ms is not None
                else None
            ),
            buffer_target_ms=round(buffer_target_ms, 3),
            crossfade_ms=round(crossfade_ms, 3),
            underrun_count=underrun_count,
            underrun_total_ms=round(underrun_total_ms, 3),
            underrun_max_ms=round(underrun_max_ms, 3),
            interrupted=interrupted,
            interruption_reason=interruption_reason,
            barge_in_mode=barge_in_mode,
            barge_in_duck_latency_ms=(
                round(barge_in_duck_latency_ms, 3)
                if barge_in_duck_latency_ms is not None
                else None
            ),
            barge_in_confirm_latency_ms=(
                round(barge_in_confirm_latency_ms, 3)
                if barge_in_confirm_latency_ms is not None
                else None
            ),
            barge_in_false_trigger_count=barge_in_false_trigger_count,
            chunk_timings=tuple(
                VoicePlaybackChunkTiming(
                    arrival_offset_ms=_required_metric(chunk, "arrival_offset_ms"),
                    duration_ms=_required_metric(chunk, "duration_ms"),
                    scheduled_start_offset_ms=_optional_metric(
                        chunk,
                        "scheduled_start_offset_ms",
                    ),
                    gap_before_ms=_required_metric(chunk, "gap_before_ms"),
                )
                for chunk in chunk_timings
            ),
        )
        self._logger.info("voice_playback %s", json.dumps(asdict(event), ensure_ascii=False))
