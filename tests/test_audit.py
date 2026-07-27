import json
import logging

import pytest

from answer_contract import TurnRequest
from observability import SafeAuditLogger
from orchestrator.service import TurnService


def test_audit_event_uses_metadata_allowlist_and_never_logs_transcript(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transcript = "我的驗證碼 123456"

    with caplog.at_level(logging.INFO, logger="sva.audit"):
        TurnService().evaluate(TurnRequest(transcript=transcript, channel="web"))

    records = [record for record in caplog.records if record.name == "sva.audit"]
    assert len(records) == 1

    message = records[0].getMessage()
    event = json.loads(message.removeprefix("turn_decision "))

    assert set(event) == {
        "schema_version",
        "turn_id",
        "decision",
        "intent",
        "policy_rule_id",
        "source_ids",
        "knowledge_versions",
        "sensitive_data_types",
        "input_character_count",
        "output_character_count",
        "total_latency_ms",
        "answer_id",
        "answer_confidence",
        "error_type",
        "answer_mode",
        "generation_model_id",
        "prompt_version",
        "prompt_hash",
        "generation_latency_ms",
        "generation_applied",
        "generation_fallback_reason",
        "intent_router_mode",
        "intent_router_model_id",
        "intent_prompt_version",
        "intent_prompt_hash",
        "intent_router_latency_ms",
        "intent_candidate_intents",
        "intent_router_confidence",
        "intent_risk_flags",
        "intent_router_applied",
        "intent_router_fallback_reason",
    }
    assert event["schema_version"] == "1.4"
    assert event["answer_mode"] == "exact"
    assert event["generation_model_id"] is None
    assert event["prompt_version"] is None
    assert event["prompt_hash"] is None
    assert event["generation_latency_ms"] is None
    assert event["generation_applied"] is False
    assert event["generation_fallback_reason"] is None
    assert event["intent_router_mode"] == "disabled"
    assert event["intent_router_model_id"] is None
    assert event["intent_candidate_intents"] == []
    assert event["intent_risk_flags"] == []
    assert event["intent_router_applied"] is False
    assert event["input_character_count"] == len(transcript)
    assert event["output_character_count"] > 0
    assert event["source_ids"] == []
    assert event["knowledge_versions"] == []
    assert event["sensitive_data_types"] == ["otp"]
    assert event["total_latency_ms"] >= 0
    assert event["answer_id"] is None
    assert event["answer_confidence"] == 1.0
    assert transcript not in caplog.text
    assert "123456" not in caplog.text


def test_shadow_generation_event_contains_only_review_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    generated_answer = "這段模型答案不可進入稽核日誌。"

    with caplog.at_level(logging.INFO, logger="sva.audit"):
        SafeAuditLogger().shadow_generation(
            turn_id="turn-shadow",
            answer_id="K-001",
            knowledge_version="1.1",
            source_id="SRC-001",
            generation_model_id="model-1",
            prompt_version="prompt-v1",
            prompt_hash="a" * 64,
            generation_latency_ms=12.3456,
            output_guard_safe=True,
            fallback_reason="shadow_only",
        )

    record = next(record for record in caplog.records if "shadow_generation" in record.message)
    event = json.loads(record.getMessage().removeprefix("shadow_generation "))
    assert event == {
        "schema_version": "1.0",
        "turn_id": "turn-shadow",
        "answer_id": "K-001",
        "knowledge_version": "1.1",
        "source_id": "SRC-001",
        "generation_model_id": "model-1",
        "prompt_version": "prompt-v1",
        "prompt_hash": "a" * 64,
        "generation_latency_ms": 12.346,
        "output_guard_safe": True,
        "fallback_reason": "shadow_only",
    }
    assert generated_answer not in caplog.text


def test_voice_playback_event_contains_only_timing_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_content = "客戶問題與語音內容不可進入播放品質日誌"

    with caplog.at_level(logging.INFO, logger="sva.audit"):
        SafeAuditLogger().voice_playback(
            turn_id="turn-playback",
            chunk_count=2,
            audio_duration_ms=1_000.1234,
            initial_buffered_ms=1_000.1234,
            first_playback_delay_ms=1_205.6789,
            buffer_target_ms=1_200,
            crossfade_ms=8,
            underrun_count=1,
            underrun_total_ms=42.6789,
            underrun_max_ms=42.6789,
            interrupted=False,
            interruption_reason=None,
            barge_in_mode=None,
            barge_in_duck_latency_ms=None,
            barge_in_confirm_latency_ms=None,
            barge_in_false_trigger_count=0,
            chunk_timings=[
                {
                    "arrival_offset_ms": 300.1234,
                    "duration_ms": 500.1234,
                    "scheduled_start_offset_ms": 1_240.1234,
                    "gap_before_ms": 0,
                },
                {
                    "arrival_offset_ms": 900.1234,
                    "duration_ms": 500,
                    "scheduled_start_offset_ms": 1_782.1234,
                    "gap_before_ms": 42.6789,
                },
            ],
        )

    record = next(record for record in caplog.records if "voice_playback" in record.message)
    event = json.loads(record.getMessage().removeprefix("voice_playback "))
    assert set(event) == {
        "schema_version",
        "turn_id",
        "chunk_count",
        "audio_duration_ms",
        "initial_buffered_ms",
        "first_playback_delay_ms",
        "buffer_target_ms",
        "crossfade_ms",
        "underrun_count",
        "underrun_total_ms",
        "underrun_max_ms",
        "interrupted",
        "interruption_reason",
        "barge_in_mode",
        "barge_in_duck_latency_ms",
        "barge_in_confirm_latency_ms",
        "barge_in_false_trigger_count",
        "chunk_timings",
    }
    assert event["schema_version"] == "1.1"
    assert event["underrun_total_ms"] == 42.679
    assert event["chunk_timings"][1]["gap_before_ms"] == 42.679
    assert private_content not in caplog.text
