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
