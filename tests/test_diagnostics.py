import json
import logging

from _pytest.logging import LogCaptureFixture

from orchestrator.diagnostics import VoiceTestDiagnosticLogger


def test_voice_test_diagnostic_logger_records_session_and_content(
    caplog: LogCaptureFixture,
) -> None:
    logger = VoiceTestDiagnosticLogger(enabled=True, app_env="development")

    with caplog.at_level(logging.INFO, logger="sva.voice_test"):
        logger.exchange(
            session_id="session-123",
            turn_id="turn-456",
            channel="web",
            reply_mode="natural",
            user_utterance="父母要帶什麼證件",
            assistant_answer="請攜帶核准知識列出的證件。",
            decision="answer",
            intent="knowledge_qa",
            policy_rule_id="POL-ALLOW-001",
            answer_id="K-FAQ-001",
            knowledge_versions=["1.0"],
            contains_sensitive_data=False,
            follow_up_kind="elaborate",
            semantic_applied=True,
            semantic_confidence=0.96,
            reference_knowledge_id="K-FAQ-001",
            semantic_focus="證件",
            resolved_query="未成年人開戶需要哪些證件",
        )

    message = caplog.records[-1].getMessage()
    payload = json.loads(message.removeprefix("voice_test_conversation "))
    assert payload["session_id"] == "session-123"
    assert payload["user_utterance"] == "父母要帶什麼證件"
    assert payload["assistant_answer"] == "請攜帶核准知識列出的證件。"
    assert payload["follow_up_kind"] == "elaborate"
    assert payload["semantic_applied"] is True
    assert payload["reference_knowledge_id"] == "K-FAQ-001"
    assert payload["resolved_query"] == "未成年人開戶需要哪些證件"


def test_voice_test_diagnostic_logger_is_off_outside_development(
    caplog: LogCaptureFixture,
) -> None:
    logger = VoiceTestDiagnosticLogger(enabled=True, app_env="production")

    with caplog.at_level(logging.INFO, logger="sva.voice_test"):
        logger.exchange(
            session_id="session-123",
            turn_id="turn-456",
            channel="web",
            reply_mode="exact",
            user_utterance="測試問題",
            assistant_answer="測試回答",
            decision="answer",
            intent="knowledge_qa",
            policy_rule_id="POL-ALLOW-001",
            answer_id="K-FAQ-001",
            knowledge_versions=["1.0"],
            contains_sensitive_data=False,
        )

    assert caplog.records == []


def test_voice_test_diagnostic_logger_redacts_sensitive_content(
    caplog: LogCaptureFixture,
) -> None:
    logger = VoiceTestDiagnosticLogger(enabled=True, app_env="development")

    with caplog.at_level(logging.INFO, logger="sva.voice_test"):
        logger.exchange(
            session_id="session-123",
            turn_id="turn-456",
            channel="web",
            reply_mode="exact",
            user_utterance="我的帳號是敏感資料",
            assistant_answer="請勿提供敏感資料",
            decision="refuse",
            intent="sensitive_data_detected",
            policy_rule_id="PII-001",
            answer_id=None,
            knowledge_versions=[],
            contains_sensitive_data=True,
        )

    message = caplog.records[-1].getMessage()
    payload = json.loads(message.removeprefix("voice_test_conversation "))
    assert payload["user_utterance"] == "[REDACTED]"
    assert payload["assistant_answer"] == "[REDACTED]"
    assert payload["content_redacted"] is True
