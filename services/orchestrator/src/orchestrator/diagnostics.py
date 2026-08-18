import json
import logging


class VoiceTestDiagnosticLogger:
    """Development-only conversation logger for the local voice test UI."""

    def __init__(
        self,
        *,
        enabled: bool,
        app_env: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.enabled = enabled and app_env == "development"
        self._logger = logger or logging.getLogger("sva.voice_test")

    def exchange(
        self,
        *,
        session_id: str | None,
        turn_id: str,
        channel: str,
        reply_mode: str,
        user_utterance: str,
        assistant_answer: str,
        decision: str,
        intent: str,
        policy_rule_id: str,
        answer_id: str | None,
        knowledge_versions: list[str],
        contains_sensitive_data: bool,
        follow_up_kind: str | None = None,
        semantic_applied: bool = False,
        semantic_confidence: float | None = None,
        reference_knowledge_id: str | None = None,
        semantic_focus: str | None = None,
        resolved_query: str | None = None,
    ) -> None:
        if not self.enabled or session_id is None:
            return
        event = {
            "schema_version": "1.1",
            "session_id": session_id,
            "turn_id": turn_id,
            "channel": channel,
            "reply_mode": reply_mode,
            "user_utterance": ("[REDACTED]" if contains_sensitive_data else user_utterance),
            "assistant_answer": ("[REDACTED]" if contains_sensitive_data else assistant_answer),
            "content_redacted": contains_sensitive_data,
            "decision": decision,
            "intent": intent,
            "policy_rule_id": policy_rule_id,
            "answer_id": answer_id,
            "knowledge_versions": knowledge_versions,
            "follow_up_kind": follow_up_kind,
            "semantic_applied": semantic_applied,
            "semantic_confidence": semantic_confidence,
            "reference_knowledge_id": reference_knowledge_id,
            "semantic_focus": semantic_focus,
            "resolved_query": (
                "[REDACTED]" if contains_sensitive_data else resolved_query
            ),
        }
        self._logger.info(
            "voice_test_conversation %s",
            json.dumps(event, ensure_ascii=False),
        )

    def acknowledgement(
        self,
        *,
        session_id: str | None,
        variant: str,
        triggered_after_ms: float,
        answer_ready_after_ms: float,
    ) -> None:
        if not self.enabled or session_id is None:
            return
        event = {
            "schema_version": "1.0",
            "session_id": session_id,
            "variant": variant,
            "triggered_after_ms": round(triggered_after_ms, 3),
            "answer_ready_after_ms": round(answer_ready_after_ms, 3),
        }
        self._logger.info(
            "voice_test_acknowledgement %s",
            json.dumps(event, ensure_ascii=False),
        )
