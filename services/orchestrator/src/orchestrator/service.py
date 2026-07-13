from uuid import uuid4

from answer_contract import AnswerContract, Decision, TurnRequest, TurnResponse
from observability import SafeAuditLogger
from policy import DomainPolicyEngine, PolicyAction, SensitiveDataGuard


class TurnService:
    def __init__(
        self,
        sensitive_data_guard: SensitiveDataGuard | None = None,
        policy_engine: DomainPolicyEngine | None = None,
        audit_logger: SafeAuditLogger | None = None,
    ) -> None:
        self._sensitive_data_guard = sensitive_data_guard or SensitiveDataGuard()
        self._policy_engine = policy_engine or DomainPolicyEngine()
        self._audit_logger = audit_logger or SafeAuditLogger()

    def evaluate(self, request: TurnRequest) -> TurnResponse:
        turn_id = str(uuid4())
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
            self._audit_logger.turn_decision(
                turn_id=turn_id,
                decision=result.decision.value,
                intent=result.intent,
                policy_rule_id=result.policy_rule_id,
                sensitive_data_types=guard_result.detected_types,
            )
            return TurnResponse(turn_id=turn_id, result=result)

        policy_result = self._policy_engine.classify(request.transcript)

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
            # Phase 2 尚未提供核准知識來源；允許意圖也不能讓模型自由補答。
            result = AnswerContract(
                decision=Decision.REFUSE,
                intent=policy_result.intent,
                policy_rule_id="KNO-001",
                answer="目前沒有足夠且已核准的知識來源可回答，請改用官方客服管道。",
                confidence=1.0,
            )

        self._audit_logger.turn_decision(
            turn_id=turn_id,
            decision=result.decision.value,
            intent=result.intent,
            policy_rule_id=result.policy_rule_id,
        )
        return TurnResponse(turn_id=turn_id, result=result)
