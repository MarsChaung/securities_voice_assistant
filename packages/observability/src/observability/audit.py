import json
import logging
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TurnDecisionEvent:
    turn_id: str
    decision: str
    intent: str
    policy_rule_id: str
    sensitive_data_types: list[str] = field(default_factory=list)


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
        sensitive_data_types: list[str] | None = None,
    ) -> None:
        event = TurnDecisionEvent(
            turn_id=turn_id,
            decision=decision,
            intent=intent,
            policy_rule_id=policy_rule_id,
            sensitive_data_types=sensitive_data_types or [],
        )
        self._logger.info("turn_decision %s", json.dumps(asdict(event), ensure_ascii=False))
