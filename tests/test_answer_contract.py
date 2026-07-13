import pytest
from pydantic import ValidationError

from answer_contract import AnswerContract, Decision


def test_answer_requires_traceable_evidence() -> None:
    with pytest.raises(ValidationError):
        AnswerContract(
            decision=Decision.ANSWER,
            intent="app_public_help",
            policy_rule_id="POL-ALLOW-001",
            answer="請依照核准步驟操作。",
            confidence=1.0,
        )


def test_answer_accepts_complete_evidence() -> None:
    contract = AnswerContract(
        decision=Decision.ANSWER,
        intent="app_public_help",
        policy_rule_id="POL-ALLOW-001",
        answer_id="FAQ-001",
        source_ids=["KB-001"],
        knowledge_versions=["1.0"],
        answer="請依照核准步驟操作。",
        confidence=1.0,
    )

    assert contract.decision is Decision.ANSWER
