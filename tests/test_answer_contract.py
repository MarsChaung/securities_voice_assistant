import pytest
from pydantic import ValidationError

from answer_contract import AnswerContract, Citation, Decision, TurnFeedback


def test_clarification_does_not_require_knowledge_evidence() -> None:
    contract = AnswerContract(
        decision=Decision.CLARIFY,
        intent="general_securities_knowledge",
        policy_rule_id="ASR-PHONETIC-002",
        answer="請再說一次完整問題。",
        confidence=0.92,
    )

    assert contract.answer_id is None
    assert contract.citations == []


def test_answer_requires_traceable_evidence() -> None:
    with pytest.raises(ValidationError):
        AnswerContract(
            decision=Decision.ANSWER,
            intent="web_public_help",
            policy_rule_id="POL-ALLOW-001",
            answer="請依照核准步驟操作。",
            confidence=1.0,
        )


def test_answer_accepts_complete_evidence() -> None:
    contract = AnswerContract(
        decision=Decision.ANSWER,
        intent="web_public_help",
        policy_rule_id="POL-ALLOW-001",
        answer_id="FAQ-001",
        source_ids=["SRC-TEST-001"],
        knowledge_versions=["1.0"],
        citations=[
            Citation(
                source_id="SRC-TEST-001",
                source_uri="https://example.com/official",
                source_title="官方來源",
                source_locator="常見問題",
            )
        ],
        answer="請依照核准步驟操作。",
        confidence=1.0,
    )

    assert contract.decision is Decision.ANSWER


def test_answer_accepts_local_import_evidence_without_url() -> None:
    citation = Citation(
        source_id="SRC-LOCAL-001",
        source_uri=None,
        source_title="本機匯入 FAQ",
        source_locator="FAQ 匯入批次 batch-1／第 4 列",
    )

    assert citation.source_uri is None


def test_feedback_only_accepts_fixed_ratings() -> None:
    assert TurnFeedback(rating="helpful").rating == "helpful"

    with pytest.raises(ValidationError):
        TurnFeedback(rating="包含自由文字")  # type: ignore[arg-type]
