import json
import subprocess
import sys
from pathlib import Path

from policy import DomainPolicyEngine
from retrieval import LocalKnowledgeRepository

ROOT = Path(__file__).parents[1]


def test_synthetic_golden_evaluations_pass() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "evals.run"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["failures"] == []
    assert payload["passed"] == payload["total"]
    assert payload["total"] >= 23


def test_hybrid_evaluation_covers_every_knowledge_item_and_policy_boundary() -> None:
    cases = [
        json.loads(line)
        for line in (ROOT / "evals" / "retrieval" / "hybrid.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    repository = LocalKnowledgeRepository.load(ROOT / "knowledge")
    expected_ids = {
        case["expected_knowledge_id"] for case in cases if case["category"] == "paraphrase"
    }

    assert expected_ids == {item.knowledge_id for item in repository.items}
    assert sum(case["category"] == "dangerous_near_miss" for case in cases) >= 8

    policy_engine = DomainPolicyEngine()
    for case in cases:
        policy_result = policy_engine.classify(case["input"])
        assert policy_result.action.value == case["expected_action"], case["case_id"]
        assert policy_result.intent == case.get("expected_policy_intent", case["intent"]), case[
            "case_id"
        ]


def test_intent_router_evaluation_covers_account_paraphrases_and_risk_families() -> None:
    cases = [
        json.loads(line)
        for line in (ROOT / "evals" / "intent_router" / "golden.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert all(case["synthetic"] is True for case in cases)
    account_cases = [case for case in cases if case["category"] == "paraphrase"]
    assert len(account_cases) >= 4
    assert all(case["intent"] == "account_opening_general" for case in account_cases)
    assert {case["expected_risk_flag"] for case in cases if "expected_risk_flag" in case} == {
        "transaction_execution",
        "personal_account_or_status",
        "investment_advice",
        "credential_or_sensitive_data",
        "complaint_or_dispute",
    }


def test_conversation_semantic_evaluation_covers_follow_up_and_new_topics() -> None:
    cases = [
        json.loads(line)
        for line in (ROOT / "evals" / "conversation_semantics" / "golden.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert all(case["synthetic"] is True for case in cases)
    assert len(cases) >= 6
    assert {case["expected_kind"] for case in cases} == {
        "new_question",
        "elaborate",
        "rephrase",
    }
