import json
import math
from pathlib import Path
from statistics import mean

from orchestrator.config import Settings
from orchestrator.conversation import (
    ConversationExchange,
    ConversationSemanticRoutingError,
    FollowUpKind,
    OpenAICompatibleConversationSemanticAnalyzer,
)

from .run import _load_cases

ROOT = Path(__file__).parents[1]


def _normalize_eval_query(value: str) -> str:
    normalized = value.casefold()
    for source, replacement in (
        ("三個月", "3個月"),
        ("註銷證券帳戶", "銷戶"),
        ("帳戶註銷", "銷戶"),
        ("註銷帳戶", "銷戶"),
        ("帳戶關閉", "銷戶"),
        ("帳戶關掉", "銷戶"),
        ("關閉帳戶", "銷戶"),
        ("線上申請", "線上開戶"),
        ("重新申請", "開戶"),
        ("重辦", "開戶"),
        ("如何", "操作"),
        ("怎麼", "操作"),
    ):
        normalized = normalized.replace(source, replacement)
    return normalized


def main() -> int:
    settings = Settings(
        retrieval_mode="lexical",
        intent_router_mode="disabled",
        conversation_semantic_mode="disabled",
    )
    model = (settings.conversation_llm_model or "").strip()
    if not model:
        print(
            json.dumps(
                {"status": "error", "error": "SVA_CONVERSATION_LLM_MODEL is required"}
            )
        )
        return 2

    analyzer = OpenAICompatibleConversationSemanticAnalyzer(
        base_url=str(settings.llm_base_url),
        model=model,
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
        timeout_seconds=settings.conversation_llm_timeout_seconds,
    )
    cases = _load_cases(ROOT / "evals" / "conversation_semantics" / "golden.jsonl")
    failures: list[str] = []
    latencies_ms: list[float] = []

    for case in cases:
        history = tuple(
            ConversationExchange(
                user_utterance=turn["user"],
                resolved_query=turn["resolved_query"],
                assistant_answer=turn["assistant"],
                decision="answer",
                knowledge_id=turn["knowledge_id"],
                knowledge_version="synthetic-1.0",
            )
            for turn in case["history"]
        )
        try:
            result = analyzer.analyze(
                utterance=case["current_utterance"],
                history=history,
            )
        except ConversationSemanticRoutingError:
            failures.append(f"{case['case_id']}: routing_error")
            continue

        assessment = result.assessment
        latencies_ms.append(result.latency_ms)
        expected_kind = FollowUpKind(case["expected_kind"])
        actual_reference_knowledge_id = None
        if assessment.reference_turn_id is not None:
            actual_reference_index = int(assessment.reference_turn_id.removeprefix("T")) - 1
            if 0 <= actual_reference_index < len(case["history"]):
                actual_reference_knowledge_id = case["history"][actual_reference_index][
                    "knowledge_id"
                ]
        expected_reference_knowledge_id = None
        if case["expected_reference_turn_id"] is not None:
            expected_reference_index = (
                int(case["expected_reference_turn_id"].removeprefix("T")) - 1
            )
            expected_reference_knowledge_id = case["history"][expected_reference_index][
                "knowledge_id"
            ]
        normalized_query = _normalize_eval_query(assessment.rewritten_query)
        missing_terms = [
            term
            for term in case["rewritten_query_terms"]
            if _normalize_eval_query(term) not in normalized_query
        ]
        passed = (
            assessment.kind is expected_kind
            and actual_reference_knowledge_id == expected_reference_knowledge_id
            and not missing_terms
        )
        if expected_kind is not FollowUpKind.NEW_QUESTION:
            passed = (
                passed
                and assessment.confidence
                >= settings.conversation_semantic_minimum_confidence
            )
        if not passed:
            failures.append(
                f"{case['case_id']}: expected_kind={expected_kind.value}, "
                f"actual_kind={assessment.kind.value}, "
                f"reference={assessment.reference_turn_id}, "
                f"confidence={assessment.confidence}, missing_terms={missing_terms}"
            )

    sorted_latencies = sorted(latencies_ms)
    p95_index = max(0, math.ceil(len(sorted_latencies) * 0.95) - 1)
    print(
        json.dumps(
            {
                "status": "passed" if not failures else "regression",
                "model": model,
                "minimum_confidence": settings.conversation_semantic_minimum_confidence,
                "total": len(cases),
                "passed": len(cases) - len(failures),
                "average_latency_ms": (
                    round(mean(sorted_latencies), 1) if sorted_latencies else None
                ),
                "p95_latency_ms": (
                    round(sorted_latencies[p95_index], 1) if sorted_latencies else None
                ),
                "failures": failures,
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
