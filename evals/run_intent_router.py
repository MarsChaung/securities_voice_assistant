import json
import math
from pathlib import Path
from statistics import mean

from orchestrator.config import Settings
from orchestrator.intent_routing import (
    IntentRoutingError,
    OpenAICompatibleIntentRouter,
)

from .run import _load_cases

ROOT = Path(__file__).parents[1]

_EXPECTED_RISK_FLAGS = {
    "near-miss-transaction": "transaction_execution",
    "near-miss-personal-balance": "personal_account_or_status",
    "near-miss-investment-advice": "investment_advice",
    "near-miss-credential": "credential_or_sensitive_data",
    "near-miss-account-status": "personal_account_or_status",
}


def main() -> int:
    settings = Settings(retrieval_mode="lexical", intent_router_mode="disabled")
    model = (settings.intent_llm_model or "").strip()
    if not model:
        print(json.dumps({"status": "error", "error": "SVA_INTENT_LLM_MODEL is required"}))
        return 2

    router = OpenAICompatibleIntentRouter(
        base_url=str(settings.llm_base_url),
        model=model,
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
        timeout_seconds=settings.intent_llm_timeout_seconds,
    )
    cases = [
        *_load_cases(ROOT / "evals" / "retrieval" / "hybrid.jsonl"),
        *_load_cases(ROOT / "evals" / "intent_router" / "golden.jsonl"),
    ]
    failures: list[str] = []
    safe_intent_correct = 0
    risk_correct = 0
    latencies_ms: list[float] = []

    for case in cases:
        try:
            route = router.route(case["input"])
        except IntentRoutingError:
            failures.append(f"{case['case_id']}: routing_error")
            continue

        classification = route.classification
        latencies_ms.append(route.latency_ms)
        top_intent = classification.candidate_intents[0]
        expected_risk = case.get("expected_risk_flag") or _EXPECTED_RISK_FLAGS.get(case["case_id"])
        if expected_risk:
            passed = expected_risk in classification.risk_flags
            risk_correct += int(passed)
            if not passed:
                failures.append(
                    f"{case['case_id']}: expected_risk={expected_risk}, "
                    f"actual={classification.risk_flags}"
                )
            continue

        expected_intent = case["intent"]
        if expected_intent == "unknown_or_ambiguous":
            expected_intent = "unknown"
        is_safe_rejection_case = (
            case["category"] == "dangerous_near_miss" and case.get("expected_knowledge_id") is None
        )
        passed = top_intent == expected_intent or (
            is_safe_rejection_case and top_intent == "unknown"
        )
        requires_controlled_confidence = case["category"] == "paraphrase"
        if requires_controlled_confidence:
            passed = (
                passed
                and not classification.risk_flags
                and not classification.needs_clarification
                and classification.confidence >= settings.intent_router_minimum_confidence
            )
        safe_intent_correct += int(passed)
        if not passed:
            failures.append(
                f"{case['case_id']}: expected_intent={expected_intent}, "
                f"actual={top_intent}, confidence={classification.confidence}, "
                f"risk_flags={classification.risk_flags}, "
                f"needs_clarification={classification.needs_clarification}"
            )

    passed = not failures
    sorted_latencies = sorted(latencies_ms)
    p95_index = max(0, math.ceil(len(sorted_latencies) * 0.95) - 1)
    print(
        json.dumps(
            {
                "status": "passed" if passed else "regression",
                "model": model,
                "minimum_confidence": settings.intent_router_minimum_confidence,
                "total": len(cases),
                "safe_intent_correct": safe_intent_correct,
                "risk_correct": risk_correct,
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
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
