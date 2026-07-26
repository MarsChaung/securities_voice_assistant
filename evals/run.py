import json
from pathlib import Path
from typing import Any

from orchestrator.asr import MandarinPhoneticResolver
from policy import DomainPolicyEngine, SensitiveDataGuard
from retrieval import (
    KnowledgeDocument,
    LexicalKnowledgeRetriever,
    LocalKnowledgeRepository,
    QuestionVariant,
    QuestionVariantUsage,
)

EVAL_ROOT = Path(__file__).parent


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        case = json.loads(line)
        if case.get("synthetic") is not True:
            raise ValueError(f"{path}:{line_number} 必須明確標示 synthetic=true")
        cases.append(case)
    return cases


def run_evaluations() -> tuple[int, int, list[str]]:
    failures: list[str] = []
    total = 0

    policy_engine = DomainPolicyEngine()
    for case in _load_cases(EVAL_ROOT / "policy" / "golden.jsonl"):
        total += 1
        policy_result = policy_engine.classify(case["input"])
        actual = (policy_result.action.value, policy_result.intent)
        expected = (case["expected_action"], case["expected_intent"])
        if actual != expected:
            failures.append(f"{case['case_id']}: expected={expected}, actual={actual}")

    sensitive_data_guard = SensitiveDataGuard()
    for case in _load_cases(EVAL_ROOT / "pii" / "golden.jsonl"):
        total += 1
        guard_result = sensitive_data_guard.scan(case["input"])
        actual_types = sorted(guard_result.detected_types)
        expected_types = sorted(case["expected_types"])
        if actual_types != expected_types:
            failures.append(
                f"{case['case_id']}: expected_types={expected_types}, actual_types={actual_types}"
            )
        if case["sensitive_value"] in guard_result.redacted_text:
            failures.append(f"{case['case_id']}: detected value was not redacted")

    local_repository = LocalKnowledgeRepository.load(EVAL_ROOT.parent / "knowledge")
    source_map = {source.source_id: source for source in local_repository.sources}
    documents = tuple(
        KnowledgeDocument(item=item, source=source_map[item.source_id])
        for item in local_repository.items
    )
    retriever = LexicalKnowledgeRetriever()
    for case in _load_cases(EVAL_ROOT / "retrieval" / "golden.jsonl"):
        total += 1
        match = retriever.search(
            query=case["input"],
            intent=case["intent"],
            documents=documents,
        )
        actual_id = match.document.item.knowledge_id if match else None
        if actual_id != case["expected_knowledge_id"]:
            failures.append(
                f"{case['case_id']}: expected={case['expected_knowledge_id']}, actual={actual_id}"
            )

    base_document = documents[0]
    phonetic_resolver = MandarinPhoneticResolver()
    for index, case in enumerate(
        _load_cases(EVAL_ROOT / "asr_phonetic" / "golden.jsonl"),
        start=1,
    ):
        total += 1
        document = KnowledgeDocument(
            item=base_document.item.model_copy(
                update={
                    "knowledge_id": f"K-ASR-EVAL-{index:03d}",
                    "title": case["target_title"],
                    "allowed_intents": ["faq_general_guidance"],
                    "question_variants": [
                        QuestionVariant(
                            variant_id=f"asr-eval-{index}",
                            question_text=case["target_variant"],
                            usage=QuestionVariantUsage.RETRIEVAL,
                        )
                    ],
                }
            ),
            source=base_document.source,
        )
        resolution = phonetic_resolver.resolve(
            query=case["input"],
            intent="general_securities_knowledge",
            documents=(document,),
        )
        actual_match = resolution.match is not None
        if actual_match is not case["expected_match"]:
            failures.append(
                f"{case['case_id']}: expected_match={case['expected_match']}, "
                f"actual_match={actual_match}"
            )

    return total - len(failures), total, failures


def main() -> int:
    passed, total, failures = run_evaluations()
    print(json.dumps({"passed": passed, "total": total, "failures": failures}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
