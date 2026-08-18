import json
import math
from pathlib import Path
from statistics import mean

from orchestrator.answer_judge import (
    GroundednessJudgeError,
    OpenAICompatibleGroundednessJudge,
)
from orchestrator.answering import (
    AnswerEvidence,
    AnswerGenerationError,
    ControlledOutputGuard,
    OpenAICompatibleAnswerComposer,
)
from orchestrator.config import Settings
from retrieval import LocalKnowledgeRepository

from .run import _load_cases

ROOT = Path(__file__).parents[1]


def _percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = max(0, math.ceil(len(sorted_values) * 0.95) - 1)
    return round(sorted_values[index], 1)


def main() -> int:
    settings = Settings(
        retrieval_mode="lexical",
        answer_mode="exact",
        intent_router_mode="disabled",
    )
    generator_model = (settings.answer_llm_model or "").strip()
    judge_model = (settings.answer_judge_model or "").strip()
    if not generator_model or not judge_model:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "SVA_ANSWER_LLM_MODEL and SVA_ANSWER_JUDGE_MODEL are required",
                }
            )
        )
        return 2

    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    composer = OpenAICompatibleAnswerComposer(
        base_url=str(settings.llm_base_url),
        model=generator_model,
        api_key=api_key,
        timeout_seconds=settings.answer_llm_timeout_seconds,
    )
    judge = OpenAICompatibleGroundednessJudge(
        base_url=str(settings.llm_base_url),
        model=judge_model,
        api_key=api_key,
        timeout_seconds=settings.answer_judge_timeout_seconds,
    )
    output_guard = ControlledOutputGuard()

    query_by_knowledge_id: dict[str, str] = {}
    for case in _load_cases(ROOT / "evals" / "retrieval" / "hybrid.jsonl"):
        knowledge_id = case.get("expected_knowledge_id")
        if case["category"] == "paraphrase" and knowledge_id:
            query_by_knowledge_id.setdefault(knowledge_id, case["input"])

    repository = LocalKnowledgeRepository.load(ROOT / "knowledge")
    source_map = {source.source_id: source for source in repository.sources}
    failures: list[str] = []
    generated_count = 0
    guard_passed = 0
    grounded_count = 0
    changed_count = 0
    generation_latencies: list[float] = []
    judge_latencies: list[float] = []
    judge_candidates: list[tuple[AnswerEvidence, str]] = []

    for item in repository.items:
        question = query_by_knowledge_id.get(item.knowledge_id)
        if question is None:
            failures.append(f"{item.knowledge_id}: missing_evaluation_question")
            continue
        evidence = AnswerEvidence(
            standard_answer=item.standard_answer,
            prohibited_extensions=tuple(item.prohibited_extensions),
            knowledge_id=item.knowledge_id,
            knowledge_version=item.version,
            source_id=source_map[item.source_id].source_id,
        )
        try:
            generated = composer.compose(evidence)
        except AnswerGenerationError:
            failures.append(f"{item.knowledge_id}: generation_error")
            continue
        generated_count += 1
        generation_latencies.append(generated.latency_ms)
        changed_count += int(generated.answer != item.standard_answer)

        guard_result = output_guard.validate(
            generated_answer=generated.answer,
            standard_answer=item.standard_answer,
            prohibited_extensions=tuple(item.prohibited_extensions),
        )
        if not guard_result.safe:
            failures.append(f"{item.knowledge_id}: output_guard:{guard_result.reason}")
            continue
        guard_passed += 1
        judge_candidates.append((evidence, generated.answer))

    for evidence, generated_answer in judge_candidates:
        try:
            judged = judge.assess(
                evidence=evidence,
                generated_answer=generated_answer,
            )
        except GroundednessJudgeError:
            failures.append(f"{evidence.knowledge_id}: judge_error")
            continue
        judge_latencies.append(judged.latency_ms)
        assessment = judged.assessment
        is_grounded = (
            assessment.verdict == "grounded"
            and assessment.preserves_required_qualifiers
            and assessment.unsupported_claim_count == 0
            and not assessment.prohibited_extension_detected
        )
        grounded_count += int(is_grounded)
        if not is_grounded:
            failures.append(f"{evidence.knowledge_id}: judge:{assessment.reason_code}")

    total = len(repository.items)
    passed = not failures and grounded_count == total
    print(
        json.dumps(
            {
                "status": "passed" if passed else "regression",
                "generator_model": generator_model,
                "judge_model": judge_model,
                "total": total,
                "generated": generated_count,
                "changed": changed_count,
                "output_guard_passed": guard_passed,
                "grounded": grounded_count,
                "generation_average_latency_ms": (
                    round(mean(generation_latencies), 1) if generation_latencies else None
                ),
                "generation_p95_latency_ms": _percentile_95(generation_latencies),
                "judge_average_latency_ms": (
                    round(mean(judge_latencies), 1) if judge_latencies else None
                ),
                "judge_p95_latency_ms": _percentile_95(judge_latencies),
                "failures": failures,
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
