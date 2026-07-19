import json
from pathlib import Path

from orchestrator.config import Settings
from policy import DomainPolicyEngine, PolicyAction
from retrieval import (
    EmbeddingServiceError,
    HybridKnowledgeRetriever,
    KnowledgeDocument,
    LexicalKnowledgeRetriever,
    LocalKnowledgeRepository,
    OpenAICompatibleEmbeddingClient,
    RetrievalMatch,
)

from .run import _load_cases

ROOT = Path(__file__).parents[1]


def _match_id(match: RetrievalMatch | None) -> str | None:
    return match.document.item.knowledge_id if match else None


def main() -> int:
    settings = Settings(retrieval_mode="lexical")
    model = (settings.embeddings_model or "").strip()
    if not model:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "SVA_EMBEDDINGS_MODEL is required",
                }
            )
        )
        return 2

    local_repository = LocalKnowledgeRepository.load(ROOT / "knowledge")
    source_map = {source.source_id: source for source in local_repository.sources}
    documents = tuple(
        KnowledgeDocument(item=item, source=source_map[item.source_id])
        for item in local_repository.items
    )
    lexical = LexicalKnowledgeRetriever()
    hybrid = HybridKnowledgeRetriever(
        lexical_retriever=lexical,
        embedder=OpenAICompatibleEmbeddingClient(
            base_url=str(settings.embeddings_base_url),
            model=model,
            timeout_seconds=settings.embeddings_timeout_seconds,
            api_key=(
                settings.embeddings_api_key.get_secret_value()
                if settings.embeddings_api_key
                else None
            ),
        ),
        fallback_on_embedding_error=False,
        query_prefix=settings.embeddings_query_prefix,
        document_prefix=settings.embeddings_document_prefix,
        minimum_score=settings.hybrid_retrieval_minimum_score,
        ambiguity_margin=settings.hybrid_retrieval_ambiguity_margin,
    )
    policy_engine = DomainPolicyEngine()

    results: list[dict[str, object]] = []
    lexical_correct = 0
    hybrid_correct = 0
    try:
        for case in _load_cases(ROOT / "evals" / "retrieval" / "hybrid.jsonl"):
            policy_result = policy_engine.classify(case["input"])
            if policy_result.action is PolicyAction.ALLOW:
                lexical_id = _match_id(
                    lexical.search(
                        query=case["input"],
                        intent=case["intent"],
                        documents=documents,
                    )
                )
                hybrid_ranked = hybrid.rank(
                    query=case["input"],
                    intent=case["intent"],
                    documents=documents,
                )
                hybrid_id = _match_id(hybrid.select(hybrid_ranked))
            else:
                lexical_id = None
                hybrid_ranked = ()
                hybrid_id = None
            expected_id = case["expected_knowledge_id"]
            lexical_passed = lexical_id == expected_id
            hybrid_passed = hybrid_id == expected_id
            lexical_correct += int(lexical_passed)
            hybrid_correct += int(hybrid_passed)
            results.append(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "expected": expected_id,
                    "lexical": lexical_id,
                    "hybrid": hybrid_id,
                    "hybrid_top": _match_id(hybrid_ranked[0]) if hybrid_ranked else None,
                    "hybrid_top_score": hybrid_ranked[0].score if hybrid_ranked else None,
                    "hybrid_margin": (
                        round(hybrid_ranked[0].score - hybrid_ranked[1].score, 4)
                        if len(hybrid_ranked) > 1
                        else None
                    ),
                    "lexical_passed": lexical_passed,
                    "hybrid_passed": hybrid_passed,
                }
            )
    except EmbeddingServiceError:
        print(
            json.dumps(
                {
                    "status": "error",
                    "model": model,
                    "error": "embedding service unavailable or invalid",
                },
                ensure_ascii=False,
            )
        )
        return 2

    passed = hybrid_correct == len(results) and hybrid_correct >= lexical_correct
    print(
        json.dumps(
            {
                "status": "passed" if passed else "regression",
                "model": model,
                "total": len(results),
                "lexical_correct": lexical_correct,
                "hybrid_correct": hybrid_correct,
                "results": results,
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
