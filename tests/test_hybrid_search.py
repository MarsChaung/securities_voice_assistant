from collections.abc import Sequence
from pathlib import Path

import pytest

from retrieval import (
    EmbeddingServiceError,
    HybridKnowledgeRetriever,
    KnowledgeDocument,
    LexicalKnowledgeRetriever,
    LocalKnowledgeRepository,
)

ROOT = Path(__file__).parents[1]


def documents() -> tuple[KnowledgeDocument, ...]:
    repository = LocalKnowledgeRepository.load(ROOT / "knowledge")
    source_map = {source.source_id: source for source in repository.sources}
    return tuple(
        KnowledgeDocument(item=item, source=source_map[item.source_id]) for item in repository.items
    )


class SyntheticEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.calls.append(tuple(texts))
        return tuple(self._vector(text) for text in texts)

    @staticmethod
    def _vector(text: str) -> tuple[float, ...]:
        if "記錄股票買賣" in text or "證券帳戶與銀行交割帳戶的差別" in text:
            return (1.0, 0.0, 0.0)
        if "無關的合成問題" in text:
            return (0.0, 0.0, 1.0)
        return (0.0, 1.0, 0.0)


class FailingEmbeddingProvider:
    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        raise EmbeddingServiceError("synthetic provider failure")


def test_hybrid_retrieval_recovers_semantic_paraphrase_and_caches_documents() -> None:
    embedder = SyntheticEmbeddingProvider()
    retriever = HybridKnowledgeRetriever(embedder=embedder)
    query = "什麼是用來記錄股票買賣的戶頭，以及負責扣款的銀行戶頭？"

    lexical_match = LexicalKnowledgeRetriever().search(
        query=query,
        intent="general_securities_knowledge",
        documents=documents(),
    )
    first_match = retriever.search(
        query=query,
        intent="general_securities_knowledge",
        documents=documents(),
    )
    second_match = retriever.search(
        query=query,
        intent="general_securities_knowledge",
        documents=documents(),
    )

    assert lexical_match is None
    assert first_match is not None
    assert first_match.document.item.knowledge_id == "K-CATHAY-NEWBIE-001"
    assert second_match is not None
    assert len(embedder.calls[0]) > 1
    assert embedder.calls[1] == (query,)


def test_hybrid_retrieval_falls_back_to_lexical_when_embedding_service_fails() -> None:
    retriever = HybridKnowledgeRetriever(embedder=FailingEmbeddingProvider())

    match = retriever.search(
        query="什麼是證券帳戶與銀行交割帳戶的差別？",
        intent="general_securities_knowledge",
        documents=documents(),
    )

    assert match is not None
    assert match.document.item.knowledge_id == "K-CATHAY-NEWBIE-001"


def test_embedding_failure_does_not_apply_the_lower_hybrid_threshold_to_lexical_scores() -> None:
    retriever = HybridKnowledgeRetriever(embedder=FailingEmbeddingProvider())

    match = retriever.search(
        query="請說明股息再投資",
        intent="general_securities_knowledge",
        documents=documents(),
    )

    assert match is None


def test_hybrid_retrieval_can_disable_fallback_for_offline_evaluation() -> None:
    retriever = HybridKnowledgeRetriever(
        embedder=FailingEmbeddingProvider(),
        fallback_on_embedding_error=False,
    )

    with pytest.raises(EmbeddingServiceError, match="synthetic provider failure"):
        retriever.search(
            query="什麼是證券帳戶與銀行交割帳戶的差別？",
            intent="general_securities_knowledge",
            documents=documents(),
        )


def test_hybrid_retrieval_keeps_low_confidence_semantic_query_unanswered() -> None:
    retriever = HybridKnowledgeRetriever(embedder=SyntheticEmbeddingProvider())

    match = retriever.search(
        query="什麼是無關的合成問題？",
        intent="general_securities_knowledge",
        documents=documents(),
    )

    assert match is None


def test_hybrid_retrieval_applies_model_specific_prefixes() -> None:
    embedder = SyntheticEmbeddingProvider()
    retriever = HybridKnowledgeRetriever(
        embedder=embedder,
        query_prefix="query: ",
        document_prefix="passage: ",
    )

    retriever.search(
        query="什麼是證券帳戶？",
        intent="general_securities_knowledge",
        documents=documents(),
    )

    assert embedder.calls[0][0] == "query: 什麼是證券帳戶？"
    assert all(text.startswith("passage: ") for text in embedder.calls[0][1:])
