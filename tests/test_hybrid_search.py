from collections.abc import Sequence
from pathlib import Path

import pytest

from retrieval import (
    EmbeddingServiceError,
    HybridKnowledgeRetriever,
    KnowledgeDocument,
    LexicalKnowledgeRetriever,
    LocalKnowledgeRepository,
    QuestionVariant,
    QuestionVariantUsage,
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
        if "阿發口語問法七七" in text:
            return (0.0, 0.0, 1.0)
        if "記錄股票買賣" in text or "證券帳戶與銀行交割帳戶的差別" in text:
            return (1.0, 0.0, 0.0)
        if "無關的合成問題" in text:
            return (0.0, 0.0, 1.0)
        return (0.0, 1.0, 0.0)


class FailingEmbeddingProvider:
    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        raise EmbeddingServiceError("synthetic provider failure")


class FailSecondBatchEmbeddingProvider(SyntheticEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.failure_enabled = True

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if self.failure_enabled and len(self.calls) == 1:
            self.calls.append(tuple(texts))
            raise EmbeddingServiceError("synthetic second batch failure")
        return super().embed(texts)


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
    assert all(len(call) <= 8 for call in embedder.calls[:-2])
    assert embedder.calls[-2:] == [(query,), (query,)]


def test_document_warmup_keeps_successful_batches_after_later_failure() -> None:
    embedder = FailSecondBatchEmbeddingProvider()
    retriever = HybridKnowledgeRetriever(embedder=embedder)
    source_documents = documents()
    representation_count = sum(
        1
        + sum(
            variant.usage is QuestionVariantUsage.RETRIEVAL
            for variant in document.item.question_variants
        )
        for document in source_documents
    )

    with pytest.raises(EmbeddingServiceError, match="second batch failure"):
        retriever.warm(source_documents)

    embedder.failure_enabled = False
    warmed = retriever.warm(source_documents)

    assert warmed == representation_count - 8
    assert len(embedder.calls[0]) == 8


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


def test_embedding_fallback_ignores_conversational_prefix_for_faq_match() -> None:
    base_document = documents()[0]
    faq_document = KnowledgeDocument(
        item=base_document.item.model_copy(
            update={
                "title": "線上開戶申請資格",
                "allowed_intents": ["faq_general_guidance"],
                "question_variants": [
                    QuestionVariant(
                        variant_id="account-opening-qualification",
                        question_text="線上開戶資格是什麼",
                        usage=QuestionVariantUsage.RETRIEVAL,
                    )
                ],
            }
        ),
        source=base_document.source,
    )

    match = HybridKnowledgeRetriever(
        embedder=FailingEmbeddingProvider()
    ).search(
        query="如果我想線上開戶，資格是什麼",
        intent="account_opening_general",
        documents=(faq_document,),
    )

    assert match is not None


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


def test_contact_phone_query_does_not_fall_back_to_a_fee_answer() -> None:
    base_document = documents()[0]
    fee_document = KnowledgeDocument(
        item=base_document.item.model_copy(
            update={
                "knowledge_id": "K-FAQ-FEE-001",
                "title": "台股人工下單手續費多少",
                "standard_answer": "台股非電子交易手續費為千分之1.425。",
                "allowed_intents": ["faq_general_guidance"],
                "question_variants": [
                    QuestionVariant(
                        variant_id="fee-question",
                        question_text="台股人工下單手續費多少",
                        usage=QuestionVariantUsage.RETRIEVAL,
                    )
                ],
            }
        ),
        source=base_document.source,
    )
    retriever = HybridKnowledgeRetriever(embedder=SyntheticEmbeddingProvider())

    match = retriever.search(
        query="臺股人工接單中心的電話是多少？",
        intent="public_service_information",
        documents=(fee_document,),
    )

    assert match is None


def test_contact_phone_query_matches_a_published_phone_answer() -> None:
    base_document = documents()[0]
    phone_document = KnowledgeDocument(
        item=base_document.item.model_copy(
            update={
                "knowledge_id": "K-FAQ-PHONE-001",
                "title": "台股人工接單中心電話",
                "standard_answer": "市話請撥：412 8881。手機請撥：02 412 8881。",
                "allowed_intents": ["faq_general_guidance"],
                "question_variants": [
                    QuestionVariant(
                        variant_id="phone-question",
                        question_text="請問人工下單專線是？",
                        usage=QuestionVariantUsage.RETRIEVAL,
                    )
                ],
            }
        ),
        source=base_document.source,
    )
    retriever = HybridKnowledgeRetriever(embedder=SyntheticEmbeddingProvider())

    match = retriever.search(
        query="人工交易的接單電話是什麼？",
        intent="public_service_information",
        documents=(phone_document,),
    )

    assert match is not None
    assert match.document.item.knowledge_id == "K-FAQ-PHONE-001"


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

    assert embedder.calls[-1] == ("query: 什麼是證券帳戶？",)
    assert all(
        text.startswith("passage: ")
        for call in embedder.calls[:-1]
        for text in call
    )


def test_hybrid_retrieval_embeds_runtime_variants_as_separate_representations() -> None:
    embedder = SyntheticEmbeddingProvider()
    base_document = documents()[0]
    item = base_document.item.model_copy(
        update={
            "question_variants": [
                QuestionVariant(
                    variant_id="runtime",
                    question_text="阿發口語問法七七",
                    usage=QuestionVariantUsage.RETRIEVAL,
                ),
                QuestionVariant(
                    variant_id="evaluation",
                    question_text="阿發評測問法八八",
                    usage=QuestionVariantUsage.EVALUATION_ONLY,
                ),
            ]
        }
    )
    document = KnowledgeDocument(item=item, source=base_document.source)

    match = HybridKnowledgeRetriever(embedder=embedder).search(
        query="阿發口語問法七七",
        intent=item.allowed_intents[0],
        documents=(document,),
    )

    assert match is not None
    embedded_texts = tuple(text for call in embedder.calls for text in call)
    assert "阿發口語問法七七" in embedded_texts
    assert "阿發評測問法八八" not in embedded_texts
