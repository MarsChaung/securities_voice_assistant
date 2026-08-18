import hashlib
import math
from collections.abc import Sequence
from threading import Lock

from .embeddings import EmbeddingProvider, EmbeddingServiceError, EmbeddingVector
from .models import KnowledgeDocument, QuestionVariantUsage, RetrievalMatch
from .search import LexicalKnowledgeRetriever


class HybridKnowledgeRetriever:
    """以詞彙與本機 embedding 分數排序；embedding 故障時退回既有詞彙檢索。"""

    _WARMUP_BATCH_SIZE = 8

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        lexical_retriever: LexicalKnowledgeRetriever | None = None,
        lexical_weight: float = 0.4,
        semantic_weight: float = 0.6,
        fallback_on_embedding_error: bool = True,
        query_prefix: str = "",
        document_prefix: str = "",
        minimum_score: float = 0.4,
        ambiguity_margin: float = 0.02,
    ) -> None:
        if lexical_weight < 0 or semantic_weight < 0 or lexical_weight + semantic_weight == 0:
            raise ValueError("hybrid retrieval weights 必須為非負且總和大於 0")
        weight_total = lexical_weight + semantic_weight
        if not 0 <= minimum_score <= 1 or not 0 <= ambiguity_margin <= 1:
            raise ValueError("hybrid retrieval 門檻必須介於 0 與 1")
        self._embedder = embedder
        self._lexical = lexical_retriever or LexicalKnowledgeRetriever()
        self._lexical_weight = lexical_weight / weight_total
        self._semantic_weight = semantic_weight / weight_total
        self._fallback_on_embedding_error = fallback_on_embedding_error
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._minimum_score = minimum_score
        self._ambiguity_margin = ambiguity_margin
        self._embedding_cache: dict[str, EmbeddingVector] = {}
        self._cache_lock = Lock()
        self._warmup_lock = Lock()

    def warm(self, documents: Sequence[KnowledgeDocument]) -> int:
        """分批預熱文件向量；每批成功後立即保留，避免大型批次逾時後全部重算。"""
        warmed = 0
        with self._warmup_lock:
            representations = tuple(
                (
                    _cache_key(document, document_text),
                    document_text,
                )
                for document in documents
                for document_text in (
                    f"{self._document_prefix}{text}" for text in _document_texts(document)
                )
            )
            with self._cache_lock:
                missing = tuple(
                    representation
                    for representation in representations
                    if representation[0] not in self._embedding_cache
                )

            for offset in range(0, len(missing), self._WARMUP_BATCH_SIZE):
                batch = missing[offset : offset + self._WARMUP_BATCH_SIZE]
                embedded = self._embedder.embed(tuple(text for _, text in batch))
                if len(embedded) != len(batch):
                    raise EmbeddingServiceError(
                        "embedding provider returned unexpected vector count"
                    )
                with self._cache_lock:
                    for (key, _), vector in zip(batch, embedded, strict=True):
                        self._embedding_cache[key] = vector
                warmed += len(batch)
        return warmed

    def search(
        self,
        *,
        query: str,
        intent: str,
        documents: Sequence[KnowledgeDocument],
    ) -> RetrievalMatch | None:
        lexical_ranked = self._lexical.rank(
            query=query,
            intent=intent,
            documents=documents,
        )
        if not lexical_ranked:
            return None
        try:
            hybrid_ranked = self._rank_from_lexical(query, lexical_ranked)
        except EmbeddingServiceError:
            if not self._fallback_on_embedding_error:
                raise
            return self._lexical.select(lexical_ranked)
        return self.select(hybrid_ranked)

    def rank(
        self,
        *,
        query: str,
        intent: str,
        documents: Sequence[KnowledgeDocument],
    ) -> tuple[RetrievalMatch, ...]:
        lexical_ranked = self._lexical.rank(
            query=query,
            intent=intent,
            documents=documents,
        )
        if not lexical_ranked:
            return ()

        try:
            return self._rank_from_lexical(query, lexical_ranked)
        except EmbeddingServiceError:
            if not self._fallback_on_embedding_error:
                raise
            return lexical_ranked

    def select(self, ranked: Sequence[RetrievalMatch]) -> RetrievalMatch | None:
        if not ranked or ranked[0].score < self._minimum_score:
            return None
        if len(ranked) > 1 and ranked[0].score - ranked[1].score < self._ambiguity_margin:
            return None
        return ranked[0]

    def _rank_from_lexical(
        self,
        query: str,
        lexical_ranked: Sequence[RetrievalMatch],
    ) -> tuple[RetrievalMatch, ...]:
        candidate_documents = tuple(match.document for match in lexical_ranked)
        query_vector, document_vectors = self._vectors_for(query, candidate_documents)
        return tuple(
            sorted(
                (
                    RetrievalMatch(
                        document=lexical_match.document,
                        score=round(
                            min(
                                1.0,
                                lexical_match.score * self._lexical_weight
                                + max(
                                    _cosine_similarity(query_vector, representation)
                                    for representation in document_vector
                                )
                                * self._semantic_weight,
                            ),
                            4,
                        ),
                    )
                    for lexical_match, document_vector in zip(
                        lexical_ranked,
                        document_vectors,
                        strict=True,
                    )
                ),
                key=lambda match: (-match.score, match.document.item.knowledge_id),
            )
        )

    def _vectors_for(
        self,
        query: str,
        documents: Sequence[KnowledgeDocument],
    ) -> tuple[EmbeddingVector, tuple[tuple[EmbeddingVector, ...], ...]]:
        self.warm(documents)
        document_texts = tuple(
            tuple(f"{self._document_prefix}{text}" for text in _document_texts(document))
            for document in documents
        )
        cache_keys = tuple(
            tuple(_cache_key(document, document_text) for document_text in texts)
            for document, texts in zip(documents, document_texts, strict=True)
        )
        with self._cache_lock:
            cached_vectors = tuple(
                tuple(self._embedding_cache.get(key) for key in keys) for keys in cache_keys
            )

        embedded = self._embedder.embed((f"{self._query_prefix}{query}",))
        if len(embedded) != 1:
            raise EmbeddingServiceError("embedding provider returned unexpected vector count")

        query_vector = embedded[0]
        if any(vector is None for vectors in cached_vectors for vector in vectors):
            raise EmbeddingServiceError("embedding cache returned incomplete vectors")
        return query_vector, tuple(
            tuple(vector for vector in vectors if vector is not None) for vectors in cached_vectors
        )


def _document_texts(document: KnowledgeDocument) -> tuple[str, ...]:
    item = document.item
    base_text = "\n".join(
        (
            item.title,
            item.standard_answer,
            " ".join(item.products),
            " ".join(item.platforms),
            item.source_locator,
        )
    )
    return (
        base_text,
        *(
            variant.question_text
            for variant in item.question_variants
            if variant.usage is QuestionVariantUsage.RETRIEVAL
        ),
    )


def _cache_key(document: KnowledgeDocument, document_text: str) -> str:
    item = document.item
    content_digest = hashlib.sha256(document_text.encode("utf-8")).hexdigest()
    return f"{item.knowledge_id}:{item.version}:{content_digest}"


def _cosine_similarity(left: EmbeddingVector, right: EmbeddingVector) -> float:
    if len(left) != len(right) or not left:
        raise EmbeddingServiceError("embedding vectors have incompatible dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    similarity = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return max(0.0, min(1.0, similarity))
