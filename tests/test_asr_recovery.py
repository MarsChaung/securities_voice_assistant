from datetime import UTC, datetime

from orchestrator.asr import MandarinPhoneticResolver, build_asr_context
from retrieval import ASRTerm, KnowledgeDocument, QuestionVariant, QuestionVariantUsage
from test_api import published_document


def phonetic_document(
    knowledge_id: str = "K-FAQ-ASR-001",
    *,
    title: str = "假除權息說明",
    asr_terms: list[ASRTerm] | None = None,
) -> KnowledgeDocument:
    base = published_document()
    return KnowledgeDocument(
        item=base.item.model_copy(
            update={
                "knowledge_id": knowledge_id,
                "title": title,
                "standard_answer": "這是假除權息的核准說明。",
                "products": ["信用交易"],
                "allowed_intents": ["faq_general_guidance"],
                "asr_terms": asr_terms or [],
                "question_variants": [
                    QuestionVariant(
                        variant_id=f"{knowledge_id}-variant",
                        question_text="阿發，請問什麼是假除權息？",
                        usage=QuestionVariantUsage.RETRIEVAL,
                    )
                ],
            }
        ),
        source=base.source,
    )


def test_asr_context_contains_terms_but_not_standard_answer() -> None:
    document = phonetic_document()

    context = build_asr_context((document,))

    assert context == "假除權息、信用交易"
    assert document.item.standard_answer not in context


def test_asr_context_stops_at_the_configured_character_limit() -> None:
    first = phonetic_document("K-FAQ-ASR-001", title="假除權息說明")
    second = phonetic_document("K-FAQ-ASR-002", title="委任授權說明")

    context = build_asr_context((first, second), max_chars=12)

    assert context == "假除權息、信用交易"
    assert len(context) <= 12


def test_governed_asr_terms_take_priority_and_aliases_are_not_prompted() -> None:
    document = phonetic_document(
        asr_terms=[
            ASRTerm(
                term_id="asr-fake-ex-rights",
                canonical_term="假除權息",
                aliases=["甲竹全席", "甲雛全息"],
            )
        ]
    )

    context = build_asr_context((document,))
    resolution = MandarinPhoneticResolver(minimum_score=1.0).resolve(
        query="什麼是甲竹全席",
        intent="general_securities_knowledge",
        documents=(document,),
    )

    assert context == "假除權息、信用交易"
    assert "甲竹全席" not in context
    assert resolution.match is not None
    assert resolution.match.document.item.knowledge_id == document.item.knowledge_id
    assert resolution.match.score == 1.0
    assert resolution.strategy == "alias"


def test_phonetic_resolver_recovers_common_asr_homophones() -> None:
    document = phonetic_document()
    resolver = MandarinPhoneticResolver()

    first = resolver.resolve(
        query="什麼是甲竹全席",
        intent="general_securities_knowledge",
        documents=(document,),
    )
    second = resolver.resolve(
        query="什麼是甲雛全息",
        intent="general_securities_knowledge",
        documents=(document,),
    )

    assert first.match is not None
    assert first.match.document.item.knowledge_id == document.item.knowledge_id
    assert first.match.score >= 0.9
    assert second.match is not None
    assert second.match.score == 1.0


def test_phonetic_resolver_requires_a_unique_candidate() -> None:
    resolver = MandarinPhoneticResolver()
    first = phonetic_document("K-FAQ-ASR-001", title="假除權息說明")
    second = phonetic_document("K-FAQ-ASR-002", title="假除權息介紹")

    resolution = resolver.resolve(
        query="什麼是甲雛全息",
        intent="general_securities_knowledge",
        documents=(first, second),
    )

    assert resolution.match is None
    assert resolution.ambiguous is True
    assert len(resolution.candidates) == 2


def test_phonetic_resolver_rejects_short_or_low_similarity_input() -> None:
    document = phonetic_document()
    resolver = MandarinPhoneticResolver()

    short = resolver.resolve(
        query="全息",
        intent="general_securities_knowledge",
        documents=(document,),
    )
    unrelated = resolver.resolve(
        query="什麼是銀行帳戶",
        intent="general_securities_knowledge",
        documents=(document,),
    )

    assert short.match is None
    assert unrelated.match is None


def test_voice_context_uses_only_currently_eligible_documents() -> None:
    document = phonetic_document()

    class Repository:
        def eligible_documents(self, *, at: datetime) -> tuple[KnowledgeDocument, ...]:
            assert at == datetime(2026, 7, 20, tzinfo=UTC)
            return (document,)

    from orchestrator.service import TurnService

    service = TurnService(
        knowledge_repository=Repository(),
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )

    assert service.voice_asr_context() == "假除權息、信用交易"
