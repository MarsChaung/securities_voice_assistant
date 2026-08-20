import pytest
from pydantic import ValidationError

from orchestrator.answering import (
    OpenAICompatibleAnswerComposer,
    OpenAICompatibleNaturalAnswerComposer,
)
from orchestrator.api import (
    _build_answer_composer,
    _build_conversation_semantic_analyzer,
    _build_intent_router,
    _build_knowledge_retriever,
    _build_natural_answer_composer,
)
from orchestrator.config import Settings
from orchestrator.conversation import OpenAICompatibleConversationSemanticAnalyzer
from orchestrator.intent_routing import OpenAICompatibleIntentRouter
from retrieval import HybridKnowledgeRetriever, LexicalKnowledgeRetriever


def test_lexical_retrieval_remains_the_default() -> None:
    assert Settings.model_fields["retrieval_mode"].default == "lexical"
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
    )

    assert isinstance(_build_knowledge_retriever(settings), LexicalKnowledgeRetriever)


def test_exact_answer_mode_remains_the_default() -> None:
    assert Settings.model_fields["answer_mode"].default == "exact"
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        answer_mode="exact",
    )

    assert _build_answer_composer(settings) is None


def test_system_diagnostics_are_disabled_by_default() -> None:
    assert Settings.model_fields["system_diagnostics_enabled"].default is False
    assert Settings.model_fields["system_diagnostics_timeout_seconds"].default == 30.0


def test_hybrid_retrieval_requires_an_embedding_model() -> None:
    with pytest.raises(ValidationError, match="SVA_EMBEDDINGS_MODEL"):
        Settings(
            database_url="sqlite+pysqlite:///:memory:",
            retrieval_mode="hybrid",
            embeddings_model="",
        )


def test_hybrid_retrieval_can_be_enabled_explicitly() -> None:
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="hybrid",
        embeddings_model="synthetic-embedding-model",
    )

    assert isinstance(_build_knowledge_retriever(settings), HybridKnowledgeRetriever)


def test_controlled_generation_requires_a_model() -> None:
    with pytest.raises(ValidationError, match="SVA_ANSWER_LLM_MODEL"):
        Settings(
            database_url="sqlite+pysqlite:///:memory:",
            retrieval_mode="lexical",
            answer_mode="controlled_llm",
            answer_llm_model="",
        )


def test_controlled_generation_can_be_enabled_explicitly() -> None:
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        answer_mode="controlled_llm",
        answer_llm_model="synthetic-model",
    )

    assert isinstance(_build_answer_composer(settings), OpenAICompatibleAnswerComposer)


def test_shadow_generation_builds_a_composer_without_changing_the_default() -> None:
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        answer_mode="shadow_llm",
        answer_llm_model="synthetic-model",
        intent_router_mode="disabled",
    )

    assert isinstance(_build_answer_composer(settings), OpenAICompatibleAnswerComposer)


def test_natural_answer_mode_is_disabled_by_default() -> None:
    assert Settings.model_fields["natural_answer_enabled"].default is False
    assert _build_natural_answer_composer(Settings(natural_answer_enabled=False)) is None


def test_voice_test_content_logging_is_disabled_by_default() -> None:
    assert Settings.model_fields["voice_test_content_logging_enabled"].default is False


def test_natural_answer_mode_requires_and_builds_answer_model() -> None:
    with pytest.raises(ValidationError, match="SVA_ANSWER_LLM_MODEL"):
        Settings(natural_answer_enabled=True, answer_llm_model="")

    settings = Settings(
        natural_answer_enabled=True,
        answer_llm_model="synthetic-model",
    )

    assert isinstance(
        _build_natural_answer_composer(settings),
        OpenAICompatibleNaturalAnswerComposer,
    )


def test_intent_router_is_disabled_by_default() -> None:
    assert Settings.model_fields["intent_router_mode"].default == "disabled"
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        intent_router_mode="disabled",
    )

    assert _build_intent_router(settings) is None


def test_enabled_intent_router_requires_a_model() -> None:
    with pytest.raises(ValidationError, match="SVA_INTENT_LLM_MODEL"):
        Settings(
            database_url="sqlite+pysqlite:///:memory:",
            retrieval_mode="lexical",
            intent_router_mode="shadow",
            intent_llm_model="",
        )


def test_controlled_intent_router_can_be_enabled_explicitly() -> None:
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        intent_router_mode="controlled",
        intent_llm_model="synthetic-model",
    )

    assert isinstance(_build_intent_router(settings), OpenAICompatibleIntentRouter)


def test_conversation_semantic_resolver_is_disabled_by_default() -> None:
    assert Settings.model_fields["conversation_semantic_mode"].default == "disabled"
    assert (
        _build_conversation_semantic_analyzer(Settings(conversation_semantic_mode="disabled"))
        is None
    )


def test_conversation_semantic_resolver_requires_and_builds_model() -> None:
    with pytest.raises(ValidationError, match="SVA_CONVERSATION_LLM_MODEL"):
        Settings(
            conversation_semantic_mode="shadow",
            conversation_llm_model="",
        )

    settings = Settings(
        conversation_semantic_mode="controlled",
        conversation_llm_model="synthetic-model",
    )

    assert isinstance(
        _build_conversation_semantic_analyzer(settings),
        OpenAICompatibleConversationSemanticAnalyzer,
    )


def test_voice_mode_requires_asr_and_tts_models() -> None:
    with pytest.raises(ValidationError, match="SVA_ASR_MODEL"):
        Settings(
            database_url="sqlite+pysqlite:///:memory:",
            retrieval_mode="lexical",
            voice_enabled=True,
            asr_model="",
            tts_model="",
        )


def test_voice_clone_requires_reference_audio_and_text_together() -> None:
    with pytest.raises(ValidationError, match="SVA_TTS_REF_AUDIO"):
        Settings(
            database_url="sqlite+pysqlite:///:memory:",
            retrieval_mode="lexical",
            tts_ref_audio="/private/reference.wav",
            tts_ref_text=None,
        )


def test_barge_in_default_mode_is_restricted_to_governed_presets() -> None:
    with pytest.raises(ValidationError, match="barge_in_default_mode"):
        Settings(
            database_url="sqlite+pysqlite:///:memory:",
            retrieval_mode="lexical",
            barge_in_default_mode="custom",  # type: ignore[arg-type]
        )


def test_asr_endpoint_grace_is_bounded() -> None:
    assert Settings(asr_endpoint_grace_ms=1200).asr_endpoint_grace_ms == 1200

    with pytest.raises(ValidationError, match="asr_endpoint_grace_ms"):
        Settings(asr_endpoint_grace_ms=5001)
