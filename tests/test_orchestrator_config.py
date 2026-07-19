import pytest
from pydantic import ValidationError

from orchestrator.answering import OpenAICompatibleAnswerComposer
from orchestrator.api import (
    _build_answer_composer,
    _build_intent_router,
    _build_knowledge_retriever,
)
from orchestrator.config import Settings
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
