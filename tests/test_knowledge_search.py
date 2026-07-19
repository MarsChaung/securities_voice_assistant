from pathlib import Path

import pytest

from retrieval import KnowledgeDocument, LexicalKnowledgeRetriever, LocalKnowledgeRepository

ROOT = Path(__file__).parents[1]


def documents() -> tuple[KnowledgeDocument, ...]:
    repository = LocalKnowledgeRepository.load(ROOT / "knowledge")
    source_map = {source.source_id: source for source in repository.sources}
    return tuple(
        KnowledgeDocument(item=item, source=source_map[item.source_id]) for item in repository.items
    )


def test_retrieval_matches_basic_concept() -> None:
    match = LexicalKnowledgeRetriever().search(
        query="什麼是台股定期定額？",
        intent="general_securities_knowledge",
        documents=documents(),
    )

    assert match is not None
    assert match.document.item.knowledge_id == "K-CATHAY-DCA-001"
    assert match.score >= 0.55


@pytest.mark.parametrize(
    "query",
    [
        "證券帳戶與銀行交割帳戶有什麼不一樣？",
        "證券帳戶跟銀行交割帳戶有何不同？",
        "證券帳戶和銀行交割帳戶的差異在哪？",
        "證券帳戶、銀行交割帳戶差在哪？",
    ],
)
def test_retrieval_normalizes_comparison_vocabulary(query: str) -> None:
    match = LexicalKnowledgeRetriever().search(
        query=query,
        intent="general_securities_knowledge",
        documents=documents(),
    )

    assert match is not None
    assert match.document.item.knowledge_id == "K-CATHAY-NEWBIE-001"


def test_retrieval_normalizes_app_tutorial_vocabulary() -> None:
    match = LexicalKnowledgeRetriever().search(
        query="國泰證券 App 的定期投資怎麼操作？",
        intent="app_public_help",
        documents=documents(),
    )

    assert match is not None
    assert match.document.item.knowledge_id == "K-CATHAY-DCA-004"


def test_web_question_does_not_match_app_only_tutorial() -> None:
    match = LexicalKnowledgeRetriever().search(
        query="Web 版如何下單？",
        intent="order_entry_tutorial",
        documents=documents(),
    )

    assert match is None


def test_query_with_multiple_target_platforms_is_not_answered() -> None:
    match = LexicalKnowledgeRetriever().search(
        query="Web 版和國泰證券 App 如何下單？",
        intent="order_entry_tutorial",
        documents=documents(),
    )

    assert match is None


def test_ambiguous_query_is_not_answered() -> None:
    match = LexicalKnowledgeRetriever().search(
        query="請說明股息再投資",
        intent="general_securities_knowledge",
        documents=documents(),
    )

    assert match is None
