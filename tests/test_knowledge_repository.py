from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from pydantic import ValidationError

from retrieval import KnowledgeItem, KnowledgeStatus, LocalKnowledgeRepository

KNOWLEDGE_ROOT = Path(__file__).parents[1] / "knowledge"


def test_initial_catalog_is_traceable_and_not_runtime_eligible() -> None:
    repository = LocalKnowledgeRepository.load(KNOWLEDGE_ROOT)

    assert len(repository.sources) == 4
    assert len(repository.items) >= 15
    assert repository.eligible_items(at=datetime.fromisoformat("2026-07-15T19:30:00+08:00")) == []

    source_ids = {source.source_id for source in repository.sources}
    assert all(
        source.canonical_url is not None and urlsplit(source.canonical_url).query == ""
        for source in repository.sources
    )
    assert all(
        source.canonical_url is not None
        and urlsplit(source.canonical_url).hostname == "istockapp.cathaysec.com.tw"
        for source in repository.sources
    )
    assert {item.source_id for item in repository.items} == source_ids
    assert all(item.source_id in source_ids for item in repository.items)
    assert all(item.status is KnowledgeStatus.DRAFT for item in repository.items)
    assert all(not item.public_answer_allowed for item in repository.items)
    assert all(item.prohibited_extensions for item in repository.items)

    app_items = [item for item in repository.items if {"ios", "android"} & set(item.platforms)]
    assert len(app_items) >= 3
    assert all(item.app_versions == [] for item in app_items)
    assert all(item.status is KnowledgeStatus.DRAFT for item in app_items)


def test_published_item_requires_complete_approval_metadata() -> None:
    draft = LocalKnowledgeRepository.load(KNOWLEDGE_ROOT).items[0]
    published_data = draft.model_dump(mode="json") | {
        "status": "published",
        "public_answer_allowed": True,
    }

    with pytest.raises(ValidationError):
        KnowledgeItem.model_validate(published_data)


def test_published_item_is_removed_after_expiry_or_review_deadline() -> None:
    repository = LocalKnowledgeRepository.load(KNOWLEDGE_ROOT)
    draft = repository.items[0]
    published = KnowledgeItem.model_validate(
        draft.model_dump(mode="json")
        | {
            "status": "published",
            "public_answer_allowed": True,
            "effective_at": "2026-07-01T00:00:00+08:00",
            "expires_at": "2026-07-20T00:00:00+08:00",
            "review_at": "2026-07-18T00:00:00+08:00",
            "owner_unit": "test-owner",
            "reviewer": "test-reviewer",
            "approver": "test-approver",
            "approved_at": "2026-06-30T00:00:00+08:00",
        }
    )
    published_repository = LocalKnowledgeRepository(
        sources=repository.sources,
        items=(published,),
    )

    assert published_repository.eligible_items(
        at=datetime.fromisoformat("2026-07-15T00:00:00+08:00")
    ) == [published]
    assert (
        published_repository.eligible_items(at=datetime.fromisoformat("2026-07-19T00:00:00+08:00"))
        == []
    )


def test_app_item_requires_version_scope_before_approval() -> None:
    repository = LocalKnowledgeRepository.load(KNOWLEDGE_ROOT)
    app_draft = next(item for item in repository.items if "ios" in item.platforms)
    approved_data = app_draft.model_dump(mode="json") | {
        "status": "approved",
        "effective_at": "2026-07-15T00:00:00+08:00",
        "review_at": "2026-08-15T00:00:00+08:00",
        "owner_unit": "test-owner",
        "reviewer": "test-reviewer",
        "approver": "test-approver",
        "approved_at": "2026-07-15T00:00:00+08:00",
    }

    with pytest.raises(ValidationError):
        KnowledgeItem.model_validate(approved_data)


@pytest.mark.parametrize(
    "asr_terms,expected_error",
    [
        (
            [
                {
                    "term_id": "asr-1",
                    "canonical_term": "假除權息",
                    "aliases": ["甲竹全席", "甲 竹全席"],
                }
            ],
            "ASR 別名不得重複",
        ),
        (
            [
                {
                    "term_id": "asr-1",
                    "canonical_term": "假除權息",
                    "aliases": ["複委託"],
                },
                {
                    "term_id": "asr-2",
                    "canonical_term": "複委託",
                    "aliases": [],
                },
            ],
            "語音辨識詞彙衝突",
        ),
    ],
)
def test_asr_terms_reject_ambiguous_aliases(
    asr_terms: list[dict[str, object]],
    expected_error: str,
) -> None:
    draft = LocalKnowledgeRepository.load(KNOWLEDGE_ROOT).items[0]

    with pytest.raises(ValidationError, match=expected_error):
        KnowledgeItem.model_validate(draft.model_dump(mode="json") | {"asr_terms": asr_terms})
