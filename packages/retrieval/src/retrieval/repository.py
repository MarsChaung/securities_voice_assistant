from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import TypeAdapter

from .models import KnowledgeItem, KnowledgeSource, KnowledgeStatus, SourceStatus


@dataclass(frozen=True)
class LocalKnowledgeRepository:
    sources: tuple[KnowledgeSource, ...]
    items: tuple[KnowledgeItem, ...]

    @classmethod
    def load(cls, root: Path) -> "LocalKnowledgeRepository":
        source_adapter = TypeAdapter(list[KnowledgeSource])
        item_adapter = TypeAdapter(list[KnowledgeItem])

        sources = source_adapter.validate_json((root / "sources.json").read_text(encoding="utf-8"))
        items: list[KnowledgeItem] = []
        for path in sorted((root / "drafts").glob("*.json")):
            items.extend(item_adapter.validate_json(path.read_text(encoding="utf-8")))

        cls._validate_relationships(sources, items)
        return cls(sources=tuple(sources), items=tuple(items))

    @staticmethod
    def _validate_relationships(
        sources: list[KnowledgeSource],
        items: list[KnowledgeItem],
    ) -> None:
        source_map = {source.source_id: source for source in sources}
        if len(source_map) != len(sources):
            raise ValueError("knowledge source_id 不得重複")

        knowledge_ids = {item.knowledge_id for item in items}
        if len(knowledge_ids) != len(items):
            raise ValueError("knowledge_id 不得重複")

        for item in items:
            source = source_map.get(item.source_id)
            if source is None:
                raise ValueError(f"{item.knowledge_id} 引用了不存在的 source_id")
            if item.source_uri != source.canonical_url:
                raise ValueError(f"{item.knowledge_id} 的 source_uri 與 source catalog 不一致")

    def eligible_items(self, *, at: datetime) -> list[KnowledgeItem]:
        active_sources = {
            source.source_id for source in self.sources if source.status is SourceStatus.ACTIVE
        }
        return [
            item
            for item in self.items
            if item.source_id in active_sources
            and item.status is KnowledgeStatus.PUBLISHED
            and item.public_answer_allowed
            and item.effective_at is not None
            and item.effective_at <= at
            and (item.expires_at is None or at < item.expires_at)
            and item.review_at is not None
            and at <= item.review_at
        ]
