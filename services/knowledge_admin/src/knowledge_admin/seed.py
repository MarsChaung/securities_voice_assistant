from pathlib import Path

from .config import KnowledgeAdminSettings
from .repository import DatabaseKnowledgeRepository

DEFAULT_KNOWLEDGE_ROOT = Path(__file__).parents[4] / "knowledge"


def main() -> None:
    settings = KnowledgeAdminSettings()
    repository = DatabaseKnowledgeRepository.from_url(settings.database_url)
    sources, items = repository.seed_from_files(DEFAULT_KNOWLEDGE_ROOT)
    print(f"seed complete: sources={sources}, items={items}")


if __name__ == "__main__":
    main()
