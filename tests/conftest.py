from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from knowledge_admin.database import Base
from knowledge_admin.repository import DatabaseKnowledgeRepository
from observability import ShadowReviewBase

ROOT = Path(__file__).parents[1]


@pytest.fixture
def knowledge_store() -> DatabaseKnowledgeRepository:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    ShadowReviewBase.metadata.create_all(engine)
    repository = DatabaseKnowledgeRepository(engine)
    repository.seed_from_files(ROOT / "knowledge")
    return repository
