from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator


class SourceStatus(StrEnum):
    ACTIVE = "active"
    INACCESSIBLE = "inaccessible"
    RETIRED = "retired"


class KnowledgeStatus(StrEnum):
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    EXPIRED = "expired"
    REVOKED = "revoked"


class KnowledgeSource(BaseModel):
    source_id: str = Field(pattern=r"^SRC-[A-Z0-9-]+$")
    supplied_url: str
    canonical_url: str
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    source_type: Literal["official_web"]
    retrieved_at: datetime
    topics: list[str] = Field(min_length=1)
    status: SourceStatus = SourceStatus.ACTIVE
    notes: str | None = None

    @field_validator("supplied_url", "canonical_url")
    @classmethod
    def require_https_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("knowledge source 必須使用完整 HTTPS URL")
        return value


class KnowledgeItem(BaseModel):
    knowledge_id: str = Field(pattern=r"^K-[A-Z0-9-]+$")
    title: str = Field(min_length=1)
    standard_answer: str = Field(min_length=1)
    source_id: str = Field(pattern=r"^SRC-[A-Z0-9-]+$")
    source_uri: str
    source_locator: str = Field(min_length=1)
    source_type: Literal["official_web"]
    products: list[str] = Field(default_factory=list)
    platforms: list[str] = Field(default_factory=list)
    app_versions: list[str] = Field(default_factory=list)
    effective_at: datetime | None = None
    expires_at: datetime | None = None
    review_at: datetime | None = None
    owner_unit: str | None = None
    author: str = Field(min_length=1)
    reviewer: str | None = None
    approver: str | None = None
    approved_at: datetime | None = None
    version: str = Field(min_length=1)
    change_summary: str = ""
    previous_version: str | None = None
    status: KnowledgeStatus
    public_answer_allowed: bool
    allowed_intents: list[str] = Field(min_length=1)
    prohibited_extensions: list[str] = Field(min_length=1)

    @field_validator("source_uri")
    @classmethod
    def require_https_source_uri(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("source_uri 必須使用完整 HTTPS URL")
        return value

    @model_validator(mode="after")
    def enforce_lifecycle_requirements(self) -> "KnowledgeItem":
        if (
            self.status in {KnowledgeStatus.DRAFT, KnowledgeStatus.REVIEW}
            and self.public_answer_allowed
        ):
            raise ValueError("draft/review knowledge 不得允許 runtime 對外回答")

        if self.status in {KnowledgeStatus.APPROVED, KnowledgeStatus.PUBLISHED}:
            required_values = (
                self.effective_at,
                self.review_at,
                self.owner_unit,
                self.reviewer,
                self.approver,
                self.approved_at,
            )
            if any(value is None for value in required_values):
                raise ValueError("approved/published knowledge 必須具備完整核准中繼資料")

            app_platforms = {"ios", "android"}
            if (
                app_platforms.intersection(platform.casefold() for platform in self.platforms)
                and not self.app_versions
            ):
                raise ValueError("App knowledge 核准前必須指定適用版本")

        if self.status is KnowledgeStatus.PUBLISHED and not self.public_answer_allowed:
            raise ValueError("published knowledge 必須明確允許 runtime 回答")

        if self.effective_at and self.expires_at and self.expires_at <= self.effective_at:
            raise ValueError("expires_at 必須晚於 effective_at")

        return self


@dataclass(frozen=True)
class KnowledgeDocument:
    item: KnowledgeItem
    source: KnowledgeSource


@dataclass(frozen=True)
class RetrievalMatch:
    document: KnowledgeDocument
    score: float
