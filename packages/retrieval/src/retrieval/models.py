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


class QuestionVariantUsage(StrEnum):
    RETRIEVAL = "retrieval"
    EVALUATION_ONLY = "evaluation_only"
    EXCLUDED = "excluded"


class QuestionVariant(BaseModel):
    variant_id: str = Field(min_length=1, max_length=36)
    question_text: str = Field(min_length=2, max_length=500)
    usage: QuestionVariantUsage = QuestionVariantUsage.RETRIEVAL

    @field_validator("question_text")
    @classmethod
    def normalize_question_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized) < 2:
            raise ValueError("問句變體不得為空白")
        return normalized


class ASRTerm(BaseModel):
    term_id: str = Field(min_length=1, max_length=36)
    canonical_term: str = Field(min_length=2, max_length=100)
    aliases: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("canonical_term")
    @classmethod
    def normalize_canonical_term(cls, value: str) -> str:
        normalized = " ".join(value.split()).strip("「」")
        if len(normalized) < 2:
            raise ValueError("語音辨識詞彙至少需要 2 個字元")
        return normalized

    @field_validator("aliases")
    @classmethod
    def normalize_aliases(cls, values: list[str]) -> list[str]:
        aliases = [" ".join(value.split()).strip("「」") for value in values]
        if any(len(alias) < 2 or len(alias) > 100 for alias in aliases):
            raise ValueError("ASR 別名長度必須介於 2 到 100 個字元")
        return aliases

    @model_validator(mode="after")
    def enforce_unique_aliases(self) -> "ASRTerm":
        canonical_key = _normalize_asr_text(self.canonical_term)
        alias_keys = [_normalize_asr_text(alias) for alias in self.aliases]
        if any(not key for key in alias_keys):
            raise ValueError("ASR 別名必須包含文字或數字")
        if canonical_key in alias_keys:
            raise ValueError("ASR 別名不得與語音辨識詞彙相同")
        if len(alias_keys) != len(set(alias_keys)):
            raise ValueError("同一語音辨識詞彙的 ASR 別名不得重複")
        return self


class KnowledgeSource(BaseModel):
    source_id: str = Field(pattern=r"^SRC-[A-Z0-9-]+$")
    supplied_url: str | None = None
    canonical_url: str | None = None
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    source_type: Literal["official_web", "approved_internal_faq", "local_import"]
    retrieved_at: datetime
    topics: list[str] = Field(min_length=1)
    status: SourceStatus = SourceStatus.ACTIVE
    notes: str | None = None

    @field_validator("supplied_url", "canonical_url", mode="before")
    @classmethod
    def normalize_optional_url(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def require_traceable_source_url(self) -> "KnowledgeSource":
        if self.source_type != "local_import" and (
            self.supplied_url is None or self.canonical_url is None
        ):
            raise ValueError("非本機匯入的 knowledge source 必須提供 HTTPS URL")
        for value in (self.supplied_url, self.canonical_url):
            if value is not None:
                self._require_https_url(value)
        return self

    @staticmethod
    def _require_https_url(value: str) -> None:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("knowledge source 必須使用完整 HTTPS URL")


class KnowledgeItem(BaseModel):
    knowledge_id: str = Field(pattern=r"^K-[A-Z0-9-]+$")
    title: str = Field(min_length=1)
    standard_answer: str = Field(min_length=1)
    source_id: str = Field(pattern=r"^SRC-[A-Z0-9-]+$")
    source_uri: str | None = None
    source_locator: str = Field(min_length=1)
    source_type: Literal["official_web", "approved_internal_faq", "local_import"]
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
    question_variants: list[QuestionVariant] = Field(default_factory=list)
    asr_terms: list[ASRTerm] = Field(default_factory=list, max_length=50)

    @field_validator("source_uri", mode="before")
    @classmethod
    def normalize_optional_source_uri(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def enforce_lifecycle_requirements(self) -> "KnowledgeItem":
        if self.source_type != "local_import" and self.source_uri is None:
            raise ValueError("非本機匯入知識必須提供 source_uri")
        if self.source_uri is not None:
            parsed = urlsplit(self.source_uri)
            if parsed.scheme != "https" or not parsed.hostname:
                raise ValueError("source_uri 必須使用完整 HTTPS URL")

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

        canonical_keys: set[str] = set()
        alias_keys: set[str] = set()
        for term in self.asr_terms:
            canonical_key = _normalize_asr_text(term.canonical_term)
            if canonical_key in canonical_keys:
                raise ValueError("同一知識項目的語音辨識詞彙不得重複")
            canonical_keys.add(canonical_key)
            for alias in term.aliases:
                alias_key = _normalize_asr_text(alias)
                if alias_key in alias_keys:
                    raise ValueError("同一知識項目的 ASR 別名不得重複")
                alias_keys.add(alias_key)
        if canonical_keys.intersection(alias_keys):
            raise ValueError("ASR 別名不得與同一知識項目的語音辨識詞彙衝突")

        return self


@dataclass(frozen=True)
class KnowledgeDocument:
    item: KnowledgeItem
    source: KnowledgeSource


@dataclass(frozen=True)
class RetrievalMatch:
    document: KnowledgeDocument
    score: float


def _normalize_asr_text(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())
