from enum import StrEnum
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator


class Decision(StrEnum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    REFUSE = "refuse"
    HANDOFF = "handoff"


class TurnRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=4_000)
    channel: Literal["web", "voice"] = "web"


class Citation(BaseModel):
    source_id: str = Field(pattern=r"^SRC-[A-Z0-9-]+$")
    source_uri: str | None = None
    source_title: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)

    @field_validator("source_uri")
    @classmethod
    def require_https_source_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("citation source_uri 必須使用完整 HTTPS URL")
        return value


class AnswerContract(BaseModel):
    decision: Decision
    intent: str = Field(min_length=1)
    policy_rule_id: str = Field(min_length=1)
    answer_id: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    knowledge_versions: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def require_evidence_for_answer(self) -> "AnswerContract":
        if self.decision is Decision.ANSWER:
            if not self.answer_id:
                raise ValueError("decision=answer 必須提供 answer_id")
            if not self.source_ids or not self.knowledge_versions or not self.citations:
                raise ValueError("decision=answer 必須提供來源與知識版本")
            if self.source_ids != [citation.source_id for citation in self.citations]:
                raise ValueError("source_ids 必須與 citations 一致")
        return self


class TurnResponse(BaseModel):
    turn_id: str
    result: AnswerContract


class TurnFeedback(BaseModel):
    rating: Literal["helpful", "not_helpful"]
