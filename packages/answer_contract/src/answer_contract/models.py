from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Decision(StrEnum):
    ANSWER = "answer"
    REFUSE = "refuse"
    HANDOFF = "handoff"


class TurnRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=4_000)
    channel: Literal["web", "phone"] = "web"


class AnswerContract(BaseModel):
    decision: Decision
    intent: str = Field(min_length=1)
    policy_rule_id: str = Field(min_length=1)
    answer_id: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    knowledge_versions: list[str] = Field(default_factory=list)
    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def require_evidence_for_answer(self) -> "AnswerContract":
        if self.decision is Decision.ANSWER:
            if not self.answer_id:
                raise ValueError("decision=answer 必須提供 answer_id")
            if not self.source_ids or not self.knowledge_versions:
                raise ValueError("decision=answer 必須提供來源與知識版本")
        return self


class TurnResponse(BaseModel):
    turn_id: str
    result: AnswerContract
