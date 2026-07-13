from enum import StrEnum

from pydantic import BaseModel, Field


class PolicyAction(StrEnum):
    ALLOW = "allow"
    REFUSE = "refuse"
    HANDOFF = "handoff"


class PolicyResult(BaseModel):
    action: PolicyAction
    intent: str
    policy_rule_id: str
    confidence: float = Field(ge=0, le=1)


class GuardResult(BaseModel):
    has_sensitive_data: bool
    detected_types: list[str]
    redacted_text: str
