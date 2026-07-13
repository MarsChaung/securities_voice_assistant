import pytest

from policy import DomainPolicyEngine, PolicyAction


@pytest.mark.parametrize(
    ("text", "action", "intent"),
    [
        ("APP 要如何下載？", PolicyAction.ALLOW, "app_public_help"),
        ("幫我下單買進台積電", PolicyAction.REFUSE, "transaction_request"),
        ("我想申訴這次服務", PolicyAction.HANDOFF, "complaint_or_dispute"),
        ("台積電明天怎麼樣", PolicyAction.REFUSE, "unknown_or_ambiguous"),
    ],
)
def test_policy_classification(text: str, action: PolicyAction, intent: str) -> None:
    result = DomainPolicyEngine().classify(text)

    assert result.action is action
    assert result.intent == intent


def test_prohibited_intent_wins_over_allowed_intent() -> None:
    result = DomainPolicyEngine().classify("APP 裡要怎麼下單？")

    assert result.action is PolicyAction.REFUSE
    assert result.intent == "transaction_request"
