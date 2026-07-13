from fastapi.testclient import TestClient

from orchestrator.api import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_sensitive_value_is_never_echoed() -> None:
    secret = "A123456789"
    response = client.post(
        "/v1/turns/evaluate",
        json={"transcript": f"我的身分證是 {secret}", "channel": "web"},
    )

    assert response.status_code == 200
    assert response.json()["result"]["policy_rule_id"] == "PII-001"
    assert secret not in response.text


def test_allowed_intent_without_approved_knowledge_is_refused() -> None:
    response = client.post(
        "/v1/turns/evaluate",
        json={"transcript": "APP 要如何下載？", "channel": "web"},
    )

    result = response.json()["result"]
    assert result["decision"] == "refuse"
    assert result["intent"] == "app_public_help"
    assert result["policy_rule_id"] == "KNO-001"


def test_transaction_request_is_refused_before_knowledge_lookup() -> None:
    response = client.post(
        "/v1/turns/evaluate",
        json={"transcript": "請幫我買進台積電", "channel": "web"},
    )

    result = response.json()["result"]
    assert result["decision"] == "refuse"
    assert result["intent"] == "transaction_request"
    assert result["policy_rule_id"] == "POL-REFUSE-001"
