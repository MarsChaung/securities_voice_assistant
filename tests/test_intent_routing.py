import json

import httpx
import pytest

from orchestrator.intent_routing import (
    IntentRoutingError,
    OpenAICompatibleIntentRouter,
)


def test_openai_compatible_intent_router_uses_structured_prompt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://llm.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer synthetic-secret"
        body = json.loads(request.read())
        assert body["model"] == "synthetic-model"
        assert body["temperature"] == 0
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert json.loads(body["messages"][1]["content"]) == {
            "question": "如何開複委託帳戶？"
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "candidate_intents": ["account_opening_general"],
                                    "confidence": 0.94,
                                    "risk_flags": [],
                                    "needs_clarification": False,
                                }
                            )
                        }
                    }
                ]
            },
        )

    router = OpenAICompatibleIntentRouter(
        base_url="http://llm.test/v1",
        model="synthetic-model",
        api_key="synthetic-secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = router.route("如何開複委託帳戶？")

    assert result.classification.candidate_intents == ["account_opening_general"]
    assert result.classification.confidence == 0.94
    assert result.model_id == "synthetic-model"
    assert result.prompt_version == "intent-router-v3"
    assert len(result.prompt_hash) == 64
    assert result.latency_ms >= 0


def test_intent_router_rejects_unknown_schema_without_leaking_remote_content() -> None:
    remote_content = "secret malformed response"
    router = OpenAICompatibleIntentRouter(
        base_url="http://llm.test/v1",
        model="synthetic-model",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "candidate_intents": ["unsupported_intent"],
                                            "confidence": 1,
                                            "risk_flags": [],
                                            "needs_clarification": False,
                                            "remote": remote_content,
                                        }
                                    )
                                }
                            }
                        ]
                    },
                )
            )
        ),
    )

    with pytest.raises(IntentRoutingError) as error:
        router.route("synthetic question")

    assert remote_content not in str(error.value)
