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
        assert body["max_tokens"] == 768
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert json.loads(body["messages"][1]["content"]) == {"question": "如何開複委託帳戶？"}
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
    assert result.prompt_version == "intent-router-v4"
    assert len(result.prompt_hash) == 64
    assert result.latency_ms >= 0


def test_intent_router_classifies_account_closure_as_supported_intent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        assert "account_closure_general" in body["messages"][0]["content"]
        assert json.loads(body["messages"][1]["content"]) == {
            "question": "註銷證券帳戶要怎麼辦理?"
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "candidate_intents": ["account_closure_general"],
                                    "confidence": 0.96,
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
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = router.route("註銷證券帳戶要怎麼辦理?")

    assert result.classification.candidate_intents == ["account_closure_general"]


def test_gpt_oss_follow_up_has_enough_output_budget_for_forced_tool_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        assert body["max_tokens"] == 768
        assert "response_format" not in body
        assert body["tool_choice"]["function"]["name"] == "intent_classification"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "intent_classification",
                                        "arguments": json.dumps(
                                            {
                                                "candidate_intents": [
                                                    "account_opening_general"
                                                ],
                                                "confidence": 0.9,
                                                "risk_flags": [],
                                                "needs_clarification": False,
                                            }
                                        ),
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
        )

    router = OpenAICompatibleIntentRouter(
        base_url="http://llm.test/v1",
        model="mlx-community/gpt-oss-20b-MXFP4-Q8",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = router.route("剛剛你說，父母要帶什麼證件?")

    assert result.classification.candidate_intents == ["account_opening_general"]


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
