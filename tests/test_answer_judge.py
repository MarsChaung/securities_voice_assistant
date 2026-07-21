import json

import httpx
import pytest

from orchestrator.answer_judge import (
    GroundednessJudgeError,
    OpenAICompatibleGroundednessJudge,
)
from orchestrator.answering import AnswerEvidence


def evidence() -> AnswerEvidence:
    return AnswerEvidence(
        standard_answer="美股交易可依官方規則使用新臺幣或美元交割。",
        prohibited_extensions=("不得查詢個人帳戶餘額",),
        knowledge_id="K-CATHAY-US-002",
        knowledge_version="1.1",
        source_id="SRC-CATHAY-USSTOCK-001",
    )


def test_groundedness_judge_uses_strict_schema_without_requesting_claim_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        assert body["response_format"]["type"] == "json_schema"
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert "unsupported_claims" not in schema["properties"]
        payload = json.loads(body["messages"][1]["content"])
        assert payload["approved_standard_answer"] == evidence().standard_answer
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "verdict": "grounded",
                                    "preserves_required_qualifiers": True,
                                    "unsupported_claim_count": 0,
                                    "prohibited_extension_detected": False,
                                    "reason_code": "fully_grounded",
                                }
                            )
                        }
                    }
                ]
            },
        )

    judge = OpenAICompatibleGroundednessJudge(
        base_url="http://llm.test/v1",
        model="synthetic-judge",
        api_key="synthetic-secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = judge.assess(
        evidence=evidence(),
        generated_answer="簡單說，美股可用新臺幣或美元交割。",
    )

    assert result.assessment.verdict == "grounded"
    assert result.assessment.preserves_required_qualifiers is True
    assert result.model_id == "synthetic-judge"
    assert len(result.prompt_hash) == 64


def test_groundedness_judge_hides_invalid_remote_content() -> None:
    secret_remote_content = "secret generated answer"
    judge = OpenAICompatibleGroundednessJudge(
        base_url="http://llm.test/v1",
        model="synthetic-judge",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"message": {"content": secret_remote_content}}
                        ]
                    },
                )
            )
        ),
    )

    with pytest.raises(GroundednessJudgeError) as error:
        judge.assess(evidence=evidence(), generated_answer="synthetic answer")

    assert secret_remote_content not in str(error.value)
