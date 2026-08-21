import json

import pytest

from orchestrator.structured_output import (
    resolve_structured_output_mode,
    structured_output_content,
    structured_output_options,
)

SCHEMA = {
    "type": "object",
    "properties": {"status": {"type": "string"}},
    "required": ["status"],
    "additionalProperties": False,
}


def test_auto_mode_uses_tool_calls_for_gpt_oss_only() -> None:
    assert (
        resolve_structured_output_mode(
            mode="auto",
            model="mlx-community/gpt-oss-20b-MXFP4-Q8",
        )
        == "tool_call"
    )
    assert resolve_structured_output_mode(mode="auto", model="Qwen3.6-35B-A3B-oQ4") == "json_schema"


def test_tool_call_mode_builds_forced_function_and_extracts_arguments() -> None:
    options = structured_output_options(
        name="system_diagnostic",
        schema=SCHEMA,
        mode="auto",
        model="gpt-oss:20b",
    )

    assert "response_format" not in options
    assert options["tools"][0]["function"]["parameters"] == SCHEMA
    assert options["tools"][0]["function"]["strict"] is True
    assert options["tool_choice"]["function"]["name"] == "system_diagnostic"

    content = structured_output_content(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "system_diagnostic",
                                    "arguments": '{"status":"ok"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
        name="system_diagnostic",
        mode="auto",
        model="gpt-oss:20b",
    )

    assert json.loads(content) == {"status": "ok"}


def test_json_schema_mode_preserves_existing_openai_request_and_response() -> None:
    options = structured_output_options(
        name="system_diagnostic",
        schema=SCHEMA,
        mode="auto",
        model="synthetic-model",
    )

    assert "tools" not in options
    assert options["response_format"]["type"] == "json_schema"
    assert options["response_format"]["json_schema"]["schema"] == SCHEMA
    assert (
        structured_output_content(
            {"choices": [{"message": {"content": '{"status":"ok"}'}}]},
            name="system_diagnostic",
            mode="auto",
            model="synthetic-model",
        )
        == '{"status":"ok"}'
    )


def test_tool_call_mode_rejects_missing_expected_function_without_leaking_content() -> None:
    with pytest.raises(ValueError, match="missing expected tool call"):
        structured_output_content(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "wrong_function",
                                        "arguments": '{"private":"value"}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            name="system_diagnostic",
            mode="tool_call",
            model="any-model",
        )
