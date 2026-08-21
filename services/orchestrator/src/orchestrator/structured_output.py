import json
from typing import Any, Literal

StructuredOutputMode = Literal["auto", "json_schema", "tool_call"]


def resolve_structured_output_mode(
    *,
    mode: StructuredOutputMode,
    model: str,
) -> Literal["json_schema", "tool_call"]:
    if mode == "auto":
        return "tool_call" if "gpt-oss" in model.lower() else "json_schema"
    return mode


def structured_output_options(
    *,
    name: str,
    schema: dict[str, Any],
    mode: StructuredOutputMode,
    model: str,
) -> dict[str, Any]:
    resolved_mode = resolve_structured_output_mode(mode=mode, model=model)
    if resolved_mode == "tool_call":
        return {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": "回傳符合指定 schema 的結構化結果。",
                        "strict": True,
                        "parameters": schema,
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": name},
            },
        }
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "strict": True,
                "schema": schema,
            },
        }
    }


def structured_output_content(
    payload: object,
    *,
    name: str,
    mode: StructuredOutputMode,
    model: str,
) -> str:
    if not isinstance(payload, dict):
        raise ValueError("invalid chat completion")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("missing choice")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("missing message")
    message = choice["message"]

    resolved_mode = resolve_structured_output_mode(mode=mode, model=model)
    if resolved_mode == "json_schema":
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError("missing structured content")
        return content

    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        raise ValueError("missing tool call")
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict) or function.get("name") != name:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments:
            return arguments
        if isinstance(arguments, dict):
            return json.dumps(arguments, ensure_ascii=False)
    raise ValueError("missing expected tool call")
