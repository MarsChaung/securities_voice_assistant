import asyncio
import io
import json
import wave
from array import array
from datetime import UTC, datetime

import httpx
from fastapi.testclient import TestClient
from pydantic import HttpUrl, SecretStr

from orchestrator.api import create_app
from orchestrator.config import Settings
from orchestrator.service import TurnService
from orchestrator.system_diagnostics import (
    DiagnosticCheck,
    DiagnosticReport,
    DiagnosticStatus,
    SystemDiagnosticRunner,
)
from orchestrator.voice import VoiceService
from test_api import StaticKnowledgeRepository, published_document


def make_wav_frame() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24_000)
        target.writeframes(array("h", [0, 1000, -1000, 0]).tobytes())
    return output.getvalue()


def make_voice_service() -> VoiceService:
    frame = make_wav_frame()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url == "http://audio.test/":
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "POST" and request.url == "http://audio.test/v1/audio/speech":
            return httpx.Response(200, content=frame)
        return httpx.Response(404)

    return VoiceService(
        audio_base_url="http://audio.test/v1",
        audio_public_base_url="http://127.0.0.1:8000/v1",
        asr_model="Qwen3-ASR-1.7B-8bit",
        tts_model="Qwen3-TTS-0.6B-Base-8bit",
        tts_voice="company-voice",
        client=httpx.AsyncClient(
            base_url="http://audio.test/v1/",
            transport=httpx.MockTransport(handler),
        ),
    )


def passing_settings() -> Settings:
    return Settings(
        app_env="development",
        system_diagnostics_enabled=True,
        system_diagnostics_timeout_seconds=5,
        database_url="sqlite+pysqlite:///:memory:",
        knowledge_admin_url=HttpUrl("http://knowledge-browser.test/admin/knowledge"),
        knowledge_admin_internal_url=HttpUrl("http://knowledge.test/admin/knowledge"),
        llm_base_url=HttpUrl("http://llm.test/v1"),
        answer_mode="exact",
        answer_llm_model="gpt-oss:20b",
        natural_answer_enabled=True,
        intent_router_mode="disabled",
        conversation_semantic_mode="disabled",
        retrieval_mode="lexical",
        voice_enabled=True,
        tts_base_url=HttpUrl("http://audio.test/v1"),
        audio_public_base_url=HttpUrl("http://127.0.0.1:8000/v1"),
        asr_model="Qwen3-ASR-1.7B-8bit",
        tts_model="Qwen3-TTS-0.6B-Base-8bit",
    )


def test_system_diagnostics_exercises_runtime_llm_tts_and_builds_browser_probe() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url == "http://knowledge.test/healthz":
            return httpx.Response(200, json={"status": "ok", "database": "connected"})
        if request.url == "http://llm.test/v1/chat/completions":
            request_payload = json.loads(request.content)
            assert request_payload["model"] == "gpt-oss:20b"
            assert request_payload["max_tokens"] == 768
            assert "response_format" not in request_payload
            assert request_payload["tool_choice"]["function"]["name"] == "system_diagnostic"
            assert "必須呼叫 system_diagnostic 工具" in request_payload["messages"][0]["content"]
            assert request_payload["messages"][1]["content"] == "請執行連線診斷。"
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "function": {
                                            "name": "system_diagnostic",
                                            "arguments": '{"diagnostic_status":"ok"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        return httpx.Response(404)

    async def run() -> DiagnosticReport:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        voice = make_voice_service()
        runner = SystemDiagnosticRunner(
            settings=passing_settings(),
            service=TurnService(
                knowledge_repository=StaticKnowledgeRepository((published_document(),)),
                clock=lambda: datetime(2026, 8, 20, tzinfo=UTC),
            ),
            voice_service=voice,
            client=client,
        )
        try:
            return await runner.run(
                page_origin="http://127.0.0.1:8080",
                secure_context=True,
            )
        finally:
            await runner.close()
            await client.aclose()
            await voice.close()

    report = asyncio.run(run())
    checks = {check.check_id: check for check in report.checks}

    assert report.overall_status is DiagnosticStatus.PASS
    assert checks["configuration"].status is DiagnosticStatus.PASS
    assert checks["runtime_database"].status is DiagnosticStatus.PASS
    assert checks["knowledge_admin"].status is DiagnosticStatus.PASS
    assert checks["llm"].status is DiagnosticStatus.PASS
    assert checks["embeddings"].status is DiagnosticStatus.SKIPPED
    assert checks["tts"].status is DiagnosticStatus.PASS
    assert checks["asr_configuration"].status is DiagnosticStatus.PASS
    assert report.browser_asr_probe is not None
    assert report.browser_asr_probe.url == ("ws://127.0.0.1:8000/v1/audio/transcriptions/realtime")
    assert report.browser_asr_probe.init_message["semantic_endpointing"] is True


def test_system_diagnostics_reports_tts_transport_failure_precisely() -> None:
    async def audio_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("synthetic connection failure", request=request)

    async def run() -> DiagnosticCheck:
        voice = VoiceService(
            audio_base_url="http://audio.test/v1",
            audio_public_base_url="http://audio.test/v1",
            asr_model="synthetic-asr",
            tts_model="synthetic-tts",
            tts_voice="synthetic-voice",
            client=httpx.AsyncClient(transport=httpx.MockTransport(audio_handler)),
        )
        runner = SystemDiagnosticRunner(
            settings=Settings(
                system_diagnostics_enabled=True,
                voice_enabled=True,
                asr_model="synthetic-asr",
                tts_model="synthetic-tts",
                tts_base_url=HttpUrl("http://audio.test/v1"),
                audio_public_base_url=HttpUrl("http://audio.test/v1"),
            ),
            service=TurnService(),
            voice_service=voice,
        )
        try:
            return await runner._check_tts()
        finally:
            await runner.close()
            await voice.close()

    check = asyncio.run(run())

    assert check.status is DiagnosticStatus.FAIL
    assert check.summary == "orchestrator 無法連線至 TTS API。"
    assert any("SVA_TTS_BASE_URL" in item for item in check.remediation)


def test_system_diagnostics_reports_actionable_failures_without_secrets() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async def run() -> DiagnosticReport:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        settings = Settings(
            system_diagnostics_enabled=True,
            llm_api_key=SecretStr("must-not-appear"),
            natural_answer_enabled=False,
            voice_enabled=False,
            audio_public_base_url=HttpUrl("http://127.0.0.1:8000/v1"),
        )
        runner = SystemDiagnosticRunner(
            settings=settings,
            service=TurnService(),
            voice_service=None,
            client=client,
        )
        try:
            return await runner.run(
                page_origin="https://voice.company.example",
                secure_context=False,
            )
        finally:
            await runner.close()
            await client.aclose()

    report = asyncio.run(run())
    serialized = report.model_dump_json()
    checks = {check.check_id: check for check in report.checks}

    assert report.overall_status is DiagnosticStatus.FAIL
    assert checks["configuration"].status is DiagnosticStatus.FAIL
    assert "loopback" in checks["configuration"].summary
    assert any("SVA_KNOWLEDGE_ADMIN_URL" in item for item in checks["configuration"].remediation)
    assert checks["runtime_database"].status is DiagnosticStatus.FAIL
    assert checks["tts"].status is DiagnosticStatus.FAIL
    assert checks["asr_configuration"].status is DiagnosticStatus.FAIL
    assert "must-not-appear" not in serialized


class StaticDiagnosticRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, bool | None]] = []
        self.closed = False

    async def run(
        self,
        *,
        page_origin: str | None = None,
        secure_context: bool | None = None,
    ) -> DiagnosticReport:
        self.calls.append((page_origin, secure_context))
        return DiagnosticReport(
            generated_at=datetime(2026, 8, 20, tzinfo=UTC),
            overall_status=DiagnosticStatus.PASS,
            checks=[
                DiagnosticCheck(
                    check_id="synthetic",
                    category="測試",
                    title="合成診斷",
                    status=DiagnosticStatus.PASS,
                    summary="通過",
                    duration_ms=1,
                )
            ],
        )

    async def close(self) -> None:
        self.closed = True


def test_system_diagnostics_page_and_api_are_explicitly_enabled() -> None:
    runner = StaticDiagnosticRunner()
    settings = Settings(
        system_diagnostics_enabled=True,
        answer_mode="exact",
        retrieval_mode="lexical",
        voice_enabled=False,
        knowledge_admin_url=HttpUrl("https://knowledge-browser.test/admin/knowledge"),
        knowledge_admin_internal_url=HttpUrl("http://knowledge-admin:8081/admin/knowledge"),
    )

    with TestClient(
        create_app(
            service=TurnService(),
            settings=settings,
            system_diagnostic_runner=runner,
        )
    ) as client:
        page = client.get("/system-diagnostics")
        stylesheet = client.get("/pilot/static/system-diagnostics.css")
        script = client.get("/pilot/static/system-diagnostics.js")
        response = client.post(
            "/v1/system-diagnostics/run",
            json={
                "page_origin": "https://voice.company.example",
                "secure_context": True,
            },
        )

    assert page.status_code == 200
    assert "https://knowledge-browser.test/admin/knowledge" in page.text
    assert "http://knowledge-admin:8081" not in page.text
    assert "系統診斷" in page.text
    assert stylesheet.status_code == 200
    assert script.status_code == 200
    assert "new WebSocket(probe.url)" in script.text
    assert response.status_code == 200
    assert response.json()["checks"][0]["check_id"] == "synthetic"
    assert runner.calls == [("https://voice.company.example", True)]
    assert runner.closed is True


def test_system_diagnostics_is_hidden_when_disabled() -> None:
    settings = Settings(system_diagnostics_enabled=False)
    client = TestClient(create_app(service=TurnService(), settings=settings))

    assert client.get("/system-diagnostics").status_code == 404
    assert client.post("/v1/system-diagnostics/run", json={}).status_code == 404
