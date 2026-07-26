import asyncio
import io
import json
import wave
from array import array
from datetime import UTC, datetime

import httpx
from fastapi.testclient import TestClient

from observability import SafeAuditLogger
from orchestrator.api import create_app
from orchestrator.config import Settings
from orchestrator.service import TurnService
from orchestrator.voice import (
    VoiceService,
    extract_wav_frames,
    realtime_asr_url,
    split_tts_text,
)
from retrieval import KnowledgeDocument, QuestionVariant, QuestionVariantUsage
from test_api import StaticKnowledgeRepository, published_document


def make_wav_frame() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24_000)
        target.writeframes(array("h", [0, 1000, -1000, 0]).tobytes())
    return output.getvalue()


class CapturingVoiceAuditLogger(SafeAuditLogger):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, object]] = []

    def voice_synthesis(
        self,
        *,
        turn_id: str,
        tts_model_id: str,
        sentence_count: int,
        audio_chunk_count: int,
        first_audio_latency_ms: float | None,
        total_latency_ms: float,
        error_type: str | None,
    ) -> None:
        self.events.append(
            {
                "turn_id": turn_id,
                "tts_model_id": tts_model_id,
                "sentence_count": sentence_count,
                "audio_chunk_count": audio_chunk_count,
                "first_audio_latency_ms": first_audio_latency_ms,
                "total_latency_ms": total_latency_ms,
                "error_type": error_type,
            }
        )


def voice_service(
    transport: httpx.AsyncBaseTransport,
    *,
    audit_logger: SafeAuditLogger | None = None,
    voice_clone: bool = False,
) -> VoiceService:
    return VoiceService(
        audio_base_url="http://audio.test/v1",
        audio_public_base_url="http://127.0.0.1:8000/v1",
        asr_model="synthetic-asr",
        tts_model="synthetic-tts",
        tts_voice="Vivian",
        tts_ref_audio="/private/reference.wav" if voice_clone else None,
        tts_ref_text="合成參考逐字稿" if voice_clone else None,
        audit_logger=audit_logger,
        client=httpx.AsyncClient(
            base_url="http://audio.test/v1/",
            transport=transport,
        ),
    )


def test_voice_helpers_preserve_realtime_url_and_complete_wav_frames() -> None:
    frame = make_wav_frame()

    frames, remainder = extract_wav_frames(frame + frame)

    assert realtime_asr_url("http://127.0.0.1:8000/v1") == (
        "ws://127.0.0.1:8000/v1/audio/transcriptions/realtime"
    )
    assert frames == [frame, frame]
    assert remainder == b""
    assert "".join(split_tts_text("第一句回答。第二句補充。", max_chars=7)) == (
        "第一句回答。第二句補充。"
    )


def test_voice_service_streams_audio_and_logs_no_answer_text() -> None:
    frame = make_wav_frame()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://audio.test/v1/audio/speech"
        payload = json.loads(request.content)
        assert payload["model"] == "synthetic-tts"
        assert payload["voice"] == "Vivian"
        assert payload["stream"] is True
        assert payload["ref_audio"] == "/private/reference.wav"
        assert payload["ref_text"] == "合成參考逐字稿"
        assert payload["temperature"] == 0.2
        assert payload["top_k"] == 1
        assert payload["top_p"] == 1.0
        assert payload["repetition_penalty"] == 1.5
        return httpx.Response(200, content=frame)

    audit = CapturingVoiceAuditLogger()
    service = voice_service(
        httpx.MockTransport(handler), audit_logger=audit, voice_clone=True
    )
    answer = "這是只應送往瀏覽器與地端 TTS 的核准答案。"

    async def collect() -> list[dict[str, object]]:
        events = [
            json.loads(payload)
            async for payload in service.stream_answer(turn_id="turn-voice", answer=answer)
        ]
        await service.close()
        return events

    events = asyncio.run(collect())

    assert events[0]["type"] == "audio"
    assert events[-1]["type"] == "done"
    assert events[-1]["audio_chunk_count"] == 1
    assert len(audit.events) == 1
    assert audit.events[0]["turn_id"] == "turn-voice"
    assert audit.events[0]["audio_chunk_count"] == 1
    assert answer not in json.dumps(audit.events[0], ensure_ascii=False)


def test_voice_endpoint_runs_policy_pipeline_before_streaming_tts() -> None:
    frame = make_wav_frame()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"message": "ok"})
        return httpx.Response(200, content=frame)

    voice = voice_service(httpx.MockTransport(handler), voice_clone=True)
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        answer_mode="exact",
        intent_router_mode="disabled",
        voice_enabled=True,
        asr_model="synthetic-asr",
        tts_model="synthetic-tts",
    )

    with TestClient(
        create_app(service=TurnService(), settings=settings, voice_service=voice)
    ) as client:
        config = client.get("/v1/voice/config")
        response = client.post(
            "/v1/voice/respond-stream",
            json={"transcript": "Web 版要如何操作？"},
        )

    events = [json.loads(line) for line in response.text.splitlines()]
    turn = events[0]["turn"]
    assert config.json()["available"] is True
    assert config.json()["models"]["asr"] == "synthetic-asr"
    assert config.json()["models"]["voice_clone"] is True
    assert config.json()["asr_context"] == ""
    assert "reference.wav" not in config.text
    assert "合成參考逐字稿" not in config.text
    assert events[0]["type"] == "turn"
    assert turn["result"]["decision"] == "refuse"
    assert turn["result"]["policy_rule_id"] == "KNO-001"
    assert events[1]["type"] == "audio"
    assert events[-1]["type"] == "done"


def test_voice_endpoint_applies_phonetic_recovery_before_tts() -> None:
    frame = make_wav_frame()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=frame)

    base = published_document()
    document = KnowledgeDocument(
        item=base.item.model_copy(
            update={
                "knowledge_id": "K-FAQ-ASR-001",
                "title": "假除權息說明",
                "standard_answer": "這是假除權息的核准說明。",
                "allowed_intents": ["faq_general_guidance"],
                "question_variants": [
                    QuestionVariant(
                        variant_id="asr-voice-endpoint",
                        question_text="什麼是假除權息？",
                        usage=QuestionVariantUsage.RETRIEVAL,
                    )
                ],
            }
        ),
        source=base.source,
    )
    service = TurnService(
        knowledge_repository=StaticKnowledgeRepository((document,)),
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    voice = voice_service(httpx.MockTransport(handler))
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        answer_mode="exact",
        intent_router_mode="disabled",
        voice_enabled=True,
        asr_model="synthetic-asr",
        tts_model="synthetic-tts",
    )

    with TestClient(
        create_app(service=service, settings=settings, voice_service=voice)
    ) as client:
        response = client.post(
            "/v1/voice/respond-stream",
            json={"transcript": "什麼是甲雛全息"},
        )

    events = [json.loads(line) for line in response.text.splitlines()]
    result = events[0]["turn"]["result"]
    assert result["decision"] == "answer"
    assert result["policy_rule_id"] == "ASR-PHONETIC-001"
    assert result["answer_id"] == "K-FAQ-ASR-001"
    assert events[1]["type"] == "audio"
