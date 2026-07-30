import asyncio
import io
import json
import wave
from array import array
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from observability import SafeAuditLogger
from orchestrator.api import create_app
from orchestrator.config import Settings
from orchestrator.service import TurnService
from orchestrator.voice import (
    VOICE_FAREWELL_MESSAGE,
    VoiceService,
    extract_wav_frames,
    is_call_ending_utterance,
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


class CapturingPlaybackAuditLogger(SafeAuditLogger):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, object]] = []

    def voice_playback(self, **metrics: object) -> None:
        self.events.append(metrics)


def voice_service(
    transport: httpx.AsyncBaseTransport,
    *,
    audit_logger: SafeAuditLogger | None = None,
    voice_clone: bool = False,
    asr_model: str = "synthetic-asr",
) -> VoiceService:
    return VoiceService(
        audio_base_url="http://audio.test/v1",
        audio_public_base_url="http://127.0.0.1:8000/v1",
        asr_model=asr_model,
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
    assert split_tts_text("第一句回答。第二句補充。", max_chars=7, hard_max_chars=9) == [
        "第一句回答。",
        "第二句補充。",
    ]


@pytest.mark.parametrize(
    "transcript",
    ["沒問題了", "沒事了。", "再見", "謝謝，再見", "先這樣，謝謝"],
)
def test_call_ending_utterance_recognises_explicit_closings(transcript: str) -> None:
    assert is_call_ending_utterance(transcript) is True


@pytest.mark.parametrize(
    "transcript",
    ["這個問題沒事了嗎", "再見之前我還想問開戶", "沒問題，請繼續", "我想問手續費"],
)
def test_call_ending_utterance_does_not_hide_real_questions(transcript: str) -> None:
    assert is_call_ending_utterance(transcript) is False


def test_tts_segmenter_uses_only_selected_punctuation_near_target_length() -> None:
    text = ("甲" * 45) + "！" + ("乙" * 24) + "，" + ("丙" * 55) + "？"

    segments = split_tts_text(text)

    assert segments == [
        ("甲" * 45) + "！" + ("乙" * 24) + "，",
        ("丙" * 55) + "？",
    ]


def test_tts_segmenter_can_extend_to_96_characters_for_selected_boundary() -> None:
    text = ("甲" * 84) + "。" + ("乙" * 30)

    segments = split_tts_text(text)

    assert segments == [("甲" * 84) + "。", "乙" * 30]


def test_tts_segmenter_uses_enumeration_comma_and_newline_boundaries() -> None:
    text = ("甲" * 79) + "、" + ("乙" * 79) + "\n" + ("丙" * 30)

    segments = split_tts_text(text)

    assert segments == [
        ("甲" * 79) + "、",
        "乙" * 79,
        "丙" * 30,
    ]


def test_tts_segmenter_hard_splits_only_when_selected_boundary_is_unavailable() -> None:
    text = ("甲" * 50) + "；" + ("乙" * 60)

    segments = split_tts_text(text)

    assert segments == [text[:96], text[96:]]


def test_tts_segmenter_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        split_tts_text("測試", max_chars=0)
    with pytest.raises(ValueError, match="hard_max_chars"):
        split_tts_text("測試", max_chars=80, hard_max_chars=79)


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

    voice = voice_service(
        httpx.MockTransport(handler),
        voice_clone=True,
        asr_model="mlx-community/Qwen3-ASR-1.7B-8bit",
    )
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        answer_mode="exact",
        intent_router_mode="disabled",
        voice_enabled=True,
        asr_model="mlx-community/Qwen3-ASR-1.7B-8bit",
        asr_candidate_model="mlx-community/whisper-large-v3-turbo-asr-fp16",
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
    assert config.json()["models"]["asr"] == "mlx-community/Qwen3-ASR-1.7B-8bit"
    assert config.json()["asr_models"] == [
        "mlx-community/Qwen3-ASR-1.7B-8bit",
        "mlx-community/whisper-large-v3-turbo-asr-fp16",
    ]
    assert config.json()["models"]["voice_clone"] is True
    assert config.json()["asr_context"] == ""
    assert config.json()["barge_in"]["enabled"] is True
    assert config.json()["barge_in"]["default_mode"] == "standard"
    assert config.json()["asr_endpoint_grace_ms"] == 1200
    assert [preset["id"] for preset in config.json()["barge_in"]["presets"]] == [
        "sensitive",
        "standard",
        "resistant",
    ]
    assert "reference.wav" not in config.text
    assert "合成參考逐字稿" not in config.text
    assert events[0]["type"] == "turn"
    assert events[0]["speech_segments"] == [turn["result"]["answer"]]
    assert turn["result"]["decision"] == "refuse"
    assert turn["result"]["policy_rule_id"] == "KNO-001"
    assert events[1]["type"] == "audio"
    assert events[-1]["type"] == "done"


def test_voice_endpoint_streams_fixed_farewell_for_call_ending_utterance() -> None:
    frame = make_wav_frame()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["input"] == VOICE_FAREWELL_MESSAGE
        return httpx.Response(200, content=frame)

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
        create_app(service=TurnService(), settings=settings, voice_service=voice)
    ) as client:
        response = client.post(
            "/v1/voice/respond-stream",
            json={"transcript": "沒問題了"},
        )

    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0] == {
        "type": "farewell",
        "speech_segments": [VOICE_FAREWELL_MESSAGE],
    }
    assert events[1]["type"] == "audio"
    assert events[-1]["type"] == "done"


def test_development_voice_test_can_stream_an_unsaved_greeting() -> None:
    frame = make_wav_frame()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=frame)

    greeting = "您好，我是 AI 語音客服。請問今天想了解什麼呢？"
    voice = voice_service(httpx.MockTransport(handler))
    settings = Settings(
        app_env="development",
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
        response = client.post(
            "/v1/voice/test-greeting-stream",
            json={"greeting": greeting},
        )

    events = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200
    assert events[0] == {"type": "greeting", "speech_segments": [greeting]}
    assert events[1]["type"] == "audio"
    assert events[-1]["type"] == "done"


def test_voice_test_greeting_is_not_available_outside_development() -> None:
    voice = voice_service(httpx.MockTransport(lambda _: httpx.Response(200)))
    settings = Settings(
        app_env="staging",
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
        response = client.post(
            "/v1/voice/test-greeting-stream",
            json={"greeting": "測試招呼語"},
        )

    assert response.status_code == 404


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


def test_voice_playback_metrics_endpoint_logs_metadata_and_rejects_content() -> None:
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        answer_mode="exact",
        intent_router_mode="disabled",
    )
    turn_id = uuid4()
    payload = {
        "chunk_count": 2,
        "audio_duration_ms": 1000,
        "initial_buffered_ms": 1000,
        "first_playback_delay_ms": 1220,
        "buffer_target_ms": 1200,
        "crossfade_ms": 8,
        "underrun_count": 0,
        "underrun_total_ms": 0,
        "underrun_max_ms": 0,
        "interrupted": False,
        "interruption_reason": None,
        "barge_in_mode": None,
        "barge_in_duck_latency_ms": None,
        "barge_in_confirm_latency_ms": None,
        "barge_in_false_trigger_count": 0,
        "chunk_timings": [
            {
                "arrival_offset_ms": 300,
                "duration_ms": 500,
                "scheduled_start_offset_ms": 1240,
                "gap_before_ms": 0,
            },
            {
                "arrival_offset_ms": 850,
                "duration_ms": 500,
                "scheduled_start_offset_ms": 1732,
                "gap_before_ms": 0,
            },
        ],
    }
    audit = CapturingPlaybackAuditLogger()

    with TestClient(
        create_app(
            service=TurnService(audit_logger=audit),
            settings=settings,
        )
    ) as client:
        response = client.post(
            f"/v1/voice/{turn_id}/playback-metrics",
            json=payload,
        )
        invalid = client.post(
            f"/v1/voice/{turn_id}/playback-metrics",
            json=payload | {"transcript": "不可記錄的問題內容"},
        )

    assert response.status_code == 204
    assert invalid.status_code == 422
    assert len(audit.events) == 1
    event = audit.events[0]
    assert event["turn_id"] == str(turn_id)
    assert event["chunk_count"] == 2
    assert "transcript" not in event
