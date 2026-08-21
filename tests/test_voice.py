import asyncio
import io
import json
import threading
import time
import wave
from array import array
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from observability import SafeAuditLogger
from orchestrator.api import create_app
from orchestrator.config import Settings
from orchestrator.conversation import (
    ConversationContextStore,
    ConversationExchange,
    ConversationResolution,
    FollowUpKind,
    FollowUpResolver,
)
from orchestrator.diagnostics import VoiceTestDiagnosticLogger
from orchestrator.intent_routing import IntentRouteResult
from orchestrator.service import TurnService
from orchestrator.voice import (
    VOICE_FAREWELL_MESSAGE,
    VOICE_IDLE_CHECK_IN_MESSAGE,
    VOICE_IDLE_FAREWELL_MESSAGE,
    VoiceService,
    VoiceSynthesisError,
    extract_wav_frames,
    is_call_ending_utterance,
    realtime_asr_url,
    select_voice_acknowledgement_variant,
    split_tts_text,
)
from retrieval import KnowledgeDocument, QuestionVariant, QuestionVariantUsage
from test_api import (
    StaticIntentRouter,
    StaticKnowledgeRepository,
    StaticNaturalAnswerComposer,
    published_document,
)


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


class CapturingVoiceTestDiagnosticLogger(VoiceTestDiagnosticLogger):
    def __init__(self) -> None:
        super().__init__(enabled=True, app_env="development")
        self.events: list[dict[str, object]] = []

    def exchange(self, **event: object) -> None:
        self.events.append(event)

    def acknowledgement(self, **event: object) -> None:
        self.events.append({"event_type": "acknowledgement", **event})


def voice_service(
    transport: httpx.AsyncBaseTransport,
    *,
    audit_logger: SafeAuditLogger | None = None,
    voice_clone: bool = False,
    asr_model: str = "synthetic-asr",
    tts_convert_traditional_to_simplified: bool = False,
) -> VoiceService:
    return VoiceService(
        audio_base_url="http://audio.test/v1",
        audio_public_base_url="http://127.0.0.1:8000/v1",
        asr_model=asr_model,
        tts_model="synthetic-tts",
        tts_voice="Vivian",
        tts_ref_audio="/private/reference.wav" if voice_clone else None,
        tts_ref_text="合成參考逐字稿" if voice_clone else None,
        tts_convert_traditional_to_simplified=tts_convert_traditional_to_simplified,
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

    partial_frame = frame[:20]
    assert extract_wav_frames(partial_frame) == ([], partial_frame)
    with pytest.raises(VoiceSynthesisError, match="invalid WAV stream"):
        extract_wav_frames(b"not-a-wave!!!")


def test_voice_service_requires_complete_clone_configuration() -> None:
    with pytest.raises(ValueError, match="reference audio and text"):
        VoiceService(
            audio_base_url="http://audio.test/v1",
            audio_public_base_url="http://127.0.0.1:8000/v1",
            asr_model="synthetic-asr",
            tts_model="synthetic-tts",
            tts_voice="Vivian",
            tts_ref_audio="/private/reference.wav",
        )


def test_voice_service_availability_hides_transport_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("synthetic unavailable", request=request)

    service = voice_service(httpx.MockTransport(handler))

    async def check() -> bool:
        available = await service.available()
        await service.close()
        return available

    assert asyncio.run(check()) is False


@pytest.mark.parametrize(
    "transcript",
    [
        "沒問題了",
        "沒事了。",
        "再見",
        "謝謝，再見",
        "先這樣，謝謝",
        "好，沒事的，拜拜。",
        "沒事了，拜拜。",
        "好了，呃，沒事的，謝謝。",
        "好，沒事的，謝謝。",
        "嗯，沒有問題了，謝謝您。",
    ],
)
def test_call_ending_utterance_recognises_explicit_closings(transcript: str) -> None:
    assert is_call_ending_utterance(transcript) is True


@pytest.mark.parametrize(
    "transcript",
    [
        "這個問題沒事了嗎",
        "再見之前我還想問開戶",
        "沒問題，請繼續",
        "好了，我還想問開戶",
        "謝謝，還有一個問題",
        "我想問手續費",
    ],
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
        assert payload["input"] == answer
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
    service = voice_service(httpx.MockTransport(handler), audit_logger=audit, voice_clone=True)
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


def test_voice_service_can_convert_only_tts_input_to_simplified_chinese() -> None:
    frame = make_wav_frame()
    answer = "這是繁體中文語音測試，請確認證券帳戶。"

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["input"] == "这是繁体中文语音测试，请确认证券帐户。"
        return httpx.Response(200, content=frame)

    service = voice_service(
        httpx.MockTransport(handler),
        tts_convert_traditional_to_simplified=True,
    )

    async def collect() -> list[dict[str, object]]:
        events = [
            json.loads(payload)
            async for payload in service.stream_answer(turn_id="turn-voice", answer=answer)
        ]
        await service.close()
        return events

    events = asyncio.run(collect())

    assert events[-1]["type"] == "done"
    assert split_tts_text(answer) == [answer]


@pytest.mark.parametrize(
    "failure_mode",
    ["rejected", "transport", "incomplete", "empty_stream", "empty_wav"],
)
def test_voice_service_reports_governed_error_for_tts_failures(
    failure_mode: str,
) -> None:
    empty_wav = io.BytesIO()
    with wave.open(empty_wav, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24_000)
        target.writeframes(b"")

    async def handler(request: httpx.Request) -> httpx.Response:
        if failure_mode == "rejected":
            return httpx.Response(503, text="private upstream response")
        if failure_mode == "transport":
            raise httpx.ConnectError("private transport detail", request=request)
        if failure_mode == "incomplete":
            return httpx.Response(200, content=b"RIFF")
        if failure_mode == "empty_wav":
            return httpx.Response(200, content=empty_wav.getvalue())
        return httpx.Response(200, content=b"")

    audit = CapturingVoiceAuditLogger()
    service = voice_service(httpx.MockTransport(handler), audit_logger=audit)

    async def collect() -> list[dict[str, object]]:
        events = [
            json.loads(payload)
            async for payload in service.stream_answer(
                turn_id=f"turn-{failure_mode}",
                answer="核准回答",
            )
        ]
        await service.close()
        return events

    events = asyncio.run(collect())

    assert events[0]["type"] == "error"
    assert events[0]["detail"] == "語音合成暫時無法使用，畫面仍保留文字答案。"
    expected_error_type = {
        "rejected": "upstream_rejected",
        "transport": "upstream_unavailable",
        "incomplete": "invalid_audio",
        "empty_stream": "invalid_audio",
        "empty_wav": "invalid_audio",
    }[failure_mode]
    assert events[0]["error_type"] == expected_error_type
    assert audit.events[0]["error_type"] == "tts_unavailable"
    assert audit.events[0]["audio_chunk_count"] == 0


def test_voice_endpoint_runs_policy_pipeline_before_streaming_tts() -> None:
    frame = make_wav_frame()
    tts_inputs: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"message": "ok"})
        tts_inputs.append(json.loads(request.content)["input"])
        return httpx.Response(200, content=frame)

    voice = voice_service(
        httpx.MockTransport(handler),
        voice_clone=True,
        asr_model="mlx-community/Qwen3-ASR-1.7B-8bit",
        tts_convert_traditional_to_simplified=True,
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
        tts_convert_traditional_to_simplified=True,
        voice_test_content_logging_enabled=False,
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
    assert config.json()["acknowledgement"] == {
        "delay_ms": 450,
        "audio_urls": {
            "context_confirmation": (
                "/pilot/static/audio/acknowledgement-confirm.mp3?v=20260817.1"
            ),
            "context_wait": ("/pilot/static/audio/voice-acknowledgement.wav?v=20260817.1"),
            "follow_up_explanation": (
                "/pilot/static/audio/acknowledgement-explain.mp3?v=20260817.1"
            ),
            "knowledge_lookup": ("/pilot/static/audio/acknowledgement-lookup.mp3?v=20260817.1"),
        },
    }
    assert config.json()["reply_modes"] == [{"id": "exact", "label": "核准原文"}]
    assert config.json()["diagnostic_content_logging_enabled"] is False
    assert [preset["id"] for preset in config.json()["barge_in"]["presets"]] == [
        "sensitive",
        "standard",
        "resistant",
    ]
    assert "reference.wav" not in config.text
    assert "合成參考逐字稿" not in config.text
    assert events[0]["type"] == "turn"
    assert events[0]["speech_segments"] == [turn["result"]["answer"]]
    assert "沒有" in events[0]["speech_segments"][0]
    assert tts_inputs == ["目前没有足够且有效的已发布知识来源可回答，请改用官方客服管道。"]
    assert turn["result"]["decision"] == "refuse"
    assert turn["result"]["policy_rule_id"] == "KNO-001"
    assert events[1]["type"] == "audio"
    assert events[-1]["type"] == "done"


def test_voice_endpoints_report_disabled_service() -> None:
    settings = Settings(
        app_env="development",
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        answer_mode="exact",
        intent_router_mode="disabled",
        voice_enabled=False,
    )

    with TestClient(create_app(service=TurnService(), settings=settings)) as client:
        config = client.get("/v1/voice/config")
        responses = [
            client.post(
                "/v1/voice/respond-stream",
                json={"transcript": "如何開戶？"},
            ),
            client.post(
                "/v1/voice/test-greeting-stream",
                json={"greeting": "您好"},
            ),
            client.post(
                "/v1/voice/idle-prompt-stream",
                json={"stage": "check_in"},
            ),
        ]

    assert config.json()["enabled"] is False
    assert config.json()["available"] is False
    assert [response.status_code for response in responses] == [503, 503, 503]


def test_voice_natural_mode_keeps_recent_context_and_can_be_cleared() -> None:
    frame = make_wav_frame()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=frame)

    document = published_document()
    composer = StaticNaturalAnswerComposer(answer="我用比較口語的方式說明這項核准內容。")
    service = TurnService(
        knowledge_repository=StaticKnowledgeRepository((document,)),
        natural_answer_composer=composer,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    store = ConversationContextStore()
    conversation_id = uuid4()
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
        create_app(
            service=service,
            settings=settings,
            voice_service=voice,
            conversation_store=store,
        )
    ) as client:
        config = client.get("/v1/voice/config")
        first = client.post(
            "/v1/voice/respond-stream",
            json={
                "transcript": "什麼是台股定期定額？",
                "reply_mode": "natural",
                "conversation_id": str(conversation_id),
            },
        )
        second = client.post(
            "/v1/voice/respond-stream",
            json={
                "transcript": "剛才那一段可以再說詳細一點嗎？",
                "reply_mode": "natural",
                "conversation_id": str(conversation_id),
            },
        )
        cleared = client.delete(f"/v1/voice/conversations/{conversation_id}")

    assert config.json()["reply_modes"] == [
        {"id": "exact", "label": "核准原文"},
        {"id": "natural", "label": "自然對話"},
    ]
    assert first.status_code == 200
    assert second.status_code == 200
    assert len(composer.calls) == 2
    assert composer.calls[1]["follow_up_kind"] is FollowUpKind.ELABORATE
    history = composer.calls[1]["history"]
    assert isinstance(history, tuple)
    assert len(history) == 1
    assert cleared.status_code == 204
    assert store.history(str(conversation_id)) == ()


def test_voice_natural_mode_resolves_context_and_intent_in_parallel() -> None:
    frame = make_wav_frame()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=frame)

    barrier = threading.Barrier(2)

    class BarrierIntentRouter(StaticIntentRouter):
        def route(self, question: str) -> IntentRouteResult:
            barrier.wait(timeout=2)
            return super().route(question)

    class BarrierFollowUpResolver(FollowUpResolver):
        def resolve(
            self,
            *,
            utterance: str,
            history: Sequence[ConversationExchange],
        ) -> ConversationResolution:
            barrier.wait(timeout=2)
            return ConversationResolution(
                kind=FollowUpKind.NEW_QUESTION,
                retrieval_query=utterance,
                history=tuple(history),
            )

    document = published_document()
    router = BarrierIntentRouter(candidate_intents=["general_securities_knowledge"])
    service = TurnService(
        knowledge_repository=StaticKnowledgeRepository((document,)),
        natural_answer_composer=StaticNaturalAnswerComposer(),
        intent_router_mode="controlled",
        intent_router=router,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        voice_enabled=True,
        asr_model="synthetic-asr",
        tts_model="synthetic-tts",
    )

    with TestClient(
        create_app(
            service=service,
            settings=settings,
            voice_service=voice_service(httpx.MockTransport(handler)),
            follow_up_resolver=BarrierFollowUpResolver(),
        )
    ) as client:
        response = client.post(
            "/v1/voice/respond-stream",
            json={
                "transcript": "什麼是台股定期定額？",
                "reply_mode": "natural",
                "conversation_id": str(uuid4()),
            },
        )

    assert response.status_code == 200
    assert router.questions == ["什麼是台股定期定額？"]


@pytest.mark.parametrize(
    ("conversation", "pending", "expected_variant"),
    [
        (None, True, "context_confirmation"),
        (
            ConversationResolution(
                kind=FollowUpKind.NEW_QUESTION,
                retrieval_query="新問題",
                history=(),
            ),
            False,
            "knowledge_lookup",
        ),
        (
            ConversationResolution(
                kind=FollowUpKind.ELABORATE,
                retrieval_query="深入說明",
                history=(),
            ),
            False,
            "follow_up_explanation",
        ),
        (
            ConversationResolution(
                kind=FollowUpKind.REPHRASE,
                retrieval_query="換句話說",
                history=(),
            ),
            False,
            "follow_up_explanation",
        ),
    ],
)
def test_voice_acknowledgement_variant_matches_conversation_context(
    conversation: ConversationResolution | None,
    pending: bool,
    expected_variant: str,
) -> None:
    assert (
        select_voice_acknowledgement_variant(
            conversation,
            conversation_pending=pending,
        )
        == expected_variant
    )


def test_slow_natural_voice_answer_emits_acknowledgement_before_turn() -> None:
    frame = make_wav_frame()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=frame)

    class SlowFollowUpResolver(FollowUpResolver):
        def resolve(
            self,
            *,
            utterance: str,
            history: Sequence[ConversationExchange],
        ) -> ConversationResolution:
            time.sleep(0.2)
            return ConversationResolution(
                kind=FollowUpKind.NEW_QUESTION,
                retrieval_query=utterance,
                history=tuple(history),
            )

    document = published_document()
    service = TurnService(
        knowledge_repository=StaticKnowledgeRepository((document,)),
        natural_answer_composer=StaticNaturalAnswerComposer(),
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        voice_enabled=True,
        asr_model="synthetic-asr",
        tts_model="synthetic-tts",
        voice_acknowledgement_delay_ms=100,
    )
    diagnostic_logger = CapturingVoiceTestDiagnosticLogger()

    with TestClient(
        create_app(
            service=service,
            settings=settings,
            voice_service=voice_service(httpx.MockTransport(handler)),
            follow_up_resolver=SlowFollowUpResolver(),
            diagnostic_logger=diagnostic_logger,
        )
    ) as client:
        response = client.post(
            "/v1/voice/respond-stream",
            json={
                "transcript": "什麼是台股定期定額？",
                "reply_mode": "natural",
                "conversation_id": str(uuid4()),
            },
        )

    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0] == {
        "type": "acknowledgement",
        "variant": "context_confirmation",
        "audio_url": ("/pilot/static/audio/acknowledgement-confirm.mp3?v=20260817.1"),
        "audio_urls": [
            "/pilot/static/audio/acknowledgement-confirm.mp3?v=20260817.1",
            "/pilot/static/audio/voice-acknowledgement.wav?v=20260817.1",
        ],
    }
    assert events[1]["type"] == "turn"
    assert events[2]["type"] == "audio"
    acknowledgement_event = next(
        event for event in diagnostic_logger.events if event.get("event_type") == "acknowledgement"
    )
    assert acknowledgement_event["variant"] == "context_confirmation"
    assert acknowledgement_event["triggered_after_ms"] == 100
    answer_ready_after_ms = acknowledgement_event["answer_ready_after_ms"]
    assert isinstance(answer_ready_after_ms, float)
    assert answer_ready_after_ms >= 100


@pytest.mark.parametrize(
    ("kind", "expected_variant", "expected_filename"),
    [
        (FollowUpKind.NEW_QUESTION, "knowledge_lookup", "acknowledgement-lookup.mp3"),
        (
            FollowUpKind.ELABORATE,
            "follow_up_explanation",
            "acknowledgement-explain.mp3",
        ),
        (
            FollowUpKind.REPHRASE,
            "follow_up_explanation",
            "acknowledgement-explain.mp3",
        ),
    ],
)
def test_slow_answer_uses_completed_conversation_for_acknowledgement(
    kind: FollowUpKind,
    expected_variant: str,
    expected_filename: str,
) -> None:
    frame = make_wav_frame()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=frame)

    class ImmediateFollowUpResolver(FollowUpResolver):
        def resolve(
            self,
            *,
            utterance: str,
            history: Sequence[ConversationExchange],
        ) -> ConversationResolution:
            return ConversationResolution(
                kind=kind,
                retrieval_query=utterance,
                history=tuple(history),
                reference_knowledge_id=(
                    document.item.knowledge_id if kind is not FollowUpKind.NEW_QUESTION else None
                ),
            )

    class SlowKnowledgeRepository(StaticKnowledgeRepository):
        def eligible_documents(self, *, at: datetime) -> tuple[KnowledgeDocument, ...]:
            time.sleep(0.2)
            return super().eligible_documents(at=at)

    document = published_document()
    service = TurnService(
        knowledge_repository=SlowKnowledgeRepository((document,)),
        natural_answer_composer=StaticNaturalAnswerComposer(),
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        voice_enabled=True,
        asr_model="synthetic-asr",
        tts_model="synthetic-tts",
        voice_acknowledgement_delay_ms=100,
    )

    with TestClient(
        create_app(
            service=service,
            settings=settings,
            voice_service=voice_service(httpx.MockTransport(handler)),
            follow_up_resolver=ImmediateFollowUpResolver(),
        )
    ) as client:
        response = client.post(
            "/v1/voice/respond-stream",
            json={
                "transcript": "什麼是台股定期定額？",
                "reply_mode": "natural",
                "conversation_id": str(uuid4()),
            },
        )

    acknowledgement = json.loads(response.text.splitlines()[0])
    assert acknowledgement["type"] == "acknowledgement"
    assert acknowledgement["variant"] == expected_variant
    assert acknowledgement["audio_url"] == (f"/pilot/static/audio/{expected_filename}?v=20260817.1")
    assert "audio_urls" not in acknowledgement


def test_fast_natural_voice_answer_skips_acknowledgement() -> None:
    frame = make_wav_frame()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=frame)

    document = published_document()
    service = TurnService(
        knowledge_repository=StaticKnowledgeRepository((document,)),
        natural_answer_composer=StaticNaturalAnswerComposer(),
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        voice_enabled=True,
        asr_model="synthetic-asr",
        tts_model="synthetic-tts",
        voice_acknowledgement_delay_ms=500,
    )

    with TestClient(
        create_app(
            service=service,
            settings=settings,
            voice_service=voice_service(httpx.MockTransport(handler)),
        )
    ) as client:
        response = client.post(
            "/v1/voice/respond-stream",
            json={
                "transcript": "什麼是台股定期定額？",
                "reply_mode": "natural",
                "conversation_id": str(uuid4()),
            },
        )

    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["type"] == "turn"
    assert all(event["type"] != "acknowledgement" for event in events)


@pytest.mark.parametrize(
    ("transcript", "expected_policy_rule_id"),
    [
        ("我的驗證碼是 123456", "PII-001"),
        ("幫我下單買進台積電", "POL-REFUSE-001"),
        ("我要申訴", "POL-HANDOFF-001"),
    ],
)
def test_blocked_voice_input_never_emits_acknowledgement(
    transcript: str,
    expected_policy_rule_id: str,
) -> None:
    frame = make_wav_frame()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=frame)

    class SlowFollowUpResolver(FollowUpResolver):
        def resolve(
            self,
            *,
            utterance: str,
            history: Sequence[ConversationExchange],
        ) -> ConversationResolution:
            time.sleep(0.2)
            return ConversationResolution(
                kind=FollowUpKind.NEW_QUESTION,
                retrieval_query=utterance,
                history=tuple(history),
            )

    service = TurnService(
        natural_answer_composer=StaticNaturalAnswerComposer(),
    )
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        voice_enabled=True,
        asr_model="synthetic-asr",
        tts_model="synthetic-tts",
        voice_acknowledgement_delay_ms=100,
    )

    with TestClient(
        create_app(
            service=service,
            settings=settings,
            voice_service=voice_service(httpx.MockTransport(handler)),
            follow_up_resolver=SlowFollowUpResolver(),
        )
    ) as client:
        response = client.post(
            "/v1/voice/respond-stream",
            json={
                "transcript": transcript,
                "reply_mode": "natural",
                "conversation_id": str(uuid4()),
            },
        )

    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["type"] == "turn"
    assert events[0]["turn"]["result"]["policy_rule_id"] == expected_policy_rule_id
    assert all(event["type"] != "acknowledgement" for event in events)


def test_voice_test_text_turn_uses_session_context_for_elliptical_follow_up() -> None:
    base = published_document()
    document = KnowledgeDocument(
        item=base.item.model_copy(
            update={
                "knowledge_id": "K-TEST-MINOR-001",
                "title": "未成年人開戶",
                "standard_answer": (
                    "未成年人開戶須由父母或法定代理人陪同，並攜帶核准清單所列證件。"
                ),
                "allowed_intents": ["faq_general_guidance"],
                "question_variants": [
                    QuestionVariant(
                        variant_id="minor-account-opening",
                        question_text="未成年怎麼開戶",
                        usage=QuestionVariantUsage.RETRIEVAL,
                    )
                ],
            }
        ),
        source=base.source,
    )
    composer = StaticNaturalAnswerComposer()
    service = TurnService(
        knowledge_repository=StaticKnowledgeRepository((document,)),
        natural_answer_composer=composer,
        intent_router_mode="controlled",
        intent_router=StaticIntentRouter(candidate_intents=["account_opening_general"]),
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    store = ConversationContextStore()
    session_id = uuid4()
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        app_env="development",
    )

    with TestClient(
        create_app(
            service=service,
            settings=settings,
            conversation_store=store,
        )
    ) as client:
        first = client.post(
            "/v1/voice/test-turns/evaluate",
            json={
                "transcript": "未成年怎麼開戶",
                "reply_mode": "natural",
                "session_id": str(session_id),
            },
        )
        follow_up = client.post(
            "/v1/voice/test-turns/evaluate",
            json={
                "transcript": "父母要帶什麼證件",
                "reply_mode": "natural",
                "session_id": str(session_id),
            },
        )

    assert first.status_code == 200
    assert first.json()["result"]["decision"] == "answer", first.text
    assert follow_up.status_code == 200
    assert follow_up.json()["result"]["decision"] == "answer", follow_up.text
    assert len(composer.calls) == 2
    assert composer.calls[1]["follow_up_kind"] is FollowUpKind.ELABORATE
    history = composer.calls[1]["history"]
    assert isinstance(history, tuple)
    assert len(history) == 1


def test_voice_test_text_turn_is_not_available_outside_development() -> None:
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        app_env="production",
    )

    with TestClient(create_app(service=TurnService(), settings=settings)) as client:
        response = client.post(
            "/v1/voice/test-turns/evaluate",
            json={
                "transcript": "未成年怎麼開戶",
                "session_id": str(uuid4()),
            },
        )

    assert response.status_code == 404


def test_voice_test_text_turn_logs_session_and_dialogue_when_enabled() -> None:
    session_id = uuid4()
    diagnostic_logger = CapturingVoiceTestDiagnosticLogger()
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        app_env="development",
    )

    with TestClient(
        create_app(
            service=TurnService(),
            settings=settings,
            diagnostic_logger=diagnostic_logger,
        )
    ) as client:
        response = client.post(
            "/v1/voice/test-turns/evaluate",
            json={
                "transcript": "測試用問題",
                "session_id": str(session_id),
            },
        )

    event = diagnostic_logger.events[-1]
    assert response.status_code == 200
    assert event["session_id"] == str(session_id)
    assert event["user_utterance"] == "測試用問題"
    assert event["assistant_answer"] == response.json()["result"]["answer"]


def test_voice_natural_mode_requires_conversation_id() -> None:
    voice = voice_service(httpx.MockTransport(lambda request: httpx.Response(200)))
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        voice_enabled=True,
        asr_model="synthetic-asr",
        tts_model="synthetic-tts",
    )

    with TestClient(
        create_app(service=TurnService(), settings=settings, voice_service=voice)
    ) as client:
        response = client.post(
            "/v1/voice/respond-stream",
            json={"transcript": "請說明", "reply_mode": "natural"},
        )

    assert response.status_code == 422


def test_voice_natural_mode_cannot_be_called_when_not_enabled() -> None:
    voice = voice_service(httpx.MockTransport(lambda request: httpx.Response(200)))
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_mode="lexical",
        voice_enabled=True,
        asr_model="synthetic-asr",
        tts_model="synthetic-tts",
    )

    with TestClient(
        create_app(service=TurnService(), settings=settings, voice_service=voice)
    ) as client:
        response = client.post(
            "/v1/voice/respond-stream",
            json={
                "transcript": "請說明",
                "reply_mode": "natural",
                "conversation_id": str(uuid4()),
            },
        )

    assert response.status_code == 409


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


@pytest.mark.parametrize(
    ("stage", "message", "ends_call"),
    [
        ("check_in", VOICE_IDLE_CHECK_IN_MESSAGE, False),
        ("farewell", VOICE_IDLE_FAREWELL_MESSAGE, True),
    ],
)
def test_voice_idle_prompt_streams_only_governed_messages(
    stage: str,
    message: str,
    ends_call: bool,
) -> None:
    frame = make_wav_frame()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["input"] == message
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
            "/v1/voice/idle-prompt-stream",
            json={"stage": stage},
        )

    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0] == {
        "type": "idle_prompt",
        "stage": stage,
        "ends_call": ends_call,
        "speech_segments": [message],
    }
    assert events[1]["type"] == "audio"
    assert events[-1]["type"] == "done"


def test_voice_idle_prompt_rejects_unknown_stage() -> None:
    voice = voice_service(httpx.MockTransport(lambda request: httpx.Response(200)))
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
            "/v1/voice/idle-prompt-stream",
            json={"stage": "say_anything"},
        )

    assert response.status_code == 422


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

    with TestClient(create_app(service=service, settings=settings, voice_service=voice)) as client:
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
