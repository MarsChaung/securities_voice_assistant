import base64
import io
import json
import re
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import perf_counter
from typing import Literal
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from observability import SafeAuditLogger

from .conversation import ConversationResolution, FollowUpKind, ReplyMode

VoiceAcknowledgementVariant = Literal[
    "context_confirmation",
    "follow_up_explanation",
    "knowledge_lookup",
]


def select_voice_acknowledgement_variant(
    conversation: ConversationResolution | None,
    *,
    conversation_pending: bool,
) -> VoiceAcknowledgementVariant:
    if conversation_pending or conversation is None:
        return "context_confirmation"
    if conversation.kind in {FollowUpKind.ELABORATE, FollowUpKind.REPHRASE}:
        return "follow_up_explanation"
    return "knowledge_lookup"


BARGE_IN_PRESETS: tuple[dict[str, str | int | float], ...] = (
    {
        "id": "sensitive",
        "label": "靈敏",
        "duck_after_ms": 80,
        "confirm_ms": 180,
        "energy_margin_db": 10,
        "minimum_dbfs": -50,
        "pre_roll_ms": 300,
        "false_trigger_timeout_ms": 300,
        "duck_volume": 0.15,
        "fade_out_ms": 50,
    },
    {
        "id": "standard",
        "label": "標準",
        "duck_after_ms": 100,
        "confirm_ms": 250,
        "energy_margin_db": 14,
        "minimum_dbfs": -45,
        "pre_roll_ms": 300,
        "false_trigger_timeout_ms": 400,
        "duck_volume": 0.15,
        "fade_out_ms": 50,
    },
    {
        "id": "resistant",
        "label": "抗干擾",
        "duck_after_ms": 150,
        "confirm_ms": 400,
        "energy_margin_db": 18,
        "minimum_dbfs": -38,
        "pre_roll_ms": 350,
        "false_trigger_timeout_ms": 500,
        "duck_volume": 0.12,
        "fade_out_ms": 50,
    },
)

VOICE_FAREWELL_MESSAGE = "謝謝您的來電，祝您順心，再見"
VOICE_IDLE_CHECK_IN_MESSAGE = "還有什麼事可以協助您的嗎?"
VOICE_IDLE_FAREWELL_MESSAGE = "很高興為你服務，歡迎您再度來電，再見。"
_CALL_ENDING_UTTERANCES = frozenset(
    {
        "沒問題了",
        "沒有問題了",
        "沒事了",
        "再見",
        "掰掰",
        "拜拜",
        "先這樣",
        "就這樣",
        "謝謝再見",
        "再見謝謝",
        "先這樣謝謝",
        "就這樣謝謝",
    }
)


def realtime_asr_url(audio_public_base_url: str) -> str:
    parsed = urlsplit(audio_public_base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path.rstrip("/") + "/audio/transcriptions/realtime"
    return urlunsplit((scheme, parsed.netloc, path, "", ""))


def is_call_ending_utterance(text: str) -> bool:
    compact = re.sub(r"[\s，,、。！？!?；;：:]+", "", text).casefold()
    if compact in _CALL_ENDING_UTTERANCES:
        return True
    filler = r"(?:呃+|嗯+|欸+|哎+|唉+|啊+|喔+|哦+|那個)*"
    return bool(
        re.fullmatch(
            rf"{filler}(?:好(?:了)?)?{filler}"
            rf"(?:沒(?:有)?問題了?|沒事(?:了|的)?|先這樣|就這樣)"
            rf"(?:謝謝(?:你|您)?|拜拜|掰掰|再見)*{filler}",
            compact,
        )
    )


def split_tts_text(
    text: str,
    *,
    max_chars: int = 80,
    hard_max_chars: int = 96,
) -> list[str]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if hard_max_chars < max_chars:
        raise ValueError("hard_max_chars must be greater than or equal to max_chars")

    remaining = text.strip()
    chunks: list[str] = []
    boundaries = "，。？、\n"
    while remaining:
        if len(remaining) <= hard_max_chars:
            chunks.append(remaining)
            break

        window = remaining[:hard_max_chars]
        candidates = [
            index + 1 for index, character in enumerate(window) if character in boundaries
        ]
        cut = (
            min(
                candidates,
                key=lambda position: (
                    abs(position - max_chars),
                    position > max_chars,
                ),
            )
            if candidates
            else hard_max_chars
        )
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    return [chunk for chunk in chunks if chunk]


def extract_wav_frames(buffer: bytes) -> tuple[list[bytes], bytes]:
    frames: list[bytes] = []
    while len(buffer) >= 12:
        if buffer[:4] != b"RIFF" or buffer[8:12] != b"WAVE":
            raise VoiceSynthesisError("invalid WAV stream")
        cursor = 12
        frame_size = None
        while len(buffer) >= cursor + 8:
            chunk_name = buffer[cursor : cursor + 4]
            chunk_size = int.from_bytes(buffer[cursor + 4 : cursor + 8], "little")
            chunk_end = cursor + 8 + chunk_size + (chunk_size % 2)
            if chunk_name == b"data":
                frame_size = chunk_end
                break
            cursor = chunk_end
        if frame_size is None or len(buffer) < frame_size:
            break
        frames.append(buffer[:frame_size])
        buffer = buffer[frame_size:]
    return frames, buffer


def ndjson_event(event: dict[str, object]) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode()


class VoiceReplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript: str = Field(min_length=1, max_length=4_000)
    conversation_id: UUID | None = None
    reply_mode: ReplyMode = ReplyMode.EXACT

    @model_validator(mode="after")
    def require_conversation_for_natural_mode(self) -> "VoiceReplyRequest":
        if self.reply_mode is ReplyMode.NATURAL and self.conversation_id is None:
            raise ValueError("自然對話模式必須提供 conversation_id")
        return self


class VoiceTestTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript: str = Field(min_length=1, max_length=4_000)
    session_id: UUID
    reply_mode: ReplyMode = ReplyMode.EXACT


class VoiceGreetingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    greeting: str = Field(min_length=1, max_length=300)


class VoiceIdlePromptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["check_in", "farewell"]


class VoicePlaybackChunkMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arrival_offset_ms: float = Field(ge=0, le=600_000)
    duration_ms: float = Field(gt=0, le=60_000)
    scheduled_start_offset_ms: float | None = Field(default=None, ge=0, le=600_000)
    gap_before_ms: float = Field(default=0, ge=0, le=60_000)


class VoicePlaybackMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_count: int = Field(ge=1, le=300)
    audio_duration_ms: float = Field(gt=0, le=600_000)
    initial_buffered_ms: float = Field(ge=0, le=600_000)
    first_playback_delay_ms: float | None = Field(default=None, ge=0, le=600_000)
    buffer_target_ms: float = Field(ge=0, le=10_000)
    crossfade_ms: float = Field(ge=0, le=100)
    underrun_count: int = Field(ge=0, le=300)
    underrun_total_ms: float = Field(ge=0, le=600_000)
    underrun_max_ms: float = Field(ge=0, le=60_000)
    interrupted: bool
    interruption_reason: Literal["manual", "barge_in"] | None = None
    barge_in_mode: Literal["sensitive", "standard", "resistant"] | None = None
    barge_in_duck_latency_ms: float | None = Field(default=None, ge=0, le=10_000)
    barge_in_confirm_latency_ms: float | None = Field(default=None, ge=0, le=10_000)
    barge_in_false_trigger_count: int = Field(default=0, ge=0, le=300)
    chunk_timings: list[VoicePlaybackChunkMetric] = Field(max_length=300)


class VoiceSynthesisError(RuntimeError):
    """TTS failure whose message must not contain answer or upstream response text."""

    def __init__(self, message: str, *, error_type: str = "invalid_audio") -> None:
        super().__init__(message)
        self.error_type = error_type


@dataclass(frozen=True)
class VoiceModels:
    asr: str
    tts: str
    voice: str


class VoiceService:
    def __init__(
        self,
        *,
        audio_base_url: str,
        audio_public_base_url: str,
        asr_model: str,
        tts_model: str,
        tts_voice: str,
        tts_ref_audio: str | None = None,
        tts_ref_text: str | None = None,
        timeout_seconds: float = 180.0,
        audit_logger: SafeAuditLogger | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.audio_public_base_url = audio_public_base_url.rstrip("/")
        self.models = VoiceModels(asr=asr_model, tts=tts_model, voice=tts_voice)
        if bool(tts_ref_audio) != bool(tts_ref_text):
            raise ValueError("voice clone requires both reference audio and text")
        self._tts_ref_audio = tts_ref_audio
        self._tts_ref_text = tts_ref_text
        self._audit_logger = audit_logger or SafeAuditLogger()
        self._client = client or httpx.AsyncClient(
            base_url=audio_base_url.rstrip("/") + "/",
            timeout=httpx.Timeout(timeout_seconds, connect=5.0),
        )

    @property
    def voice_clone_enabled(self) -> bool:
        return bool(self._tts_ref_audio and self._tts_ref_text)

    async def close(self) -> None:
        await self._client.aclose()

    async def available(self) -> bool:
        try:
            response = await self._client.get("../")
            return response.is_success
        except httpx.HTTPError:
            return False

    async def stream_answer(self, *, turn_id: str, answer: str) -> AsyncIterator[bytes]:
        started_at = perf_counter()
        first_audio_latency_ms: float | None = None
        audio_chunk_count = 0
        sentences = split_tts_text(answer)
        try:
            for sentence_index, sentence in enumerate(sentences, start=1):
                sentence_chunk_index = 0
                async for audio_frame in self._stream_sentence(
                    sentence,
                    turn_id=turn_id,
                    sentence_index=sentence_index,
                ):
                    sentence_chunk_index += 1
                    audio_chunk_count += 1
                    if first_audio_latency_ms is None:
                        first_audio_latency_ms = (perf_counter() - started_at) * 1_000
                    yield ndjson_event(
                        {
                            "type": "audio",
                            "audio": base64.b64encode(audio_frame).decode("ascii"),
                            "audio_content_type": "audio/wav",
                            "chunk_index": audio_chunk_count,
                            "sentence_index": sentence_index,
                            "sentence_chunk_index": sentence_chunk_index,
                        }
                    )
                if sentence_chunk_index == 0:
                    raise VoiceSynthesisError("empty TTS stream")
        except VoiceSynthesisError as error:
            total_latency_ms = (perf_counter() - started_at) * 1_000
            self._audit_logger.voice_synthesis(
                turn_id=turn_id,
                tts_model_id=self.models.tts,
                sentence_count=len(sentences),
                audio_chunk_count=audio_chunk_count,
                first_audio_latency_ms=first_audio_latency_ms,
                total_latency_ms=total_latency_ms,
                error_type="tts_unavailable",
            )
            yield ndjson_event(
                {
                    "type": "error",
                    "error_type": error.error_type,
                    "detail": "語音合成暫時無法使用，畫面仍保留文字答案。",
                }
            )
            return

        total_latency_ms = (perf_counter() - started_at) * 1_000
        self._audit_logger.voice_synthesis(
            turn_id=turn_id,
            tts_model_id=self.models.tts,
            sentence_count=len(sentences),
            audio_chunk_count=audio_chunk_count,
            first_audio_latency_ms=first_audio_latency_ms,
            total_latency_ms=total_latency_ms,
            error_type=None,
        )
        yield ndjson_event(
            {
                "type": "done",
                "turn_id": turn_id,
                "sentence_count": len(sentences),
                "audio_chunk_count": audio_chunk_count,
                "first_audio_latency_ms": first_audio_latency_ms,
                "total_latency_ms": total_latency_ms,
            }
        )

    async def _stream_sentence(
        self,
        text: str,
        *,
        turn_id: str,
        sentence_index: int,
    ) -> AsyncIterator[bytes]:
        payload = {
            "model": self.models.tts,
            "input": text,
            "voice": self.models.voice,
            "lang_code": "Chinese",
            "instruct": "使用自然、親切的台灣國語口音說話。",
            "response_format": "wav",
            "stream": True,
            "streaming_interval": 0.5,
            "turn_id": turn_id,
            "sentence_index": sentence_index,
        }
        if self.voice_clone_enabled:
            payload.update(
                {
                    "ref_audio": self._tts_ref_audio,
                    "ref_text": self._tts_ref_text,
                    "temperature": 0.2,
                    "top_k": 1,
                    "top_p": 1.0,
                    "repetition_penalty": 1.5,
                }
            )
        buffer = b""
        try:
            async with self._client.stream("POST", "audio/speech", json=payload) as response:
                if not response.is_success:
                    await response.aread()
                    raise VoiceSynthesisError(
                        "TTS upstream rejected request",
                        error_type="upstream_rejected",
                    )
                async for chunk in response.aiter_bytes():
                    buffer += chunk
                    frames, buffer = extract_wav_frames(buffer)
                    for frame in frames:
                        self._validate_wav(frame)
                        yield frame
        except httpx.HTTPError as error:
            raise VoiceSynthesisError(
                "TTS upstream unavailable",
                error_type="upstream_unavailable",
            ) from error
        if buffer:
            raise VoiceSynthesisError("incomplete WAV stream")

    @staticmethod
    def _validate_wav(frame: bytes) -> None:
        try:
            with wave.open(io.BytesIO(frame), "rb") as source:
                if source.getnframes() <= 0:
                    raise VoiceSynthesisError("empty WAV frame")
        except (EOFError, wave.Error) as error:
            raise VoiceSynthesisError("invalid WAV frame") from error
