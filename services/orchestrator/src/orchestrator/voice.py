import base64
import io
import json
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import perf_counter
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

from observability import SafeAuditLogger


def realtime_asr_url(audio_public_base_url: str) -> str:
    parsed = urlsplit(audio_public_base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path.rstrip("/") + "/audio/transcriptions/realtime"
    return urlunsplit((scheme, parsed.netloc, path, "", ""))


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
            index + 1
            for index, character in enumerate(window)
            if character in boundaries
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
    chunk_timings: list[VoicePlaybackChunkMetric] = Field(max_length=300)


class VoiceSynthesisError(RuntimeError):
    """TTS failure whose message must not contain answer or upstream response text."""


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
        except VoiceSynthesisError:
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
            async with self._client.stream(
                "POST", "audio/speech", json=payload
            ) as response:
                if not response.is_success:
                    await response.aread()
                    raise VoiceSynthesisError("TTS upstream rejected request")
                async for chunk in response.aiter_bytes():
                    buffer += chunk
                    frames, buffer = extract_wav_frames(buffer)
                    for frame in frames:
                        self._validate_wav(frame)
                        yield frame
        except httpx.HTTPError as error:
            raise VoiceSynthesisError("TTS upstream unavailable") from error
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
