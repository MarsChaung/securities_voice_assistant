import argparse
import io
import json
import math
import sys
import wave
from array import array
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from statistics import mean
from time import perf_counter

import httpx
from sqlalchemy import create_engine, text

from orchestrator.config import Settings
from orchestrator.voice import extract_wav_frames, split_tts_text

LEGACY_BOUNDARIES = "。！？；!?;\n，、,：:"


@dataclass(frozen=True)
class TTSBenchmarkCase:
    knowledge_id: str
    answer: str
    length_bucket: str


@dataclass(frozen=True)
class TTSBenchmarkResult:
    model_id: str
    segmenter: str
    knowledge_id: str
    length_bucket: str
    input_character_count: int
    repetition: int
    sentence_count: int
    audio_chunk_count: int
    first_audio_latency_ms: float
    total_latency_ms: float
    audio_duration_ms: float
    real_time_factor: float
    waveform_peak_abs: float
    waveform_clipped_sample_ratio: float
    waveform_boundary_jump_max: float


def legacy_split_tts_text(text_value: str, *, max_chars: int = 42) -> list[str]:
    remaining = text_value.strip()
    chunks: list[str] = []
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        window = remaining[:max_chars]
        cut = (
            max(
                (window.rfind(mark) for mark in LEGACY_BOUNDARIES),
                default=-1,
            )
            + 1
        )
        if cut <= 0:
            cut = max_chars
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    return [chunk for chunk in chunks if chunk]


def _length_bucket(character_count: int) -> str:
    if character_count <= 80:
        return "short"
    if character_count <= 160:
        return "medium"
    return "long"


def load_benchmark_cases(
    database_url: str,
    *,
    sample_per_bucket: int,
) -> list[TTSBenchmarkCase]:
    engine = create_engine(database_url)
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT knowledge_id, standard_answer
                FROM knowledge_items
                WHERE status = 'published'
                  AND public_answer_allowed = true
                ORDER BY knowledge_id
                """
            )
        ).mappings()
        cases = [
            TTSBenchmarkCase(
                knowledge_id=str(row["knowledge_id"]),
                answer=str(row["standard_answer"]),
                length_bucket=_length_bucket(len(str(row["standard_answer"]))),
            )
            for row in rows
        ]
    engine.dispose()

    targets = {"short": 60, "medium": 120, "long": 220}
    selected: list[TTSBenchmarkCase] = []
    for bucket, target in targets.items():
        candidates = sorted(
            (case for case in cases if case.length_bucket == bucket),
            key=lambda case: (
                abs(len(case.answer) - target),
                case.knowledge_id,
            ),
        )
        selected.extend(candidates[:sample_per_bucket])
    return selected


def _wav_samples(frame: bytes) -> tuple[array[int], int]:
    with wave.open(io.BytesIO(frame), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError("benchmark supports only 16-bit PCM WAV")
        if source.getnchannels() != 1:
            raise ValueError("benchmark supports only mono WAV")
        samples = array("h")
        samples.frombytes(source.readframes(source.getnframes()))
        if sys.byteorder != "little":
            samples.byteswap()
        return samples, source.getframerate()


def benchmark_answer(
    client: httpx.Client,
    *,
    model_id: str,
    segmenter_name: str,
    segmenter: Callable[[str], list[str]],
    benchmark_case: TTSBenchmarkCase,
    repetition: int,
    voice: str,
    ref_audio: str,
    ref_text: str,
) -> TTSBenchmarkResult:
    started_at = perf_counter()
    first_audio_latency_ms: float | None = None
    audio_chunk_count = 0
    audio_duration_seconds = 0.0
    peak_abs = 0.0
    clipped_samples = 0
    total_samples = 0
    boundary_jump_max = 0.0
    previous_last_sample: int | None = None
    sentences = segmenter(benchmark_case.answer)

    for sentence_index, sentence in enumerate(sentences, start=1):
        payload = {
            "model": model_id,
            "input": sentence,
            "voice": voice,
            "lang_code": "Chinese",
            "instruct": "使用自然、親切的台灣國語口音說話。",
            "response_format": "wav",
            "stream": True,
            "streaming_interval": 0.5,
            "turn_id": (
                f"tts-benchmark-{benchmark_case.knowledge_id}-{segmenter_name}-{repetition}"
            ),
            "sentence_index": sentence_index,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "temperature": 0.2,
            "top_k": 1,
            "top_p": 1.0,
            "repetition_penalty": 1.5,
        }
        buffer = b""
        with client.stream("POST", "audio/speech", json=payload) as response:
            if not response.is_success:
                response.read()
                raise RuntimeError(f"TTS request failed with status {response.status_code}")
            for raw_chunk in response.iter_bytes():
                buffer += raw_chunk
                frames, buffer = extract_wav_frames(buffer)
                for frame in frames:
                    if first_audio_latency_ms is None:
                        first_audio_latency_ms = (perf_counter() - started_at) * 1_000
                    samples, sample_rate = _wav_samples(frame)
                    if samples:
                        audio_chunk_count += 1
                        audio_duration_seconds += len(samples) / sample_rate
                        frame_peak = max(abs(sample) for sample in samples) / 32_768
                        peak_abs = max(peak_abs, frame_peak)
                        clipped_samples += sum(1 for sample in samples if abs(sample) >= 32_767)
                        total_samples += len(samples)
                        if previous_last_sample is not None:
                            boundary_jump_max = max(
                                boundary_jump_max,
                                abs(samples[0] - previous_last_sample) / 32_768,
                            )
                        previous_last_sample = samples[-1]
            if buffer:
                raise RuntimeError("TTS returned an incomplete WAV stream")

    total_latency_ms = (perf_counter() - started_at) * 1_000
    if first_audio_latency_ms is None or audio_duration_seconds <= 0:
        raise RuntimeError("TTS returned no audio")
    return TTSBenchmarkResult(
        model_id=model_id,
        segmenter=segmenter_name,
        knowledge_id=benchmark_case.knowledge_id,
        length_bucket=benchmark_case.length_bucket,
        input_character_count=len(benchmark_case.answer),
        repetition=repetition,
        sentence_count=len(sentences),
        audio_chunk_count=audio_chunk_count,
        first_audio_latency_ms=round(first_audio_latency_ms, 3),
        total_latency_ms=round(total_latency_ms, 3),
        audio_duration_ms=round(audio_duration_seconds * 1_000, 3),
        real_time_factor=round(total_latency_ms / (audio_duration_seconds * 1_000), 4),
        waveform_peak_abs=round(peak_abs, 6),
        waveform_clipped_sample_ratio=round(
            clipped_samples / total_samples if total_samples else 0,
            8,
        ),
        waveform_boundary_jump_max=round(boundary_jump_max, 6),
    )


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return round(ordered[index], 3)


def summarize_results(
    results: Sequence[TTSBenchmarkResult],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[TTSBenchmarkResult]] = {}
    for result in results:
        groups.setdefault((result.model_id, result.segmenter), []).append(result)

    summaries: list[dict[str, object]] = []
    for (model_id, segmenter), group in groups.items():
        total_latency_ms = sum(result.total_latency_ms for result in group)
        total_audio_ms = sum(result.audio_duration_ms for result in group)
        summaries.append(
            {
                "model_id": model_id,
                "segmenter": segmenter,
                "case_count": len(group),
                "sentence_count": sum(result.sentence_count for result in group),
                "audio_chunk_count": sum(result.audio_chunk_count for result in group),
                "first_audio_average_ms": round(
                    mean(result.first_audio_latency_ms for result in group),
                    3,
                ),
                "first_audio_p95_ms": _percentile(
                    [result.first_audio_latency_ms for result in group],
                    0.95,
                ),
                "total_latency_ms": round(total_latency_ms, 3),
                "audio_duration_ms": round(total_audio_ms, 3),
                "real_time_factor": round(total_latency_ms / total_audio_ms, 4),
                "waveform_peak_abs_max": max(result.waveform_peak_abs for result in group),
                "waveform_clipped_sample_ratio_max": max(
                    result.waveform_clipped_sample_ratio for result in group
                ),
                "waveform_boundary_jump_max": max(
                    result.waveform_boundary_jump_max for result in group
                ),
            }
        )
    return summaries


def _segmenters(names: Sequence[str]) -> dict[str, Callable[[str], list[str]]]:
    available: dict[str, Callable[[str], list[str]]] = {
        "legacy": legacy_split_tts_text,
        "selected_punctuation": split_tts_text,
    }
    return {name: available[name] for name in names}


def _warm_model(
    client: httpx.Client,
    *,
    model_id: str,
    benchmark_case: TTSBenchmarkCase,
    voice: str,
    ref_audio: str,
    ref_text: str,
) -> None:
    benchmark_answer(
        client,
        model_id=model_id,
        segmenter_name="warmup",
        segmenter=split_tts_text,
        benchmark_case=benchmark_case,
        repetition=0,
        voice=voice,
        ref_audio=ref_audio,
        ref_text=ref_text,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare local Qwen3-TTS models without logging answer or audio content."
    )
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument(
        "--segmenter",
        action="append",
        choices=["legacy", "selected_punctuation"],
        dest="segmenters",
    )
    parser.add_argument("--sample-per-bucket", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=1_200)
    parser.add_argument("--skip-warmup", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.sample_per_bucket < 1 or arguments.repetitions < 1:
        print(json.dumps({"status": "error", "error_type": "invalid_arguments"}))
        return 2

    settings = Settings()
    models = arguments.models or [settings.tts_model]
    if not all(models) or not settings.tts_ref_audio or not settings.tts_ref_text:
        print(json.dumps({"status": "error", "error_type": "missing_tts_settings"}))
        return 2

    segmenters = _segmenters(arguments.segmenters or ["legacy", "selected_punctuation"])
    cases = load_benchmark_cases(
        settings.database_url,
        sample_per_bucket=arguments.sample_per_bucket,
    )
    if not cases:
        print(json.dumps({"status": "error", "error_type": "no_benchmark_cases"}))
        return 2

    results: list[TTSBenchmarkResult] = []
    failures: list[dict[str, object]] = []
    ref_text = settings.tts_ref_text.get_secret_value()
    with httpx.Client(
        base_url=str(settings.tts_base_url).rstrip("/") + "/",
        timeout=httpx.Timeout(arguments.timeout_seconds, connect=5),
    ) as client:
        for model_id in models:
            if not arguments.skip_warmup:
                try:
                    _warm_model(
                        client,
                        model_id=model_id,
                        benchmark_case=cases[0],
                        voice=settings.tts_voice,
                        ref_audio=settings.tts_ref_audio,
                        ref_text=ref_text,
                    )
                except (httpx.HTTPError, RuntimeError, ValueError):
                    failures.append(
                        {
                            "model_id": model_id,
                            "stage": "warmup",
                            "error_type": "tts_unavailable",
                        }
                    )
                    continue

            for benchmark_case in cases:
                for segmenter_name, segmenter in segmenters.items():
                    for repetition in range(1, arguments.repetitions + 1):
                        try:
                            results.append(
                                benchmark_answer(
                                    client,
                                    model_id=model_id,
                                    segmenter_name=segmenter_name,
                                    segmenter=segmenter,
                                    benchmark_case=benchmark_case,
                                    repetition=repetition,
                                    voice=settings.tts_voice,
                                    ref_audio=settings.tts_ref_audio,
                                    ref_text=ref_text,
                                )
                            )
                        except (httpx.HTTPError, RuntimeError, ValueError):
                            failures.append(
                                {
                                    "model_id": model_id,
                                    "segmenter": segmenter_name,
                                    "knowledge_id": benchmark_case.knowledge_id,
                                    "length_bucket": benchmark_case.length_bucket,
                                    "repetition": repetition,
                                    "error_type": "tts_unavailable",
                                }
                            )

    payload = {
        "schema_version": "1.0",
        "status": "passed" if results and not failures else "incomplete",
        "case_ids": [case.knowledge_id for case in cases],
        "results": [asdict(result) for result in results],
        "summaries": summarize_results(results),
        "failures": failures,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if results and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
