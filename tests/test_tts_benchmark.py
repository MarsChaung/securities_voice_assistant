import io
import json
import wave
from array import array

import httpx
from evals.run_tts import (
    TTSBenchmarkCase,
    benchmark_answer,
    legacy_split_tts_text,
    summarize_results,
)

from orchestrator.voice import split_tts_text


def make_wav_frame() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24_000)
        target.writeframes(array("h", [0, 1000, -1000, 0]).tobytes())
    return output.getvalue()


def test_tts_benchmark_result_never_contains_answer_or_audio_content() -> None:
    frame = make_wav_frame()
    approved_answer = "這是不可寫入評測報表的核准答案內容。"

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["input"] == approved_answer
        return httpx.Response(200, content=frame)

    benchmark_case = TTSBenchmarkCase(
        knowledge_id="K-SYNTHETIC-TTS-001",
        answer=approved_answer,
        length_bucket="short",
    )
    with httpx.Client(
        base_url="http://audio.test/v1/",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = benchmark_answer(
            client,
            model_id="synthetic-model",
            segmenter_name="selected_punctuation",
            segmenter=split_tts_text,
            benchmark_case=benchmark_case,
            repetition=1,
            voice="Vivian",
            ref_audio="/private/reference.wav",
            ref_text="不可輸出的參考文字",
        )

    serialized = json.dumps(result.__dict__, ensure_ascii=False)
    assert result.knowledge_id == "K-SYNTHETIC-TTS-001"
    assert result.audio_chunk_count == 1
    assert result.waveform_clipped_sample_ratio == 0
    assert approved_answer not in serialized
    assert "不可輸出的參考文字" not in serialized
    assert "audio" not in result.__dict__


def test_tts_benchmark_compares_legacy_and_selected_punctuation_segments() -> None:
    text = ("甲" * 45) + "！" + ("乙" * 45) + "，" + ("丙" * 50)

    legacy = legacy_split_tts_text(text)
    selected = split_tts_text(text)

    assert len(legacy) == 6
    assert len(selected) == 2


def test_tts_benchmark_summary_uses_aggregate_real_time_factor() -> None:
    frame = make_wav_frame()

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=frame)

    results = []
    with httpx.Client(
        base_url="http://audio.test/v1/",
        transport=httpx.MockTransport(handler),
    ) as client:
        for index in range(2):
            results.append(
                benchmark_answer(
                    client,
                    model_id="synthetic-model",
                    segmenter_name="selected_punctuation",
                    segmenter=split_tts_text,
                    benchmark_case=TTSBenchmarkCase(
                        knowledge_id=f"K-SYNTHETIC-{index}",
                        answer="合成測試。",
                        length_bucket="short",
                    ),
                    repetition=1,
                    voice="Vivian",
                    ref_audio="/private/reference.wav",
                    ref_text="合成參考文字",
                )
            )

    summary = summarize_results(results)[0]
    assert summary["case_count"] == 2
    assert summary["sentence_count"] == 2
    real_time_factor = summary["real_time_factor"]
    assert isinstance(real_time_factor, float)
    assert real_time_factor > 0
