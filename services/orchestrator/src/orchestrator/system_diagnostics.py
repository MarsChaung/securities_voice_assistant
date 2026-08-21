import asyncio
import json
import math
from datetime import UTC, datetime
from enum import StrEnum
from time import perf_counter
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .service import TurnService
from .structured_output import (
    resolve_structured_output_mode,
    structured_output_content,
    structured_output_options,
)
from .voice import VoiceService, realtime_asr_url


class DiagnosticStatus(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIPPED = "skipped"


class DiagnosticRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_origin: str | None = Field(default=None, max_length=500)
    secure_context: bool | None = None


class DiagnosticCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    category: str
    title: str
    status: DiagnosticStatus
    summary: str
    evidence: list[str] = Field(default_factory=list)
    remediation: list[str] = Field(default_factory=list)
    duration_ms: float = Field(ge=0)


class BrowserASRProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    model: str
    init_message: dict[str, object]
    timeout_ms: int = Field(ge=1_000, le=180_000)


class DiagnosticReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    overall_status: DiagnosticStatus
    checks: list[DiagnosticCheck]
    browser_asr_probe: BrowserASRProbe | None = None


class SystemDiagnosticRunnerProtocol(Protocol):
    async def run(
        self,
        *,
        page_origin: str | None = None,
        secure_context: bool | None = None,
    ) -> DiagnosticReport: ...

    async def close(self) -> None: ...


class SystemDiagnosticRunner:
    """Run synthetic, read-only checks without returning secrets or upstream bodies."""

    def __init__(
        self,
        *,
        settings: Settings,
        service: TurnService,
        voice_service: VoiceService | None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._service = service
        self._voice_service = voice_service
        self._timeout_seconds = settings.system_diagnostics_timeout_seconds
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                self._timeout_seconds,
                connect=min(5.0, self._timeout_seconds),
            )
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def run(
        self,
        *,
        page_origin: str | None = None,
        secure_context: bool | None = None,
    ) -> DiagnosticReport:
        checks = [
            self._check_configuration(
                page_origin=page_origin,
                secure_context=secure_context,
            ),
            await self._check_runtime_database(),
            await self._check_knowledge_admin(),
            await self._check_llm(),
            await self._check_embeddings(),
            await self._check_tts(),
        ]
        asr_check, browser_probe = self._check_asr_configuration()
        checks.append(asr_check)
        return DiagnosticReport(
            generated_at=datetime.now(UTC),
            overall_status=_overall_status(checks),
            checks=checks,
            browser_asr_probe=browser_probe,
        )

    def _check_configuration(
        self,
        *,
        page_origin: str | None,
        secure_context: bool | None,
    ) -> DiagnosticCheck:
        started_at = perf_counter()
        failures: list[str] = []
        warnings: list[str] = []
        remediation: list[str] = []
        page = urlsplit(page_origin or "")
        audio = urlsplit(str(self._settings.audio_public_base_url))
        knowledge_admin = urlsplit(str(self._settings.knowledge_admin_url))

        if not self._settings.voice_enabled:
            failures.append("SVA_VOICE_ENABLED 目前為 false")
            remediation.append("將 SVA_VOICE_ENABLED 設為 true，重新建立 orchestrator 容器。")
        if secure_context is False:
            failures.append("目前頁面不是瀏覽器安全內容（Secure Context）")
            remediation.append("從其他電腦測試麥克風時，請以有效的 HTTPS 網址開啟網站。")
        if page.scheme == "https" and audio.scheme == "http":
            failures.append("HTTPS 頁面設定成連線未加密的 ASR HTTP／WebSocket")
            remediation.append("將 SVA_AUDIO_PUBLIC_BASE_URL 改成瀏覽器可達的 HTTPS URL。")
        if (
            page.hostname
            and not _is_loopback_host(page.hostname)
            and _is_loopback_host(audio.hostname)
        ):
            failures.append("遠端瀏覽器的 ASR Public URL 指向 loopback 位址")
            remediation.append(
                "SVA_AUDIO_PUBLIC_BASE_URL 不可使用 127.0.0.1 或 localhost；請填公司 DNS 名稱。"
            )
        if (
            page.hostname
            and not _is_loopback_host(page.hostname)
            and _is_loopback_host(knowledge_admin.hostname)
        ):
            failures.append("遠端瀏覽器的 knowledge-admin Public URL 指向 loopback 位址")
            remediation.append(
                "SVA_KNOWLEDGE_ADMIN_URL 不可使用 127.0.0.1 或 localhost；"
                "請填使用者瀏覽器可達的公司 DNS 名稱。"
            )
        if self._settings.answer_mode == "fixed_message":
            warnings.append("回答模式為 fixed_message，LLM 與自然對話不會套用")
        if not self._settings.natural_answer_enabled:
            warnings.append("自然對話回答未啟用，客服回答模式只會顯示核准原文")

        status = (
            DiagnosticStatus.FAIL
            if failures
            else DiagnosticStatus.WARNING
            if warnings
            else DiagnosticStatus.PASS
        )
        issues = [*failures, *warnings]
        return _check(
            check_id="configuration",
            category="設定",
            title="部署與功能設定",
            status=status,
            summary="；".join(issues) if issues else "語音與瀏覽器部署設定未發現明顯衝突。",
            evidence=[
                f"環境：{self._settings.app_env}",
                f"回答模式：{self._settings.answer_mode}",
                f"自然回答：{'啟用' if self._settings.natural_answer_enabled else '停用'}",
                f"語音功能：{'啟用' if self._settings.voice_enabled else '停用'}",
                f"ASR Public URL：{_safe_url(str(self._settings.audio_public_base_url))}",
                f"診斷頁來源：{page_origin or '未提供'}",
            ],
            remediation=remediation,
            started_at=started_at,
        )

    async def _check_runtime_database(self) -> DiagnosticCheck:
        started_at = perf_counter()
        availability = await asyncio.to_thread(self._service.knowledge_availability)
        if availability.status == "unavailable":
            return _check(
                check_id="runtime_database",
                category="資料庫",
                title="Runtime 與 PostgreSQL",
                status=DiagnosticStatus.FAIL,
                summary="orchestrator 無法查詢知識資料庫。",
                evidence=["資料庫狀態：unavailable"],
                remediation=[
                    "確認 SVA_DATABASE_URL 的主機、port、資料庫、帳密與 URL encoding。",
                    "從 orchestrator 容器測試 PostgreSQL DNS 與 TCP 連線，"
                    "並檢查 migration 是否完成。",
                ],
                started_at=started_at,
            )
        if availability.status != "connected":
            return _check(
                check_id="runtime_database",
                category="資料庫",
                title="Runtime 與 PostgreSQL",
                status=DiagnosticStatus.FAIL,
                summary="Runtime 沒有可用的知識 Repository。",
                evidence=[f"資料庫狀態：{availability.status}"],
                remediation=[
                    "確認 orchestrator 使用正式 create_app 流程且已設定 SVA_DATABASE_URL。"
                ],
                started_at=started_at,
            )
        if availability.eligible_document_count == 0:
            return _check(
                check_id="runtime_database",
                category="資料庫",
                title="Runtime 與 PostgreSQL",
                status=DiagnosticStatus.WARNING,
                summary="資料庫可連線，但目前沒有符合條件的已發布知識。",
                evidence=["可回答知識：0 筆"],
                remediation=[
                    "確認知識狀態為 published、public_answer_allowed=true，且仍在生效與複審期限內。"
                ],
                started_at=started_at,
            )
        return _check(
            check_id="runtime_database",
            category="資料庫",
            title="Runtime 與 PostgreSQL",
            status=DiagnosticStatus.PASS,
            summary="Runtime 可查詢 PostgreSQL，且有可回答的已發布知識。",
            evidence=[f"可回答知識：{availability.eligible_document_count} 筆"],
            remediation=[],
            started_at=started_at,
        )

    async def _check_knowledge_admin(self) -> DiagnosticCheck:
        started_at = perf_counter()
        health_url = _health_url(
            str(self._settings.knowledge_admin_internal_url or self._settings.knowledge_admin_url)
        )
        try:
            response = await self._client.get(health_url)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("database") != "connected":
                raise ValueError("invalid health response")
        except (httpx.HTTPError, json.JSONDecodeError, TypeError, ValueError):
            return _check(
                check_id="knowledge_admin",
                category="服務",
                title="知識治理中心",
                status=DiagnosticStatus.WARNING,
                summary="orchestrator 無法從設定網址取得 knowledge-admin 健康狀態。",
                evidence=[f"健康檢查：{_safe_url(health_url)}"],
                remediation=[
                    "容器間請使用可解析的 service DNS，例如 http://knowledge-admin:8081/admin/knowledge。",
                    "若設定的是給瀏覽器使用的網址，請確認 orchestrator 容器也能解析 DNS "
                    "與信任 TLS 憑證。",
                ],
                started_at=started_at,
            )
        return _check(
            check_id="knowledge_admin",
            category="服務",
            title="知識治理中心",
            status=DiagnosticStatus.PASS,
            summary="knowledge-admin 可達，且其資料庫連線正常。",
            evidence=[f"健康檢查：{_safe_url(health_url)}"],
            remediation=[],
            started_at=started_at,
        )

    async def _check_llm(self) -> DiagnosticCheck:
        started_at = perf_counter()
        models = self._configured_llm_models()
        if not models:
            return _check(
                check_id="llm",
                category="模型 API",
                title="LLM OpenAI 相容介面",
                status=DiagnosticStatus.SKIPPED,
                summary="目前沒有啟用需要 LLM 的線上功能。",
                evidence=[f"LLM Base URL：{_safe_url(str(self._settings.llm_base_url))}"],
                remediation=[
                    "若要測試自然對話，設定 SVA_NATURAL_ANSWER_ENABLED=true "
                    "與 SVA_ANSWER_LLM_MODEL。"
                ],
                started_at=started_at,
            )

        headers = _bearer_headers(
            self._settings.llm_api_key.get_secret_value() if self._settings.llm_api_key else None
        )
        endpoint = f"{str(self._settings.llm_base_url).rstrip('/')}/chat/completions"
        schema = {
            "type": "object",
            "properties": {"diagnostic_status": {"type": "string", "enum": ["ok"]}},
            "required": ["diagnostic_status"],
            "additionalProperties": False,
        }
        try:
            for model in models:
                output_mode = resolve_structured_output_mode(
                    mode=self._settings.llm_structured_output_mode,
                    model=model,
                )
                if output_mode == "tool_call":
                    system_message = (
                        "這是連線診斷。不得直接輸出文字或 JSON；"
                        "必須呼叫 system_diagnostic 工具回傳結果。"
                    )
                    user_message = "請執行連線診斷。"
                else:
                    system_message = "這是連線診斷。只輸出符合 schema 的 JSON。"
                    user_message = '{"diagnostic_status":"ok"}'
                response = await self._client.post(
                    endpoint,
                    headers=headers,
                    json={
                        "model": model,
                        "temperature": 0,
                        "max_tokens": max(
                            self._settings.answer_llm_max_tokens,
                            self._settings.intent_llm_max_tokens,
                            self._settings.conversation_llm_max_tokens,
                        ),
                        **structured_output_options(
                            name="system_diagnostic",
                            schema=schema,
                            mode=self._settings.llm_structured_output_mode,
                            model=model,
                        ),
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message},
                        ],
                    },
                )
                response.raise_for_status()
                content = structured_output_content(
                    response.json(),
                    name="system_diagnostic",
                    mode=self._settings.llm_structured_output_mode,
                    model=model,
                )
                parsed = json.loads(content)
                if parsed != {"diagnostic_status": "ok"}:
                    raise ValueError("unexpected structured response")
        except httpx.TimeoutException:
            return _llm_failure(
                summary="LLM 請求逾時。",
                endpoint=endpoint,
                models=models,
                started_at=started_at,
                extra="提高 SVA_SYSTEM_DIAGNOSTICS_TIMEOUT_SECONDS，並先預熱模型。",
            )
        except httpx.HTTPStatusError as error:
            return _llm_failure(
                summary=(
                    f"LLM API 拒絕 structured output 測試（HTTP {error.response.status_code}）。"
                ),
                endpoint=endpoint,
                models=models,
                started_at=started_at,
                extra="確認 API key、模型 ID，以及服務是否支援設定的 Structured Output 模式。",
            )
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
            return _llm_failure(
                summary="LLM API 無法連線，或回應不符合 OpenAI structured output 格式。",
                endpoint=endpoint,
                models=models,
                started_at=started_at,
                extra=(
                    "確認 Base URL 以 /v1 結尾，並以同一模型測試 /chat/completions；"
                    "若 finish_reason=length，請提高對應的 SVA_*_LLM_MAX_TOKENS。"
                ),
            )
        return _check(
            check_id="llm",
            category="模型 API",
            title="LLM OpenAI 相容介面",
            status=DiagnosticStatus.PASS,
            summary="所有已啟用的 LLM 模型皆通過 Chat Completions 與結構化輸出測試。",
            evidence=[
                f"端點：{_safe_url(endpoint)}",
                f"模型：{', '.join(models)}",
                f"設定模式：{self._settings.llm_structured_output_mode}",
            ],
            remediation=[],
            started_at=started_at,
        )

    async def _check_embeddings(self) -> DiagnosticCheck:
        started_at = perf_counter()
        if self._settings.retrieval_mode != "hybrid":
            return _check(
                check_id="embeddings",
                category="模型 API",
                title="Embedding API",
                status=DiagnosticStatus.SKIPPED,
                summary="目前使用 lexical 檢索，不會呼叫 embedding API。",
                evidence=["檢索模式：lexical"],
                remediation=[],
                started_at=started_at,
            )
        endpoint = f"{str(self._settings.embeddings_base_url).rstrip('/')}/embeddings"
        headers = _bearer_headers(
            self._settings.embeddings_api_key.get_secret_value()
            if self._settings.embeddings_api_key
            else None
        )
        try:
            response = await self._client.post(
                endpoint,
                headers=headers,
                json={
                    "model": self._settings.embeddings_model,
                    "input": [f"{self._settings.embeddings_query_prefix}系統診斷測試"],
                },
            )
            response.raise_for_status()
            payload = response.json()
            vector = payload["data"][0]["embedding"]
            if (
                not isinstance(vector, list)
                or not vector
                or any(not isinstance(value, (int, float)) for value in vector)
                or any(not math.isfinite(float(value)) for value in vector)
            ):
                raise ValueError("invalid vector")
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
            return _check(
                check_id="embeddings",
                category="模型 API",
                title="Embedding API",
                status=DiagnosticStatus.FAIL,
                summary="Hybrid retrieval 已啟用，但 embedding API 測試失敗。",
                evidence=[
                    f"端點：{_safe_url(endpoint)}",
                    f"模型：{self._settings.embeddings_model or '未設定'}",
                ],
                remediation=[
                    "確認服務支援 POST /v1/embeddings、模型 ID 與 Authorization。",
                    "確認 query/document prefix 與公司評測時的設定一致。",
                ],
                started_at=started_at,
            )
        return _check(
            check_id="embeddings",
            category="模型 API",
            title="Embedding API",
            status=DiagnosticStatus.PASS,
            summary="Embedding API 回傳有效向量。",
            evidence=[
                f"模型：{self._settings.embeddings_model}",
                f"向量維度：{len(vector)}",
            ],
            remediation=[],
            started_at=started_at,
        )

    async def _check_tts(self) -> DiagnosticCheck:
        started_at = perf_counter()
        if not self._settings.voice_enabled or self._voice_service is None:
            return _check(
                check_id="tts",
                category="語音 API",
                title="TTS 實際合成",
                status=DiagnosticStatus.FAIL,
                summary="語音服務未建立，無法執行 TTS。",
                evidence=[f"TTS 模型：{self._settings.tts_model or '未設定'}"],
                remediation=["啟用 SVA_VOICE_ENABLED，並設定 SVA_TTS_BASE_URL 與 SVA_TTS_MODEL。"],
                started_at=started_at,
            )

        try:
            async with asyncio.timeout(self._timeout_seconds):
                service_available = await self._voice_service.available()
        except TimeoutError:
            service_available = False
        audio_chunks = 0
        synthesis_error_type: str | None = None
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async for raw_event in self._voice_service.stream_answer(
                    turn_id=f"system-diagnostic-{uuid4()}",
                    answer="系統語音測試。",
                ):
                    event = json.loads(raw_event)
                    if event.get("type") == "audio":
                        audio_chunks += 1
                    elif event.get("type") == "error":
                        synthesis_error_type = str(event.get("error_type") or "invalid_audio")
        except TimeoutError:
            return _check(
                check_id="tts",
                category="語音 API",
                title="TTS 實際合成",
                status=DiagnosticStatus.FAIL,
                summary="TTS 合成逾時。",
                evidence=[
                    f"端點：{_safe_url(str(self._settings.tts_base_url))}",
                    f"模型：{self._settings.tts_model}",
                ],
                remediation=[
                    "確認模型已載入，並提高 SVA_SYSTEM_DIAGNOSTICS_TIMEOUT_SECONDS 或預熱 TTS。"
                ],
                started_at=started_at,
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            synthesis_error_type = "invalid_audio"

        if synthesis_error_type or audio_chunks == 0:
            if synthesis_error_type == "upstream_unavailable":
                summary = "orchestrator 無法連線至 TTS API。"
                remediation = [
                    "確認 TTS 服務已啟動，並從 orchestrator 容器測試 "
                    "SVA_TTS_BASE_URL 的 DNS、port、防火牆與 TLS 憑證。",
                    "Docker Compose 可用 SVA_TTS_DOCKER_BASE_URL 指向宿主機或公司服務 DNS。",
                ]
            elif synthesis_error_type == "upstream_rejected":
                summary = "TTS API 拒絕語音合成請求。"
                remediation = [
                    "確認 POST /v1/audio/speech 支援目前模型、voice、語言與 voice clone payload。",
                    "Base voice-clone 模型需確認 ref_audio 與 ref_text 均存在，"
                    "且路徑是 TTS 主機可讀取的位置。",
                ]
            else:
                summary = "TTS 端點未產生本系統可解析的 WAV 音訊。"
                remediation = [
                    "確認 POST /v1/audio/speech 回傳完整 WAV frame，不是 JSON、MP3 或 raw PCM。",
                ]
            return _check(
                check_id="tts",
                category="語音 API",
                title="TTS 實際合成",
                status=DiagnosticStatus.FAIL,
                summary=summary,
                evidence=[
                    f"端點：{_safe_url(str(self._settings.tts_base_url))}/audio/speech",
                    f"模型：{self._settings.tts_model}",
                    f"Voice clone：{'啟用' if self._voice_service.voice_clone_enabled else '停用'}",
                ],
                remediation=remediation,
                started_at=started_at,
            )
        status = DiagnosticStatus.PASS if service_available else DiagnosticStatus.WARNING
        return _check(
            check_id="tts",
            category="語音 API",
            title="TTS 實際合成",
            status=status,
            summary=(
                "TTS 合成與 WAV 解析成功。"
                if service_available
                else "TTS 合成成功，但服務根路徑健康檢查不是 2xx。"
            ),
            evidence=[
                f"模型：{self._settings.tts_model}",
                f"WAV 音訊片段：{audio_chunks}",
            ],
            remediation=(
                []
                if service_available
                else ["讓 TTS Base URL 的服務根路徑回傳 2xx，避免語音測試頁誤判未就緒。"]
            ),
            started_at=started_at,
        )

    def _check_asr_configuration(self) -> tuple[DiagnosticCheck, BrowserASRProbe | None]:
        started_at = perf_counter()
        if (
            not self._settings.voice_enabled
            or self._voice_service is None
            or not self._settings.asr_model
        ):
            return (
                _check(
                    check_id="asr_configuration",
                    category="語音 API",
                    title="ASR Realtime 設定",
                    status=DiagnosticStatus.FAIL,
                    summary="ASR Realtime 設定不完整。",
                    evidence=[f"ASR 模型：{self._settings.asr_model or '未設定'}"],
                    remediation=[
                        "啟用 SVA_VOICE_ENABLED，並設定 SVA_ASR_MODEL "
                        "與 SVA_AUDIO_PUBLIC_BASE_URL。"
                    ],
                    started_at=started_at,
                ),
                None,
            )

        model = self._settings.asr_model
        url = realtime_asr_url(self._voice_service.audio_public_base_url)
        parsed = urlsplit(url)
        status = DiagnosticStatus.PASS
        summary = "ASR Realtime URL 與模型設定完整，接著由瀏覽器實際測試 WebSocket。"
        remediation: list[str] = []
        if parsed.scheme == "ws" and not _is_loopback_host(parsed.hostname):
            status = DiagnosticStatus.WARNING
            summary = "ASR 使用未加密 WebSocket；HTTPS 頁面會阻擋這個連線。"
            remediation.append("將 ASR 對外入口改為 HTTPS/WSS，並設定有效的公司 TLS 憑證。")
        check = _check(
            check_id="asr_configuration",
            category="語音 API",
            title="ASR Realtime 設定",
            status=status,
            summary=summary,
            evidence=[f"WebSocket：{_safe_url(url)}", f"模型：{model}"],
            remediation=remediation,
            started_at=started_at,
        )
        probe = BrowserASRProbe(
            url=url,
            model=model,
            init_message={
                "model": model,
                "language": "Chinese" if "Qwen3-ASR" in model else "zh",
                "sample_rate": 16_000,
                "streaming": "Qwen3-ASR" in model,
                "semantic_endpointing": True,
                "output_script": "traditional",
            },
            timeout_ms=120_000,
        )
        return check, probe

    def _configured_llm_models(self) -> list[str]:
        candidates: list[str | None] = []
        if self._settings.answer_mode in {"shadow_llm", "controlled_llm"}:
            candidates.append(self._settings.answer_llm_model)
        if self._settings.natural_answer_enabled:
            candidates.append(self._settings.answer_llm_model)
        if self._settings.intent_router_mode != "disabled":
            candidates.append(self._settings.intent_llm_model)
        if self._settings.conversation_semantic_mode != "disabled":
            candidates.append(self._settings.conversation_llm_model)
        return list(dict.fromkeys(model for model in candidates if model))


def _check(
    *,
    check_id: str,
    category: str,
    title: str,
    status: DiagnosticStatus,
    summary: str,
    evidence: list[str],
    remediation: list[str],
    started_at: float,
) -> DiagnosticCheck:
    return DiagnosticCheck(
        check_id=check_id,
        category=category,
        title=title,
        status=status,
        summary=summary,
        evidence=evidence,
        remediation=remediation,
        duration_ms=round((perf_counter() - started_at) * 1_000, 3),
    )


def _llm_failure(
    *,
    summary: str,
    endpoint: str,
    models: list[str],
    started_at: float,
    extra: str,
) -> DiagnosticCheck:
    return _check(
        check_id="llm",
        category="模型 API",
        title="LLM OpenAI 相容介面",
        status=DiagnosticStatus.FAIL,
        summary=summary,
        evidence=[f"端點：{_safe_url(endpoint)}", f"模型：{', '.join(models)}"],
        remediation=[
            extra,
            "從 orchestrator 容器確認 DNS、port、防火牆與公司 CA 憑證，不要只從 Docker host 測試。",
        ],
        started_at=started_at,
    )


def _overall_status(checks: list[DiagnosticCheck]) -> DiagnosticStatus:
    statuses = {check.status for check in checks}
    if DiagnosticStatus.FAIL in statuses:
        return DiagnosticStatus.FAIL
    if DiagnosticStatus.WARNING in statuses:
        return DiagnosticStatus.WARNING
    return DiagnosticStatus.PASS


def _health_url(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, "/healthz", "", ""))


def _safe_url(url: str) -> str:
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    try:
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
    except ValueError:
        netloc = hostname
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _is_loopback_host(hostname: str | None) -> bool:
    return (hostname or "").casefold() in {"127.0.0.1", "::1", "localhost"}


def _bearer_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}
