import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from answer_contract import TurnFeedback, TurnRequest, TurnResponse
from observability import DatabaseShadowReviewRepository, configure_logging
from retrieval import (
    EmbeddingServiceError,
    HybridKnowledgeRetriever,
    KnowledgeRepositoryError,
    LexicalKnowledgeRetriever,
    OpenAICompatibleEmbeddingClient,
    SqlKnowledgeRepository,
)

from .answering import OpenAICompatibleAnswerComposer
from .config import Settings, get_settings
from .intent_routing import OpenAICompatibleIntentRouter
from .service import TurnService
from .shadow import ThreadedShadowAnswerRunner
from .voice import (
    BARGE_IN_PRESETS,
    VOICE_FAREWELL_MESSAGE,
    VoiceGreetingRequest,
    VoicePlaybackMetrics,
    VoiceReplyRequest,
    VoiceService,
    is_call_ending_utterance,
    ndjson_event,
    realtime_asr_url,
    split_tts_text,
)

_PACKAGE_ROOT = Path(__file__).parent
_PILOT_ASSET_VERSION = "20260730.1"


def _build_knowledge_retriever(
    settings: Settings,
) -> LexicalKnowledgeRetriever | HybridKnowledgeRetriever:
    lexical_retriever = LexicalKnowledgeRetriever(
        minimum_score=settings.retrieval_minimum_score,
        ambiguity_margin=settings.retrieval_ambiguity_margin,
    )
    if settings.retrieval_mode == "lexical":
        return lexical_retriever

    return HybridKnowledgeRetriever(
        lexical_retriever=lexical_retriever,
        embedder=OpenAICompatibleEmbeddingClient(
            base_url=str(settings.embeddings_base_url),
            model=settings.embeddings_model or "",
            timeout_seconds=settings.embeddings_timeout_seconds,
            api_key=(
                settings.embeddings_api_key.get_secret_value()
                if settings.embeddings_api_key
                else None
            ),
        ),
        query_prefix=settings.embeddings_query_prefix,
        document_prefix=settings.embeddings_document_prefix,
        minimum_score=settings.hybrid_retrieval_minimum_score,
        ambiguity_margin=settings.hybrid_retrieval_ambiguity_margin,
    )


def _build_answer_composer(settings: Settings) -> OpenAICompatibleAnswerComposer | None:
    if settings.answer_mode not in {"shadow_llm", "controlled_llm"}:
        return None

    return OpenAICompatibleAnswerComposer(
        base_url=str(settings.llm_base_url),
        model=settings.answer_llm_model or "",
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
        timeout_seconds=settings.answer_llm_timeout_seconds,
    )


def _build_intent_router(settings: Settings) -> OpenAICompatibleIntentRouter | None:
    if settings.intent_router_mode == "disabled":
        return None

    return OpenAICompatibleIntentRouter(
        base_url=str(settings.llm_base_url),
        model=settings.intent_llm_model or "",
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
        timeout_seconds=settings.intent_llm_timeout_seconds,
    )


def create_app(
    *,
    service: TurnService | None = None,
    settings: Settings | None = None,
    voice_service: VoiceService | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    shadow_runner: ThreadedShadowAnswerRunner | None = None
    knowledge_repository: SqlKnowledgeRepository | None = None
    knowledge_retriever: LexicalKnowledgeRetriever | HybridKnowledgeRetriever | None = None
    if service is None:
        answer_composer = _build_answer_composer(resolved_settings)
        if resolved_settings.answer_mode == "shadow_llm" and answer_composer is not None:
            shadow_runner = ThreadedShadowAnswerRunner(
                composer=answer_composer,
                review_writer=DatabaseShadowReviewRepository.from_url(
                    resolved_settings.database_url
                ),
                max_pending=resolved_settings.shadow_max_pending,
            )
        knowledge_repository = SqlKnowledgeRepository.from_url(
            resolved_settings.database_url
        )
        knowledge_retriever = _build_knowledge_retriever(resolved_settings)
        resolved_service = TurnService(
            knowledge_repository=knowledge_repository,
            knowledge_retriever=knowledge_retriever,
            answer_mode=resolved_settings.answer_mode,
            answer_composer=(
                answer_composer
                if resolved_settings.answer_mode == "controlled_llm"
                else None
            ),
            shadow_runner=shadow_runner,
            intent_router_mode=resolved_settings.intent_router_mode,
            intent_router=_build_intent_router(resolved_settings),
            intent_router_minimum_confidence=(
                resolved_settings.intent_router_minimum_confidence
            ),
        )
    else:
        resolved_service = service
    resolved_voice_service = voice_service
    if resolved_voice_service is None and resolved_settings.voice_enabled:
        resolved_voice_service = VoiceService(
            audio_base_url=str(resolved_settings.tts_base_url),
            audio_public_base_url=str(resolved_settings.audio_public_base_url),
            asr_model=resolved_settings.asr_model or "",
            tts_model=resolved_settings.tts_model or "",
            tts_voice=resolved_settings.tts_voice,
            tts_ref_audio=resolved_settings.tts_ref_audio,
            tts_ref_text=(
                resolved_settings.tts_ref_text.get_secret_value()
                if resolved_settings.tts_ref_text
                else None
            ),
            timeout_seconds=resolved_settings.voice_timeout_seconds,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if (
            knowledge_repository is not None
            and isinstance(knowledge_retriever, HybridKnowledgeRetriever)
        ):
            try:
                documents = knowledge_repository.eligible_documents(
                    at=datetime.now(UTC)
                )
                warmed = knowledge_retriever.warm(documents)
                logging.getLogger("sva.retrieval").info(
                    "embedding warmup complete representations=%d",
                    warmed,
                )
            except (EmbeddingServiceError, KnowledgeRepositoryError):
                logging.getLogger("sva.retrieval").warning(
                    "embedding warmup incomplete; runtime will fail safe"
                )
        yield
        if shadow_runner is not None:
            shadow_runner.close()
        if resolved_voice_service is not None:
            await resolved_voice_service.close()

    app = FastAPI(
        title="Securities Voice Assistant Orchestrator",
        version="0.7.0",
        description="證券知識型語音客服的安全決策與核准知識檢索 API",
        lifespan=lifespan,
    )
    app.mount(
        "/pilot/static",
        StaticFiles(directory=_PACKAGE_ROOT / "static"),
        name="pilot-static",
    )
    templates = Jinja2Templates(directory=_PACKAGE_ROOT / "templates")

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse(url="/pilot", status_code=307)

    @app.get("/pilot", response_class=HTMLResponse, include_in_schema=False)
    def pilot(request: Request) -> Response:
        return templates.TemplateResponse(
            request=request,
            name="pilot.html",
            context={
                "knowledge_admin_url": str(resolved_settings.knowledge_admin_url),
                "pilot_asset_version": _PILOT_ASSET_VERSION,
            },
        )

    @app.get("/voice-test", response_class=HTMLResponse, include_in_schema=False)
    def voice_test(request: Request) -> Response:
        return templates.TemplateResponse(
            request=request,
            name="voice_test.html",
            context={
                "knowledge_admin_url": str(resolved_settings.knowledge_admin_url),
                "pilot_asset_version": _PILOT_ASSET_VERSION,
            },
        )

    @app.get("/healthz")
    def health() -> Response:
        availability = resolved_service.knowledge_availability()
        status_code = 503 if availability.status == "unavailable" else 200
        return JSONResponse(
            {
                "status": "ok" if status_code == 200 else "error",
                "environment": resolved_settings.app_env,
                "knowledge_database": availability.status,
                "eligible_knowledge_count": availability.eligible_document_count,
                "retrieval_mode": resolved_settings.retrieval_mode,
                "answer_mode": resolved_settings.answer_mode,
                "intent_router_mode": resolved_settings.intent_router_mode,
                "voice_enabled": resolved_voice_service is not None,
            },
            status_code=status_code,
        )

    @app.post("/v1/turns/evaluate", response_model=TurnResponse)
    def evaluate_turn(request: TurnRequest) -> TurnResponse:
        return resolved_service.evaluate(request)

    @app.post("/v1/turns/{turn_id}/feedback", status_code=204)
    def record_feedback(turn_id: UUID, feedback: TurnFeedback) -> Response:
        resolved_service.record_feedback(turn_id=str(turn_id), rating=feedback.rating)
        return Response(status_code=204)

    @app.get("/v1/voice/config")
    async def voice_config() -> Response:
        if resolved_voice_service is None:
            return JSONResponse({"enabled": False, "available": False})
        asr_models = list(
            dict.fromkeys(
                model
                for model in (
                    resolved_voice_service.models.asr,
                    resolved_settings.asr_candidate_model,
                )
                if model
            )
        )
        return JSONResponse(
            {
                "enabled": True,
                "available": await resolved_voice_service.available(),
                "realtime_asr_url": realtime_asr_url(
                    resolved_voice_service.audio_public_base_url
                ),
                "models": {
                    "asr": resolved_voice_service.models.asr,
                    "tts": resolved_voice_service.models.tts,
                    "voice": resolved_voice_service.models.voice,
                    "voice_clone": resolved_voice_service.voice_clone_enabled,
                },
                "asr_models": asr_models,
                "asr_context": resolved_service.voice_asr_context(),
                "asr_endpoint_grace_ms": resolved_settings.asr_endpoint_grace_ms,
                "barge_in": {
                    "enabled": resolved_settings.barge_in_enabled,
                    "default_mode": resolved_settings.barge_in_default_mode,
                    "presets": BARGE_IN_PRESETS,
                },
            }
        )

    @app.post("/v1/voice/respond-stream")
    async def voice_respond(request: VoiceReplyRequest) -> StreamingResponse:
        if resolved_voice_service is None:
            raise HTTPException(503, "語音服務目前未啟用。")
        if is_call_ending_utterance(request.transcript):
            segments = split_tts_text(VOICE_FAREWELL_MESSAGE)

            async def farewell_stream() -> AsyncIterator[bytes]:
                yield ndjson_event({"type": "farewell", "speech_segments": segments})
                async for event in resolved_voice_service.stream_answer(
                    turn_id=f"voice-farewell-{uuid4()}",
                    answer=VOICE_FAREWELL_MESSAGE,
                ):
                    yield event

            return StreamingResponse(
                farewell_stream(),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-store"},
            )

        turn = await run_in_threadpool(
            resolved_service.evaluate,
            TurnRequest(transcript=request.transcript, channel="voice"),
        )

        async def stream() -> AsyncIterator[bytes]:
            yield ndjson_event(
                {
                    "type": "turn",
                    "turn": turn.model_dump(mode="json"),
                    "speech_segments": split_tts_text(turn.result.answer),
                }
            )
            async for event in resolved_voice_service.stream_answer(
                turn_id=turn.turn_id,
                answer=turn.result.answer,
            ):
                yield event

        return StreamingResponse(
            stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/v1/voice/test-greeting-stream")
    async def voice_test_greeting(
        request: VoiceGreetingRequest,
    ) -> StreamingResponse:
        if resolved_settings.app_env != "development":
            raise HTTPException(404, "找不到資源。")
        if resolved_voice_service is None:
            raise HTTPException(503, "語音服務目前未啟用。")
        segments = split_tts_text(request.greeting)

        async def stream() -> AsyncIterator[bytes]:
            yield ndjson_event({"type": "greeting", "speech_segments": segments})
            async for event in resolved_voice_service.stream_answer(
                turn_id=f"voice-test-greeting-{uuid4()}",
                answer=request.greeting,
            ):
                yield event

        return StreamingResponse(
            stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/v1/voice/{turn_id}/playback-metrics", status_code=204)
    def record_voice_playback(turn_id: UUID, metrics: VoicePlaybackMetrics) -> Response:
        resolved_service.record_voice_playback(
            turn_id=str(turn_id),
            **metrics.model_dump(),
        )
        return Response(status_code=204)

    return app


app = create_app()
