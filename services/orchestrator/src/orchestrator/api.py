import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
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

from .answering import (
    OpenAICompatibleAnswerComposer,
    OpenAICompatibleNaturalAnswerComposer,
)
from .config import Settings, get_settings
from .conversation import (
    ConversationContextStore,
    ConversationExchange,
    ConversationResolution,
    FollowUpResolver,
    OpenAICompatibleConversationSemanticAnalyzer,
    ReplyMode,
)
from .diagnostics import VoiceTestDiagnosticLogger
from .intent_routing import OpenAICompatibleIntentRouter
from .service import TurnService
from .shadow import ThreadedShadowAnswerRunner
from .system_diagnostics import (
    DiagnosticReport,
    DiagnosticRunRequest,
    SystemDiagnosticRunner,
    SystemDiagnosticRunnerProtocol,
)
from .voice import (
    BARGE_IN_PRESETS,
    VOICE_FAREWELL_MESSAGE,
    VOICE_IDLE_CHECK_IN_MESSAGE,
    VOICE_IDLE_FAREWELL_MESSAGE,
    VoiceGreetingRequest,
    VoiceIdlePromptRequest,
    VoicePlaybackMetrics,
    VoiceReplyRequest,
    VoiceService,
    VoiceTestTurnRequest,
    is_call_ending_utterance,
    ndjson_event,
    realtime_asr_url,
    select_voice_acknowledgement_variant,
    split_tts_text,
)

_PACKAGE_ROOT = Path(__file__).parent
_PILOT_ASSET_VERSION = "20260817.1"
_VOICE_ACKNOWLEDGEMENT_AUDIO_URLS = {
    "context_confirmation": (
        f"/pilot/static/audio/acknowledgement-confirm.mp3?v={_PILOT_ASSET_VERSION}"
    ),
    "context_wait": (f"/pilot/static/audio/voice-acknowledgement.wav?v={_PILOT_ASSET_VERSION}"),
    "follow_up_explanation": (
        f"/pilot/static/audio/acknowledgement-explain.mp3?v={_PILOT_ASSET_VERSION}"
    ),
    "knowledge_lookup": (
        f"/pilot/static/audio/acknowledgement-lookup.mp3?v={_PILOT_ASSET_VERSION}"
    ),
}


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
        max_tokens=settings.answer_llm_max_tokens,
        structured_output_mode=settings.llm_structured_output_mode,
    )


def _build_natural_answer_composer(
    settings: Settings,
) -> OpenAICompatibleNaturalAnswerComposer | None:
    if not settings.natural_answer_enabled:
        return None

    return OpenAICompatibleNaturalAnswerComposer(
        base_url=str(settings.llm_base_url),
        model=settings.answer_llm_model or "",
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
        timeout_seconds=settings.answer_llm_timeout_seconds,
        max_tokens=settings.answer_llm_max_tokens,
        structured_output_mode=settings.llm_structured_output_mode,
    )


def _build_intent_router(settings: Settings) -> OpenAICompatibleIntentRouter | None:
    if settings.intent_router_mode == "disabled":
        return None

    return OpenAICompatibleIntentRouter(
        base_url=str(settings.llm_base_url),
        model=settings.intent_llm_model or "",
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
        timeout_seconds=settings.intent_llm_timeout_seconds,
        max_tokens=settings.intent_llm_max_tokens,
        structured_output_mode=settings.llm_structured_output_mode,
    )


def _build_conversation_semantic_analyzer(
    settings: Settings,
) -> OpenAICompatibleConversationSemanticAnalyzer | None:
    if settings.conversation_semantic_mode == "disabled":
        return None

    return OpenAICompatibleConversationSemanticAnalyzer(
        base_url=str(settings.llm_base_url),
        model=settings.conversation_llm_model or "",
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
        timeout_seconds=settings.conversation_llm_timeout_seconds,
        max_tokens=settings.conversation_llm_max_tokens,
        structured_output_mode=settings.llm_structured_output_mode,
    )


def create_app(
    *,
    service: TurnService | None = None,
    settings: Settings | None = None,
    voice_service: VoiceService | None = None,
    conversation_store: ConversationContextStore | None = None,
    follow_up_resolver: FollowUpResolver | None = None,
    diagnostic_logger: VoiceTestDiagnosticLogger | None = None,
    system_diagnostic_runner: SystemDiagnosticRunnerProtocol | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    shadow_runner: ThreadedShadowAnswerRunner | None = None
    knowledge_repository: SqlKnowledgeRepository | None = None
    knowledge_retriever: LexicalKnowledgeRetriever | HybridKnowledgeRetriever | None = None
    if service is None:
        answer_composer = _build_answer_composer(resolved_settings)
        natural_answer_composer = _build_natural_answer_composer(resolved_settings)
        if resolved_settings.answer_mode == "shadow_llm" and answer_composer is not None:
            shadow_runner = ThreadedShadowAnswerRunner(
                composer=answer_composer,
                review_writer=DatabaseShadowReviewRepository.from_url(
                    resolved_settings.database_url
                ),
                max_pending=resolved_settings.shadow_max_pending,
            )
        knowledge_repository = SqlKnowledgeRepository.from_url(resolved_settings.database_url)
        knowledge_retriever = _build_knowledge_retriever(resolved_settings)
        resolved_service = TurnService(
            knowledge_repository=knowledge_repository,
            knowledge_retriever=knowledge_retriever,
            answer_mode=resolved_settings.answer_mode,
            answer_composer=(
                answer_composer if resolved_settings.answer_mode == "controlled_llm" else None
            ),
            natural_answer_composer=natural_answer_composer,
            shadow_runner=shadow_runner,
            intent_router_mode=resolved_settings.intent_router_mode,
            intent_router=_build_intent_router(resolved_settings),
            intent_router_minimum_confidence=(resolved_settings.intent_router_minimum_confidence),
        )
    else:
        resolved_service = service
    resolved_conversation_store = conversation_store or ConversationContextStore()
    resolved_follow_up_resolver = follow_up_resolver or FollowUpResolver(
        semantic_mode=resolved_settings.conversation_semantic_mode,
        semantic_analyzer=_build_conversation_semantic_analyzer(resolved_settings),
        semantic_minimum_confidence=(resolved_settings.conversation_semantic_minimum_confidence),
    )
    resolved_diagnostic_logger = diagnostic_logger or VoiceTestDiagnosticLogger(
        enabled=resolved_settings.voice_test_content_logging_enabled,
        app_env=resolved_settings.app_env,
    )
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
    resolved_system_diagnostic_runner = system_diagnostic_runner
    if resolved_system_diagnostic_runner is None and resolved_settings.system_diagnostics_enabled:
        resolved_system_diagnostic_runner = SystemDiagnosticRunner(
            settings=resolved_settings,
            service=resolved_service,
            voice_service=resolved_voice_service,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if knowledge_repository is not None and isinstance(
            knowledge_retriever, HybridKnowledgeRetriever
        ):
            try:
                documents = knowledge_repository.eligible_documents(at=datetime.now(UTC))
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
        if resolved_system_diagnostic_runner is not None:
            await resolved_system_diagnostic_runner.close()
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

    def resolve_conversation(
        *,
        transcript: str,
        reply_mode: ReplyMode,
        session_id: str | None,
    ) -> ConversationResolution | None:
        if reply_mode is not ReplyMode.NATURAL or session_id is None:
            return None
        return resolved_follow_up_resolver.resolve(
            utterance=transcript,
            history=resolved_conversation_store.history(session_id),
        )

    def record_session_turn(
        *,
        request: TurnRequest,
        reply_mode: ReplyMode,
        session_id: str | None,
        conversation: ConversationResolution | None,
        turn: TurnResponse,
    ) -> None:
        contains_sensitive_data = turn.result.intent == "sensitive_data_detected"
        if session_id is not None and not contains_sensitive_data:
            resolved_conversation_store.append(
                session_id,
                ConversationExchange(
                    user_utterance=request.transcript,
                    resolved_query=(
                        conversation.retrieval_query
                        if conversation is not None
                        else request.transcript
                    ),
                    assistant_answer=turn.result.answer,
                    decision=turn.result.decision.value,
                    knowledge_id=turn.result.answer_id,
                    knowledge_version=(
                        turn.result.knowledge_versions[0]
                        if turn.result.knowledge_versions
                        else None
                    ),
                ),
            )
        resolved_diagnostic_logger.exchange(
            session_id=session_id,
            turn_id=turn.turn_id,
            channel=request.channel,
            reply_mode=reply_mode.value,
            user_utterance=request.transcript,
            assistant_answer=turn.result.answer,
            decision=turn.result.decision.value,
            intent=turn.result.intent,
            policy_rule_id=turn.result.policy_rule_id,
            answer_id=turn.result.answer_id,
            knowledge_versions=turn.result.knowledge_versions,
            contains_sensitive_data=contains_sensitive_data,
            follow_up_kind=(conversation.kind.value if conversation is not None else None),
            semantic_applied=(conversation.semantic_applied if conversation is not None else False),
            semantic_confidence=(
                conversation.semantic_confidence if conversation is not None else None
            ),
            reference_knowledge_id=(
                conversation.reference_knowledge_id if conversation is not None else None
            ),
            semantic_focus=(conversation.focus if conversation is not None else None),
            resolved_query=(
                conversation.retrieval_query if conversation is not None else request.transcript
            ),
        )

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

    @app.get("/system-diagnostics", response_class=HTMLResponse, include_in_schema=False)
    def system_diagnostics(request: Request) -> Response:
        if resolved_system_diagnostic_runner is None:
            raise HTTPException(404, "找不到資源。")
        return templates.TemplateResponse(
            request=request,
            name="system_diagnostics.html",
            context={
                "knowledge_admin_url": str(resolved_settings.knowledge_admin_url),
                "pilot_asset_version": _PILOT_ASSET_VERSION,
            },
        )

    @app.post(
        "/v1/system-diagnostics/run",
        response_model=DiagnosticReport,
    )
    async def run_system_diagnostics(request: DiagnosticRunRequest) -> DiagnosticReport:
        if resolved_system_diagnostic_runner is None:
            raise HTTPException(404, "找不到資源。")
        return await resolved_system_diagnostic_runner.run(
            page_origin=request.page_origin,
            secure_context=request.secure_context,
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
                "natural_answer_enabled": resolved_service.natural_answer_available,
                "intent_router_mode": resolved_settings.intent_router_mode,
                "conversation_semantic_mode": resolved_settings.conversation_semantic_mode,
                "voice_enabled": resolved_voice_service is not None,
            },
            status_code=status_code,
        )

    @app.post("/v1/turns/evaluate", response_model=TurnResponse)
    def evaluate_turn(request: TurnRequest) -> TurnResponse:
        return resolved_service.evaluate(request)

    @app.post("/v1/voice/test-turns/evaluate", response_model=TurnResponse)
    def evaluate_voice_test_turn(request: VoiceTestTurnRequest) -> TurnResponse:
        if resolved_settings.app_env != "development":
            raise HTTPException(404, "找不到資源。")
        if (
            request.reply_mode is ReplyMode.NATURAL
            and not resolved_service.natural_answer_available
        ):
            raise HTTPException(409, "自然對話模式目前未啟用。")
        session_id = str(request.session_id)
        conversation = resolve_conversation(
            transcript=request.transcript,
            reply_mode=request.reply_mode,
            session_id=session_id,
        )
        turn_request = TurnRequest(transcript=request.transcript, channel="web")
        turn = resolved_service.evaluate(turn_request, conversation=conversation)
        record_session_turn(
            request=turn_request,
            reply_mode=request.reply_mode,
            session_id=session_id,
            conversation=conversation,
            turn=turn,
        )
        return turn

    @app.post("/v1/turns/{turn_id}/feedback", status_code=204)
    def record_feedback(turn_id: UUID, feedback: TurnFeedback) -> Response:
        resolved_service.record_feedback(turn_id=str(turn_id), rating=feedback.rating)
        return Response(status_code=204)

    @app.get("/v1/voice/config")
    async def voice_config() -> Response:
        test_config = {
            "reply_modes": [
                {"id": ReplyMode.EXACT.value, "label": "核准原文"},
                *(
                    [{"id": ReplyMode.NATURAL.value, "label": "自然對話"}]
                    if resolved_service.natural_answer_available
                    else []
                ),
            ],
            "diagnostic_content_logging_enabled": resolved_diagnostic_logger.enabled,
            "conversation_semantic_mode": resolved_settings.conversation_semantic_mode,
        }
        if resolved_voice_service is None:
            return JSONResponse({"enabled": False, "available": False, **test_config})
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
                "realtime_asr_url": realtime_asr_url(resolved_voice_service.audio_public_base_url),
                "models": {
                    "asr": resolved_voice_service.models.asr,
                    "tts": resolved_voice_service.models.tts,
                    "voice": resolved_voice_service.models.voice,
                    "voice_clone": resolved_voice_service.voice_clone_enabled,
                },
                "asr_models": asr_models,
                "asr_context": resolved_service.voice_asr_context(),
                "asr_endpoint_grace_ms": resolved_settings.asr_endpoint_grace_ms,
                "acknowledgement": {
                    "audio_urls": _VOICE_ACKNOWLEDGEMENT_AUDIO_URLS,
                    "delay_ms": resolved_settings.voice_acknowledgement_delay_ms,
                },
                "barge_in": {
                    "enabled": resolved_settings.barge_in_enabled,
                    "default_mode": resolved_settings.barge_in_default_mode,
                    "presets": BARGE_IN_PRESETS,
                },
                **test_config,
            }
        )

    @app.post("/v1/voice/respond-stream")
    async def voice_respond(request: VoiceReplyRequest) -> StreamingResponse:
        if resolved_voice_service is None:
            raise HTTPException(503, "語音服務目前未啟用。")
        if (
            request.reply_mode is ReplyMode.NATURAL
            and not resolved_service.natural_answer_available
        ):
            raise HTTPException(409, "自然對話模式目前未啟用。")
        if is_call_ending_utterance(request.transcript):
            session_id = (
                str(request.conversation_id) if request.conversation_id is not None else None
            )
            if request.conversation_id is not None:
                resolved_conversation_store.clear(session_id or "")
            segments = split_tts_text(VOICE_FAREWELL_MESSAGE)
            farewell_turn_id = f"voice-farewell-{uuid4()}"
            resolved_diagnostic_logger.exchange(
                session_id=session_id,
                turn_id=farewell_turn_id,
                channel="voice",
                reply_mode=request.reply_mode.value,
                user_utterance=request.transcript,
                assistant_answer=VOICE_FAREWELL_MESSAGE,
                decision="answer",
                intent="call_ending",
                policy_rule_id="VOICE-END-001",
                answer_id=None,
                knowledge_versions=[],
                contains_sensitive_data=False,
            )

            async def farewell_stream() -> AsyncIterator[bytes]:
                yield ndjson_event({"type": "farewell", "speech_segments": segments})
                async for event in resolved_voice_service.stream_answer(
                    turn_id=farewell_turn_id,
                    answer=VOICE_FAREWELL_MESSAGE,
                ):
                    yield event

            return StreamingResponse(
                farewell_stream(),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-store"},
            )

        session_id = str(request.conversation_id) if request.conversation_id is not None else None
        turn_request = TurnRequest(transcript=request.transcript, channel="voice")
        preflight = resolved_service.preflight(turn_request)

        async def emit_turn_and_audio(turn: TurnResponse) -> AsyncIterator[bytes]:
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

        if request.reply_mode is ReplyMode.NATURAL:
            end_to_end_started_at = perf_counter()

            conversation_task = asyncio.create_task(
                run_in_threadpool(
                    resolve_conversation,
                    transcript=request.transcript,
                    reply_mode=request.reply_mode,
                    session_id=session_id,
                )
            )
            intent_task = asyncio.create_task(
                run_in_threadpool(
                    resolved_service.prefetch_intent_route,
                    turn_request,
                    preflight=preflight,
                )
            )

            async def resolve_natural_turn() -> tuple[ConversationResolution, TurnResponse]:
                conversation, prefetched_intent_route = await asyncio.gather(
                    conversation_task,
                    intent_task,
                )
                assert conversation is not None
                turn = await run_in_threadpool(
                    resolved_service.evaluate,
                    turn_request,
                    conversation=conversation,
                    preflight=preflight,
                    prefetched_intent_route=prefetched_intent_route,
                    end_to_end_started_at=end_to_end_started_at,
                )
                return conversation, turn

            turn_task = asyncio.create_task(resolve_natural_turn())

            async def natural_stream() -> AsyncIterator[bytes]:
                resolved: tuple[ConversationResolution, TurnResponse] | None = None
                acknowledgement_variant: str | None = None
                try:
                    if preflight.allows_acknowledgement:
                        try:
                            resolved = await asyncio.wait_for(
                                asyncio.shield(turn_task),
                                timeout=(resolved_settings.voice_acknowledgement_delay_ms / 1_000),
                            )
                        except TimeoutError:
                            conversation_pending = not conversation_task.done()
                            acknowledgement_conversation = None
                            if (
                                not conversation_pending
                                and not conversation_task.cancelled()
                                and conversation_task.exception() is None
                            ):
                                acknowledgement_conversation = conversation_task.result()
                            acknowledgement_variant = select_voice_acknowledgement_variant(
                                acknowledgement_conversation,
                                conversation_pending=conversation_pending,
                            )
                            acknowledgement_event: dict[str, object] = {
                                "type": "acknowledgement",
                                "variant": acknowledgement_variant,
                                "audio_url": _VOICE_ACKNOWLEDGEMENT_AUDIO_URLS[
                                    acknowledgement_variant
                                ],
                            }
                            if acknowledgement_variant == "context_confirmation":
                                acknowledgement_event["audio_urls"] = [
                                    _VOICE_ACKNOWLEDGEMENT_AUDIO_URLS["context_confirmation"],
                                    _VOICE_ACKNOWLEDGEMENT_AUDIO_URLS["context_wait"],
                                ]
                            yield ndjson_event(acknowledgement_event)
                    if resolved is None:
                        resolved = await turn_task
                    conversation, turn = resolved
                    if acknowledgement_variant is not None:
                        resolved_diagnostic_logger.acknowledgement(
                            session_id=session_id,
                            variant=acknowledgement_variant,
                            triggered_after_ms=(resolved_settings.voice_acknowledgement_delay_ms),
                            answer_ready_after_ms=(
                                (perf_counter() - end_to_end_started_at) * 1_000
                            ),
                        )
                    record_session_turn(
                        request=turn_request,
                        reply_mode=request.reply_mode,
                        session_id=session_id,
                        conversation=conversation,
                        turn=turn,
                    )
                    async for event in emit_turn_and_audio(turn):
                        yield event
                finally:
                    if not turn_task.done():
                        turn_task.cancel()
                    if not conversation_task.done():
                        conversation_task.cancel()
                    if not intent_task.done():
                        intent_task.cancel()

            return StreamingResponse(
                natural_stream(),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-store"},
            )

        turn = await run_in_threadpool(
            resolved_service.evaluate,
            turn_request,
            conversation=None,
            preflight=preflight,
        )
        record_session_turn(
            request=turn_request,
            reply_mode=request.reply_mode,
            session_id=session_id,
            conversation=None,
            turn=turn,
        )

        async def stream() -> AsyncIterator[bytes]:
            async for event in emit_turn_and_audio(turn):
                yield event

        return StreamingResponse(
            stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store"},
        )

    @app.delete("/v1/voice/conversations/{conversation_id}", status_code=204)
    def clear_voice_conversation(conversation_id: UUID) -> Response:
        resolved_conversation_store.clear(str(conversation_id))
        return Response(status_code=204)

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

    @app.post("/v1/voice/idle-prompt-stream")
    async def voice_idle_prompt(
        request: VoiceIdlePromptRequest,
    ) -> StreamingResponse:
        if resolved_voice_service is None:
            raise HTTPException(503, "語音服務目前未啟用。")
        ends_call = request.stage == "farewell"
        message = VOICE_IDLE_FAREWELL_MESSAGE if ends_call else VOICE_IDLE_CHECK_IN_MESSAGE
        segments = split_tts_text(message)

        async def stream() -> AsyncIterator[bytes]:
            yield ndjson_event(
                {
                    "type": "idle_prompt",
                    "stage": request.stage,
                    "ends_call": ends_call,
                    "speech_segments": segments,
                }
            )
            async for event in resolved_voice_service.stream_answer(
                turn_id=f"voice-idle-{request.stage}-{uuid4()}",
                answer=message,
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
