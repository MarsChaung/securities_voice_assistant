from pathlib import Path
from uuid import UUID

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from answer_contract import TurnFeedback, TurnRequest, TurnResponse
from observability import configure_logging
from retrieval import (
    HybridKnowledgeRetriever,
    LexicalKnowledgeRetriever,
    OpenAICompatibleEmbeddingClient,
    SqlKnowledgeRepository,
)

from .answering import OpenAICompatibleAnswerComposer
from .config import Settings, get_settings
from .intent_routing import OpenAICompatibleIntentRouter
from .service import TurnService

_PACKAGE_ROOT = Path(__file__).parent


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
    if settings.answer_mode != "controlled_llm":
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
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    resolved_service = service or TurnService(
        knowledge_repository=SqlKnowledgeRepository.from_url(resolved_settings.database_url),
        knowledge_retriever=_build_knowledge_retriever(resolved_settings),
        answer_mode=resolved_settings.answer_mode,
        answer_composer=_build_answer_composer(resolved_settings),
        intent_router_mode=resolved_settings.intent_router_mode,
        intent_router=_build_intent_router(resolved_settings),
        intent_router_minimum_confidence=(
            resolved_settings.intent_router_minimum_confidence
        ),
    )

    app = FastAPI(
        title="Securities Voice Assistant Orchestrator",
        version="0.5.0",
        description="證券知識型語音客服的安全決策與核准知識檢索 API",
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
            context={"knowledge_admin_url": str(resolved_settings.knowledge_admin_url)},
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

    return app


app = create_app()
