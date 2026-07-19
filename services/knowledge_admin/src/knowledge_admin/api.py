from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from policy import SensitiveDataGuard
from retrieval import KnowledgeItem, KnowledgeStatus

from .config import KnowledgeAdminSettings
from .governance import GovernanceAction, GovernanceError, KnowledgeRole
from .identity import DevelopmentIdentityProvider
from .repository import (
    ConcurrentUpdateError,
    DatabaseKnowledgeRepository,
    GovernancePayload,
    KnowledgeNotFoundError,
)

PACKAGE_ROOT = Path(__file__).parent
TAIPEI = ZoneInfo("Asia/Taipei")


def create_app(
    *,
    repository: DatabaseKnowledgeRepository | None = None,
    settings: KnowledgeAdminSettings | None = None,
    identity_provider: DevelopmentIdentityProvider | None = None,
    clock: Callable[[], datetime] | None = None,
) -> FastAPI:
    resolved_settings = settings or KnowledgeAdminSettings()
    resolved_settings.validate_identity_mode()
    knowledge_repository = repository or DatabaseKnowledgeRepository.from_url(
        resolved_settings.database_url
    )
    identities = identity_provider or DevelopmentIdentityProvider()
    resolved_clock = clock or (lambda: datetime.now(UTC))

    app = FastAPI(
        title="證券知識治理中心",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    templates = Jinja2Templates(directory=str(PACKAGE_ROOT / "templates"))
    templates.env.globals.update(
        dev_identity_enabled=resolved_settings.knowledge_admin_dev_identity_enabled,
        action_labels=_ACTION_LABELS,
        format_event_time=_format_event_time,
    )
    app.mount("/static", StaticFiles(directory=PACKAGE_ROOT / "static"), name="static")

    @app.middleware("http")
    async def security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            expected_origin = f"{request.url.scheme}://{request.headers.get('host', '')}"
            if origin != expected_origin:
                response = Response("拒絕跨來源寫入", status_code=403)
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; img-src 'self' data:; "
            "form-action 'self'; frame-ancestors 'none'"
        )
        return response

    @app.get("/healthz")
    def healthz() -> Response:
        try:
            knowledge_repository.check_connection()
        except Exception:
            return JSONResponse(
                {"status": "error", "database": "unavailable"},
                status_code=503,
            )
        return JSONResponse(
            {
                "status": "ok",
                "database": "connected",
                "identity_mode": (
                    "development"
                    if resolved_settings.knowledge_admin_dev_identity_enabled
                    else "disabled"
                ),
            }
        )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/admin/knowledge", status_code=307)

    @app.get("/admin/knowledge", response_class=HTMLResponse)
    def knowledge_list(
        request: Request,
        status: Annotated[str | None, Query(max_length=30)] = None,
        source_id: str | None = None,
        q: Annotated[str | None, Query(max_length=100)] = None,
    ) -> Response:
        records = knowledge_repository.list_items()
        selected_status = _parse_status_filter(status)
        current_time = resolved_clock()
        items = _filter_items(
            tuple(record.item for record in records),
            status=selected_status,
            source_id=source_id,
            query=q,
            now=current_time,
        )
        sources = knowledge_repository.list_sources()
        counts = {
            knowledge_status: sum(
                _display_status(record.item, current_time) is knowledge_status for record in records
            )
            for knowledge_status in KnowledgeStatus
        }
        overdue_ids = {
            item.knowledge_id for item in items if _is_review_overdue(item, current_time)
        }
        return templates.TemplateResponse(
            request=request,
            name="knowledge_list.html",
            context={
                "items": items,
                "display_statuses": {
                    item.knowledge_id: _display_status(item, current_time) for item in items
                },
                "sources": sources,
                "counts": counts,
                "selected_status": selected_status,
                "selected_source_id": source_id,
                "query": q or "",
                "status_labels": _STATUS_LABELS,
                "overdue_ids": overdue_ids,
            },
        )

    @app.get("/admin/knowledge/{knowledge_id}", response_class=HTMLResponse)
    def knowledge_detail(
        request: Request,
        knowledge_id: str,
        notice: str | None = None,
        error: str | None = None,
    ) -> Response:
        try:
            record = knowledge_repository.get_item(knowledge_id)
            source = knowledge_repository.get_source(record.item.source_id)
        except KnowledgeNotFoundError as exc:
            raise HTTPException(status_code=404, detail="找不到知識項目") from exc
        current_time = resolved_clock()

        return templates.TemplateResponse(
            request=request,
            name="knowledge_detail.html",
            context={
                "item": record.item,
                "row_version": record.row_version,
                "source": source,
                "events": knowledge_repository.list_events(knowledge_id),
                "versions": knowledge_repository.list_versions(knowledge_id),
                "status_labels": _STATUS_LABELS,
                "display_status": _display_status(record.item, current_time),
                "review_overdue": _is_review_overdue(record.item, current_time),
                "actors_by_role": {
                    role.value: identities.actors_for(role) for role in KnowledgeRole
                },
                "notice": notice,
                "error": error,
            },
        )

    @app.post("/admin/knowledge/{knowledge_id}/actions/{action}")
    def perform_action(
        knowledge_id: str,
        action: GovernanceAction,
        actor_id: Annotated[str, Form()],
        expected_version: Annotated[int, Form()],
        effective_at: Annotated[str, Form()] = "",
        review_at: Annotated[str, Form()] = "",
        owner_unit: Annotated[str, Form(max_length=200)] = "",
        app_versions: Annotated[str, Form(max_length=500)] = "",
        reason: Annotated[str, Form(max_length=1000)] = "",
    ) -> RedirectResponse:
        if not resolved_settings.knowledge_admin_dev_identity_enabled:
            raise HTTPException(status_code=403, detail="開發身分模式未啟用")

        try:
            actor = identities.get_actor(actor_id)
            payload = GovernancePayload(
                effective_at=_parse_local_datetime(effective_at),
                review_at=_parse_local_datetime(review_at),
                owner_unit=owner_unit.strip() or None,
                app_versions=tuple(
                    version.strip() for version in app_versions.split(",") if version.strip()
                ),
                reason=reason.strip() or None,
            )
            if payload.reason and SensitiveDataGuard().scan(payload.reason).has_sensitive_data:
                raise ValueError("原因欄位不得包含個資、帳號、密碼或驗證碼")
            knowledge_repository.perform_action(
                knowledge_id=knowledge_id,
                action=action,
                actor=actor,
                expected_version=expected_version,
                payload=payload,
                now=resolved_clock(),
            )
        except (
            ConcurrentUpdateError,
            GovernanceError,
            KnowledgeNotFoundError,
            ValueError,
        ) as exc:
            return _detail_redirect(knowledge_id, error=str(exc))

        return _detail_redirect(knowledge_id, notice=f"已完成：{_ACTION_LABELS[action]}")

    @app.get("/admin/sources", response_class=HTMLResponse)
    def source_list(request: Request) -> Response:
        sources = knowledge_repository.list_sources()
        items = knowledge_repository.list_items()
        item_counts = {
            source.source_id: sum(record.item.source_id == source.source_id for record in items)
            for source in sources
        }
        return templates.TemplateResponse(
            request=request,
            name="source_list.html",
            context={"sources": sources, "item_counts": item_counts},
        )

    return app


def _detail_redirect(
    knowledge_id: str,
    *,
    notice: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    messages = {"notice": notice, "error": error}
    query = urlencode({key: value for key, value in messages.items() if value})
    return RedirectResponse(
        url=f"/admin/knowledge/{knowledge_id}?{query}",
        status_code=303,
    )


def _parse_local_datetime(value: str) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=TAIPEI)


def _format_event_time(value: datetime) -> str:
    timestamp = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return timestamp.astimezone(TAIPEI).strftime("%Y-%m-%d %H:%M")


def _is_review_overdue(item: KnowledgeItem, now: datetime) -> bool:
    if item.status is not KnowledgeStatus.PUBLISHED or item.review_at is None:
        return False
    deadline = (
        item.review_at if item.review_at.tzinfo is not None else item.review_at.replace(tzinfo=UTC)
    )
    current = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return deadline <= current


def _display_status(item: KnowledgeItem, now: datetime) -> KnowledgeStatus:
    if item.status is not KnowledgeStatus.PUBLISHED:
        return item.status
    current = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    deadlines = (item.expires_at, item.review_at)
    if any(
        deadline is not None
        and (deadline if deadline.tzinfo is not None else deadline.replace(tzinfo=UTC)) <= current
        for deadline in deadlines
    ):
        return KnowledgeStatus.EXPIRED
    return item.status


def _parse_status_filter(value: str | None) -> KnowledgeStatus | None:
    if value is None or not value.strip():
        return None
    try:
        return KnowledgeStatus(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="不支援的知識狀態篩選") from exc


def _filter_items(
    items: tuple[KnowledgeItem, ...],
    *,
    status: KnowledgeStatus | None,
    source_id: str | None,
    query: str | None,
    now: datetime,
) -> list[KnowledgeItem]:
    normalized_query = query.casefold().strip() if query else None
    return [
        item
        for item in items
        if (status is None or _display_status(item, now) is status)
        and (source_id is None or item.source_id == source_id)
        and (
            normalized_query is None
            or normalized_query in item.knowledge_id.casefold()
            or normalized_query in item.title.casefold()
            or normalized_query in item.standard_answer.casefold()
        )
    ]


_STATUS_LABELS = {
    KnowledgeStatus.DRAFT: "草稿",
    KnowledgeStatus.REVIEW: "審核中",
    KnowledgeStatus.APPROVED: "已核准",
    KnowledgeStatus.PUBLISHED: "已發布",
    KnowledgeStatus.EXPIRED: "已過期",
    KnowledgeStatus.REVOKED: "已撤銷",
}

_ACTION_LABELS = {
    GovernanceAction.START_REVISION: "建立複審新版",
    GovernanceAction.SUBMIT_REVIEW: "送交審核",
    GovernanceAction.COMPLETE_REVIEW: "完成審核",
    GovernanceAction.APPROVE: "核准",
    GovernanceAction.PUBLISH: "發布",
    GovernanceAction.RETURN_DRAFT: "退回草稿",
    GovernanceAction.REVOKE: "撤銷",
}


app = create_app()
