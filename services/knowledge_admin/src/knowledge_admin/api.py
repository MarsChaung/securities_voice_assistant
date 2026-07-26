from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import FormData

from observability import (
    DatabaseShadowReviewRepository,
    ShadowReviewConcurrentUpdateError,
    ShadowReviewLabel,
    ShadowReviewNotFoundError,
    ShadowReviewStateError,
    ShadowReviewStatus,
)
from policy import SensitiveDataGuard
from retrieval import KnowledgeItem, KnowledgeStatus, QuestionVariantUsage

from .config import KnowledgeAdminSettings
from .faq_import import (
    MAX_XLSX_BYTES,
    FaqImportError,
    FaqImportNotFoundError,
    FaqImportRepository,
    FaqImportStatus,
    FaqXlsxParser,
)
from .governance import GovernanceAction, GovernanceError, KnowledgeRole
from .identity import DevelopmentIdentityProvider
from .repository import (
    ConcurrentUpdateError,
    DatabaseKnowledgeRepository,
    GovernancePayload,
    KnowledgeNotFoundError,
    QuestionVariantInput,
)

PACKAGE_ROOT = Path(__file__).parent
TAIPEI = ZoneInfo("Asia/Taipei")


def create_app(
    *,
    repository: DatabaseKnowledgeRepository | None = None,
    settings: KnowledgeAdminSettings | None = None,
    identity_provider: DevelopmentIdentityProvider | None = None,
    shadow_repository: DatabaseShadowReviewRepository | None = None,
    clock: Callable[[], datetime] | None = None,
) -> FastAPI:
    resolved_settings = settings or KnowledgeAdminSettings()
    resolved_settings.validate_identity_mode()
    knowledge_repository = repository or DatabaseKnowledgeRepository.from_url(
        resolved_settings.database_url
    )
    faq_imports = FaqImportRepository(knowledge_repository.engine)
    shadow_reviews = shadow_repository or (
        DatabaseShadowReviewRepository(knowledge_repository.engine)
        if repository is not None
        else DatabaseShadowReviewRepository.from_url(resolved_settings.database_url)
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
        shadow_status_labels=_SHADOW_STATUS_LABELS,
        shadow_label_labels=_SHADOW_LABEL_LABELS,
        question_variant_usage_labels=_QUESTION_VARIANT_USAGE_LABELS,
        faq_import_status_labels=_FAQ_IMPORT_STATUS_LABELS,
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
            shadow_reviews.check_connection()
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
        if action in {
            GovernanceAction.UPDATE_CONTENT,
            GovernanceAction.UPDATE_QUESTION_VARIANTS,
        }:
            raise HTTPException(status_code=404, detail="不支援的治理操作")

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

    @app.post("/admin/knowledge/{knowledge_id}/content")
    def update_content(
        knowledge_id: str,
        actor_id: Annotated[str, Form()],
        expected_version: Annotated[int, Form()],
        title: Annotated[str, Form(min_length=1, max_length=200)],
        standard_answer: Annotated[str, Form(min_length=1, max_length=10000)],
    ) -> RedirectResponse:
        if not resolved_settings.knowledge_admin_dev_identity_enabled:
            raise HTTPException(status_code=403, detail="開發身分模式未啟用")
        try:
            actor = identities.get_actor(actor_id)
            knowledge_repository.update_content(
                knowledge_id=knowledge_id,
                title=title,
                standard_answer=standard_answer,
                actor=actor,
                expected_version=expected_version,
                now=resolved_clock(),
            )
        except (
            ConcurrentUpdateError,
            GovernanceError,
            KnowledgeNotFoundError,
            ValueError,
        ) as exc:
            return _detail_redirect(knowledge_id, error=str(exc))
        return _detail_redirect(knowledge_id, notice="標題與標準答案已儲存")

    @app.post("/admin/knowledge/{knowledge_id}/question-variants")
    async def update_question_variants(
        request: Request,
        knowledge_id: str,
    ) -> RedirectResponse:
        if not resolved_settings.knowledge_admin_dev_identity_enabled:
            raise HTTPException(status_code=403, detail="開發身分模式未啟用")

        try:
            form = await request.form()
            actor = identities.get_actor(_form_string(form, "actor_id"))
            expected_version = int(_form_string(form, "expected_version"))
            variants = _question_variant_inputs(form)
            if len(variants) > 200:
                raise ValueError("單一知識項目最多可保存 200 筆問句變體")
            guard = SensitiveDataGuard()
            if any(
                guard.scan(variant.question_text).has_sensitive_data
                for variant in variants
            ):
                raise ValueError("問句變體不得包含個資、帳號、密碼或驗證碼")
            knowledge_repository.update_question_variants(
                knowledge_id=knowledge_id,
                variants=variants,
                actor=actor,
                expected_version=expected_version,
                now=resolved_clock(),
            )
        except (
            ConcurrentUpdateError,
            GovernanceError,
            KnowledgeNotFoundError,
            ValueError,
        ) as exc:
            return _detail_redirect(knowledge_id, error=str(exc))

        return _detail_redirect(knowledge_id, notice="問句變體已儲存")

    @app.get("/admin/faq-imports", response_class=HTMLResponse)
    def faq_import_list(
        request: Request,
        notice: str | None = None,
        error: str | None = None,
    ) -> Response:
        return templates.TemplateResponse(
            request=request,
            name="faq_import_list.html",
            context={
                "batches": faq_imports.list_batches(),
                "authors": identities.actors_for(KnowledgeRole.AUTHOR),
                "notice": notice,
                "error": error,
            },
        )

    @app.post("/admin/faq-imports/preview")
    async def create_faq_import_preview(
        actor_id: Annotated[str, Form()],
        dataset_title: Annotated[str, Form(min_length=1, max_length=200)],
        publisher: Annotated[str, Form(min_length=1, max_length=200)],
        workbook_file: Annotated[UploadFile, File()],
        source_type: Annotated[
            Literal["local_import", "approved_internal_faq"],
            Form(),
        ] = "local_import",
        source_url: Annotated[str, Form(max_length=2000)] = "",
    ) -> RedirectResponse:
        if not resolved_settings.knowledge_admin_dev_identity_enabled:
            raise HTTPException(status_code=403, detail="開發身分模式未啟用")
        try:
            actor = identities.get_actor(actor_id)
            if KnowledgeRole.AUTHOR not in actor.roles:
                raise FaqImportError("FAQ 匯入預覽必須使用作者身分")
            filename = workbook_file.filename or ""
            content = await workbook_file.read(MAX_XLSX_BYTES + 1)
            workbook = FaqXlsxParser().parse(content)
            batch = faq_imports.create_preview(
                original_filename=filename,
                dataset_title=dataset_title,
                publisher=publisher,
                source_type=source_type,
                source_url=source_url.strip() or None,
                uploaded_by=actor.actor_id,
                workbook=workbook,
                now=resolved_clock(),
            )
        except (FaqImportError, ValueError) as exc:
            return _faq_import_list_redirect(error=str(exc))
        finally:
            await workbook_file.close()
        return _faq_import_detail_redirect(batch.batch_id, notice="FAQ 匯入預覽已建立")

    @app.get("/admin/faq-imports/{batch_id}", response_class=HTMLResponse)
    def faq_import_detail(
        request: Request,
        batch_id: str,
        notice: str | None = None,
        error: str | None = None,
    ) -> Response:
        try:
            batch = faq_imports.get_batch(batch_id)
        except FaqImportNotFoundError as exc:
            raise HTTPException(status_code=404, detail="找不到 FAQ 匯入批次") from exc
        return templates.TemplateResponse(
            request=request,
            name="faq_import_detail.html",
            context={
                "batch": batch,
                "authors": identities.actors_for(KnowledgeRole.AUTHOR),
                "notice": notice,
                "error": error,
            },
        )

    @app.post("/admin/faq-imports/{batch_id}/commit")
    async def commit_faq_import(request: Request, batch_id: str) -> RedirectResponse:
        if not resolved_settings.knowledge_admin_dev_identity_enabled:
            raise HTTPException(status_code=403, detail="開發身分模式未啟用")
        try:
            form = await request.form()
            actor = identities.get_actor(_form_string(form, "actor_id"))
            expected_version = int(_form_string(form, "expected_version"))
            selected_row_ids = tuple(_form_strings(form, "selected_row_id"))
            knowledge_ids = faq_imports.import_drafts(
                batch_id=batch_id,
                selected_row_ids=selected_row_ids,
                actor=actor,
                expected_version=expected_version,
                now=resolved_clock(),
            )
        except (
            FaqImportError,
            FaqImportNotFoundError,
            ValueError,
        ) as exc:
            return _faq_import_detail_redirect(batch_id, error=str(exc))
        return _faq_import_detail_redirect(
            batch_id,
            notice=f"已建立 {len(knowledge_ids)} 筆 FAQ 知識草稿",
        )

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

    @app.get("/admin/shadow", response_class=HTMLResponse)
    def shadow_list(
        request: Request,
        status: Annotated[str | None, Query(max_length=30)] = None,
        knowledge_id: Annotated[str | None, Query(max_length=80)] = None,
    ) -> Response:
        selected_status = _parse_shadow_status_filter(status)
        results = shadow_reviews.list_results(
            status=selected_status,
            knowledge_id=knowledge_id.strip() if knowledge_id else None,
        )
        titles = {
            record.item.knowledge_id: record.item.title
            for record in knowledge_repository.list_items()
        }
        return templates.TemplateResponse(
            request=request,
            name="shadow_list.html",
            context={
                "results": results,
                "metrics": shadow_reviews.metrics(),
                "titles": titles,
                "selected_status": selected_status,
                "knowledge_id": knowledge_id or "",
            },
        )

    @app.get("/admin/shadow/{shadow_id}", response_class=HTMLResponse)
    def shadow_detail(
        request: Request,
        shadow_id: str,
        notice: str | None = None,
        error: str | None = None,
    ) -> Response:
        try:
            result = shadow_reviews.get_result(shadow_id)
            item = knowledge_repository.get_item(result.knowledge_id).item
            source = knowledge_repository.get_source(result.source_id)
        except (KnowledgeNotFoundError, ShadowReviewNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="找不到 Shadow 複核結果") from exc
        return templates.TemplateResponse(
            request=request,
            name="shadow_detail.html",
            context={
                "result": result,
                "item": item,
                "source": source,
                "reviewers": identities.actors_for(KnowledgeRole.REVIEWER),
                "notice": notice,
                "error": error,
            },
        )

    @app.post("/admin/shadow/{shadow_id}/review")
    def review_shadow_result(
        shadow_id: str,
        actor_id: Annotated[str, Form()],
        label: Annotated[ShadowReviewLabel, Form()],
        expected_version: Annotated[int, Form()],
        note: Annotated[str, Form(max_length=1000)] = "",
    ) -> RedirectResponse:
        if not resolved_settings.knowledge_admin_dev_identity_enabled:
            raise HTTPException(status_code=403, detail="開發身分模式未啟用")
        try:
            actor = identities.get_actor(actor_id)
            if KnowledgeRole.REVIEWER not in actor.roles:
                raise ValueError("Shadow 複核必須使用審核身分")
            normalized_note = note.strip() or None
            if (
                normalized_note
                and SensitiveDataGuard().scan(normalized_note).has_sensitive_data
            ):
                raise ValueError("複核說明不得包含個資、帳號、密碼或驗證碼")
            shadow_reviews.review(
                shadow_id=shadow_id,
                label=label,
                reviewer_id=actor.actor_id,
                reviewer_note=normalized_note,
                expected_version=expected_version,
                now=resolved_clock(),
            )
        except (
            ShadowReviewConcurrentUpdateError,
            ShadowReviewNotFoundError,
            ShadowReviewStateError,
            ValueError,
        ) as exc:
            return _shadow_detail_redirect(shadow_id, error=str(exc))
        return _shadow_detail_redirect(shadow_id, notice="Shadow 複核已完成")

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


def _shadow_detail_redirect(
    shadow_id: str,
    *,
    notice: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    messages = {"notice": notice, "error": error}
    query = urlencode({key: value for key, value in messages.items() if value})
    return RedirectResponse(url=f"/admin/shadow/{shadow_id}?{query}", status_code=303)


def _faq_import_list_redirect(
    *,
    notice: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    messages = {"notice": notice, "error": error}
    query = urlencode({key: value for key, value in messages.items() if value})
    return RedirectResponse(url=f"/admin/faq-imports?{query}", status_code=303)


def _faq_import_detail_redirect(
    batch_id: str,
    *,
    notice: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    messages = {"notice": notice, "error": error}
    query = urlencode({key: value for key, value in messages.items() if value})
    return RedirectResponse(
        url=f"/admin/faq-imports/{batch_id}?{query}",
        status_code=303,
    )


def _parse_local_datetime(value: str) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=TAIPEI)


def _question_variant_inputs(form: FormData) -> tuple[QuestionVariantInput, ...]:
    variant_ids = _form_strings(form, "variant_id")
    question_texts = _form_strings(form, "question_text")
    usages = _form_strings(form, "variant_usage")
    if not (len(variant_ids) == len(question_texts) == len(usages)):
        raise ValueError("問句變體表單格式不正確")

    deleted_ids = set(_form_strings(form, "delete_variant_id"))
    variants = [
        QuestionVariantInput(
            variant_id=variant_id,
            question_text=question_text,
            usage=QuestionVariantUsage(usage),
        )
        for variant_id, question_text, usage in zip(
            variant_ids,
            question_texts,
            usages,
            strict=True,
        )
        if variant_id not in deleted_ids
    ]

    new_usage = QuestionVariantUsage(
        _form_string(
            form,
            "new_variant_usage",
            default=QuestionVariantUsage.RETRIEVAL.value,
        )
    )
    new_questions = _form_string(form, "new_question_texts", default="")
    variants.extend(
        QuestionVariantInput(
            variant_id=None,
            question_text=line,
            usage=new_usage,
        )
        for line in (line.strip() for line in new_questions.splitlines())
        if line
    )
    return tuple(variants)


def _form_strings(form: FormData, key: str) -> list[str]:
    values = form.getlist(key)
    if any(not isinstance(value, str) for value in values):
        raise ValueError("問句變體表單格式不正確")
    return [value for value in values if isinstance(value, str)]


def _form_string(form: FormData, key: str, *, default: str | None = None) -> str:
    value = form.get(key, default)
    if not isinstance(value, str):
        raise ValueError("問句變體表單格式不正確")
    return value


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


def _parse_shadow_status_filter(value: str | None) -> ShadowReviewStatus | None:
    if value is None or not value.strip():
        return None
    try:
        return ShadowReviewStatus(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="不支援的 Shadow 複核狀態篩選") from exc


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
            or any(
                normalized_query in variant.question_text.casefold()
                for variant in item.question_variants
            )
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
    GovernanceAction.UPDATE_CONTENT: "更新標題與標準答案",
    GovernanceAction.UPDATE_QUESTION_VARIANTS: "更新問句變體",
    GovernanceAction.START_REVISION: "建立複審新版",
    GovernanceAction.SUBMIT_REVIEW: "送交審核",
    GovernanceAction.COMPLETE_REVIEW: "完成審核",
    GovernanceAction.APPROVE: "核准",
    GovernanceAction.PUBLISH: "發布",
    GovernanceAction.RETURN_DRAFT: "退回草稿",
    GovernanceAction.REVOKE: "撤銷",
}

_QUESTION_VARIANT_USAGE_LABELS = {
    QuestionVariantUsage.RETRIEVAL: "正式檢索",
    QuestionVariantUsage.EVALUATION_ONLY: "僅供評測",
    QuestionVariantUsage.EXCLUDED: "排除",
}

_SHADOW_STATUS_LABELS = {
    ShadowReviewStatus.PENDING: "待複核",
    ShadowReviewStatus.ACCEPTED: "可接受",
    ShadowReviewStatus.REJECTED: "不採用",
    ShadowReviewStatus.NOT_REVIEWABLE: "無法產生",
}

_SHADOW_LABEL_LABELS = {
    ShadowReviewLabel.ACCEPTABLE: "可接受",
    ShadowReviewLabel.INCORRECT: "內容不正確",
    ShadowReviewLabel.UNSUPPORTED_EXTENSION: "超出核准內容",
    ShadowReviewLabel.MISSING_QUALIFIER: "遺漏限制或警語",
    ShadowReviewLabel.UNSAFE: "包含安全風險",
    ShadowReviewLabel.TONE: "語氣或表達不佳",
    ShadowReviewLabel.OTHER: "其他問題",
}

_FAQ_IMPORT_STATUS_LABELS = {
    FaqImportStatus.PREVIEW: "待確認",
    FaqImportStatus.IMPORTED: "已建立草稿",
}


app = create_app()
