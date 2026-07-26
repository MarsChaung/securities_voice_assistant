from datetime import UTC, datetime

from fastapi.testclient import TestClient

from knowledge_admin.api import create_app
from knowledge_admin.config import KnowledgeAdminSettings
from knowledge_admin.governance import GovernanceAction, GovernanceActor, KnowledgeRole
from knowledge_admin.repository import (
    DatabaseKnowledgeRepository,
    GovernancePayload,
)
from observability import (
    DatabaseShadowReviewRepository,
    ShadowReviewInput,
    ShadowReviewStatus,
)

ORIGIN_HEADERS = {"Origin": "http://testserver"}


def make_client(repository: DatabaseKnowledgeRepository) -> TestClient:
    settings = KnowledgeAdminSettings(
        app_env="development",
        database_url="sqlite+pysqlite:///:memory:",
        knowledge_admin_dev_identity_enabled=True,
    )
    return TestClient(
        create_app(
            repository=repository,
            settings=settings,
            clock=lambda: datetime(2026, 7, 21, tzinfo=UTC),
        )
    )


def actor(actor_id: str, role: KnowledgeRole) -> GovernanceActor:
    return GovernanceActor(actor_id=actor_id, roles=frozenset({role}))


def seed_shadow_result(repository: DatabaseKnowledgeRepository) -> str:
    result = DatabaseShadowReviewRepository(repository.engine).record(
        ShadowReviewInput(
            turn_id="turn-shadow-ui",
            knowledge_id="K-CATHAY-US-003",
            knowledge_version="1.1",
            source_id="SRC-CATHAY-USSTOCK-001",
            standard_answer="美股交易時間會因夏令與冬令時間不同而改變。",
            prohibited_extensions=("不得保證固定成交時間",),
            generated_answer="簡單說，美股交易時段會依夏令與冬令時間調整。",
            generation_model_id="Qwen3.6-35B-A3B-oQ4",
            prompt_version="controlled-answer-v4",
            prompt_hash="a" * 64,
            generation_latency_ms=1250.0,
            output_guard_safe=True,
            fallback_reason="shadow_only",
        ),
        now=datetime(2026, 7, 21, tzinfo=UTC),
    )
    return result.shadow_id


def publish_overdue_item(repository: DatabaseKnowledgeRepository) -> None:
    knowledge_id = "K-CATHAY-DCA-001"
    submitted = repository.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.SUBMIT_REVIEW,
        actor=actor("Codex-assisted draft import", KnowledgeRole.AUTHOR),
        expected_version=1,
        now=datetime(2026, 7, 18, tzinfo=UTC),
    )
    reviewed = repository.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.COMPLETE_REVIEW,
        actor=actor("reviewer.dev", KnowledgeRole.REVIEWER),
        expected_version=submitted.row_version,
        now=datetime(2026, 7, 18, tzinfo=UTC),
    )
    approved = repository.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.APPROVE,
        actor=actor("approver.dev", KnowledgeRole.APPROVER),
        expected_version=reviewed.row_version,
        payload=GovernancePayload(
            effective_at=datetime(2026, 7, 18, tzinfo=UTC),
            review_at=datetime(2026, 7, 20, tzinfo=UTC),
            owner_unit="數位通路處",
        ),
        now=datetime(2026, 7, 18, tzinfo=UTC),
    )
    repository.perform_action(
        knowledge_id=knowledge_id,
        action=GovernanceAction.PUBLISH,
        actor=actor("publisher.dev", KnowledgeRole.PUBLISHER),
        expected_version=approved.row_version,
        now=datetime(2026, 7, 18, tzinfo=UTC),
    )


def test_knowledge_list_contains_drafts(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    response = make_client(knowledge_store).get("/admin/knowledge")

    assert response.status_code == 200
    assert "知識治理中心" in response.text
    assert "本機開發模式" in response.text
    assert "台股定期定額的基本概念" in response.text
    assert "複審到期時間" in response.text
    assert "<strong>15</strong><span>筆</span>" in response.text


def test_knowledge_list_can_filter_by_status(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    client = make_client(knowledge_store)

    drafts = client.get("/admin/knowledge", params={"status": "draft"})
    published = client.get("/admin/knowledge", params={"status": "published"})

    assert drafts.status_code == 200
    assert "<strong>15</strong><span>筆</span>" in drafts.text
    assert published.status_code == 200
    assert "<strong>0</strong><span>筆</span>" in published.text
    assert "台股定期定額的基本概念" not in published.text


def test_empty_status_filter_means_all_statuses(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    response = make_client(knowledge_store).get(
        "/admin/knowledge",
        params={"status": ""},
    )

    assert response.status_code == 200
    assert "<strong>15</strong><span>筆</span>" in response.text
    assert response.text.count('class="item-link"') == 15


def test_knowledge_detail_preserves_source_and_restrictions(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    response = make_client(knowledge_store).get("/admin/knowledge/K-CATHAY-DCA-001")

    assert response.status_code == 200
    assert "台股定期定額的基本概念" in response.text
    assert "常見問題：什麼是台股定期定額" in response.text
    assert "https://istockapp.cathaysec.com.tw/Marketing/DCA/" in response.text
    assert "不得延伸為個人化投資建議" in response.text
    assert "草稿" in response.text
    assert "送交審核" in response.text
    assert "複審到期時間" in response.text
    assert "問句變體" in response.text
    assert "儲存問句變體" in response.text


def test_draft_question_variants_can_be_managed_from_ui(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    client = make_client(knowledge_store)
    created = client.post(
        "/admin/knowledge/K-CATHAY-DCA-001/question-variants",
        data={
            "actor_id": "Codex-assisted draft import",
            "expected_version": "1",
            "new_question_texts": "什麼是台股定期定額？\n每月固定買台股是什麼？",
            "new_variant_usage": "retrieval",
        },
        headers=ORIGIN_HEADERS,
        follow_redirects=False,
    )

    assert created.status_code == 303
    current = knowledge_store.get_item("K-CATHAY-DCA-001")
    assert len(current.item.question_variants) == 2
    first, second = current.item.question_variants

    edited = client.post(
        "/admin/knowledge/K-CATHAY-DCA-001/question-variants",
        data={
            "actor_id": "Codex-assisted draft import",
            "expected_version": str(current.row_version),
            "variant_id": [first.variant_id, second.variant_id],
            "question_text": ["台股定期定額有哪些特色？", second.question_text],
            "variant_usage": ["evaluation_only", "retrieval"],
            "delete_variant_id": second.variant_id,
            "new_question_texts": "",
            "new_variant_usage": "retrieval",
        },
        headers=ORIGIN_HEADERS,
        follow_redirects=False,
    )

    assert edited.status_code == 303
    revised = knowledge_store.get_item("K-CATHAY-DCA-001")
    assert len(revised.item.question_variants) == 1
    assert revised.item.question_variants[0].question_text == "台股定期定額有哪些特色？"
    assert revised.item.question_variants[0].usage.value == "evaluation_only"
    detail = client.get(edited.headers["location"])
    assert "問句變體已儲存" in detail.text
    assert "台股定期定額有哪些特色？" in detail.text
    search = client.get(
        "/admin/knowledge",
        params={"q": "台股定期定額有哪些特色"},
    )
    assert "台股定期定額的基本概念" in search.text


def test_draft_title_and_standard_answer_can_be_edited_from_ui(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    client = make_client(knowledge_store)
    response = client.post(
        "/admin/knowledge/K-CATHAY-DCA-001/content",
        data={
            "actor_id": "Codex-assisted draft import",
            "expected_version": "1",
            "title": "台股定期定額完整說明",
            "standard_answer": "這是一段經人工編輯、等待審核的標準答案。",
        },
        headers=ORIGIN_HEADERS,
        follow_redirects=False,
    )

    assert response.status_code == 303
    current = knowledge_store.get_item("K-CATHAY-DCA-001")
    assert current.item.title == "台股定期定額完整說明"
    assert current.item.standard_answer == "這是一段經人工編輯、等待審核的標準答案。"
    detail = client.get(response.headers["location"])
    assert "標題與標準答案已儲存" in detail.text
    assert "這是一段經人工編輯、等待審核的標準答案。" in detail.text


def test_overdue_published_item_can_start_revision_from_ui(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    publish_overdue_item(knowledge_store)
    client = make_client(knowledge_store)

    item_list = client.get("/admin/knowledge")
    detail = client.get("/admin/knowledge/K-CATHAY-DCA-001")
    published_filter = client.get("/admin/knowledge", params={"status": "published"})
    expired_filter = client.get("/admin/knowledge", params={"status": "expired"})

    assert "2026-07-20 08:00" in item_list.text
    assert "已逾期" in item_list.text
    assert "<span>已發布</span><strong>0</strong>" in item_list.text
    assert "<span>已過期</span><strong>1</strong>" in item_list.text
    assert "status-expired" in item_list.text
    assert "<strong>0</strong><span>筆</span>" in published_filter.text
    assert "台股定期定額的基本概念" not in published_filter.text
    assert "<strong>1</strong><span>筆</span>" in expired_filter.text
    assert "台股定期定額的基本概念" in expired_filter.text
    assert "複審到期時間" in detail.text
    assert "status-expired" in detail.text
    assert "資料庫狀態" in detail.text
    assert "Runtime 狀態" in detail.text
    assert "不可用" in detail.text
    assert "建立複審新版" in detail.text
    assert knowledge_store.get_item("K-CATHAY-DCA-001").item.status.value == "published"

    current = knowledge_store.get_item("K-CATHAY-DCA-001")
    response = client.post(
        "/admin/knowledge/K-CATHAY-DCA-001/actions/start_revision",
        data={
            "actor_id": "Codex-assisted draft import",
            "expected_version": str(current.row_version),
            "reason": "例行複審並展延複審到期時間",
        },
        headers=ORIGIN_HEADERS,
        follow_redirects=False,
    )

    assert response.status_code == 303
    revised = knowledge_store.get_item("K-CATHAY-DCA-001")
    assert revised.item.version == "1.1-draft"
    assert len(knowledge_store.list_versions("K-CATHAY-DCA-001")) == 1
    revised_detail = client.get(response.headers["location"])
    assert "已完成：建立複審新版" in revised_detail.text
    assert "已封存發布版本" in revised_detail.text
    assert "版本 1.0" in revised_detail.text


def test_source_list_contains_four_sources(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    response = make_client(knowledge_store).get("/admin/sources")

    assert response.status_code == 200
    assert "4 筆" in response.text
    assert response.text.count("查看原始頁面") == 4


def test_submit_for_review_updates_persisted_state(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    client = make_client(knowledge_store)
    response = client.post(
        "/admin/knowledge/K-CATHAY-DCA-001/actions/submit_review",
        data={"actor_id": "Codex-assisted draft import", "expected_version": "1"},
        headers=ORIGIN_HEADERS,
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert knowledge_store.get_item("K-CATHAY-DCA-001").item.status.value == "review"
    detail = client.get(response.headers["location"])
    assert "已完成：送交審核" in detail.text
    assert "完成內容審核" in detail.text


def test_cross_origin_mutation_is_rejected(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    response = make_client(knowledge_store).post(
        "/admin/knowledge/K-CATHAY-DCA-001/actions/submit_review",
        data={"actor_id": "Codex-assisted draft import", "expected_version": "1"},
        headers={"Origin": "https://attacker.example"},
    )

    assert response.status_code == 403
    assert knowledge_store.get_item("K-CATHAY-DCA-001").item.status.value == "draft"


def test_governance_reason_rejects_sensitive_data(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    client = make_client(knowledge_store)
    response = client.post(
        "/admin/knowledge/K-CATHAY-DCA-001/actions/submit_review",
        data={
            "actor_id": "Codex-assisted draft import",
            "expected_version": "1",
            "reason": "客戶手機 0912-345-678",
        },
        headers=ORIGIN_HEADERS,
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "error=" in response.headers["location"]
    assert knowledge_store.get_item("K-CATHAY-DCA-001").item.status.value == "draft"


def test_admin_responses_set_security_headers(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    response = make_client(knowledge_store).get("/admin/knowledge")

    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_unknown_knowledge_returns_not_found(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    response = make_client(knowledge_store).get("/admin/knowledge/K-NOT-FOUND")

    assert response.status_code == 404


def test_healthz_checks_database(knowledge_store: DatabaseKnowledgeRepository) -> None:
    response = make_client(knowledge_store).get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "connected",
        "identity_mode": "development",
    }


def test_shadow_workbench_lists_and_compares_results(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    shadow_id = seed_shadow_result(knowledge_store)
    client = make_client(knowledge_store)

    result_list = client.get("/admin/shadow")
    detail = client.get(f"/admin/shadow/{shadow_id}")

    assert result_list.status_code == 200
    assert "Shadow 複核工作台" in result_list.text
    assert "美股一般交易時段" in result_list.text
    assert "待複核" in result_list.text
    assert "1250 ms" in result_list.text
    assert detail.status_code == 200
    assert "核准標準答案" in detail.text
    assert "模型改寫答案" in detail.text
    assert "完成複核" in detail.text
    assert "美股交易時段會依夏令與冬令時間調整" in detail.text
    assert "原始問句" not in detail.text


def test_shadow_review_submission_is_persisted(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    shadow_id = seed_shadow_result(knowledge_store)
    client = make_client(knowledge_store)

    response = client.post(
        f"/admin/shadow/{shadow_id}/review",
        data={
            "actor_id": "reviewer.dev",
            "label": "acceptable",
            "expected_version": "1",
            "note": "語意完整且沒有新增內容",
        },
        headers=ORIGIN_HEADERS,
        follow_redirects=False,
    )

    assert response.status_code == 303
    reviewed = DatabaseShadowReviewRepository(knowledge_store.engine).get_result(shadow_id)
    assert reviewed.review_status is ShadowReviewStatus.ACCEPTED
    detail = client.get(response.headers["location"])
    assert "Shadow 複核已完成" in detail.text
    assert "人工複核結果" in detail.text
    assert "語意完整且沒有新增內容" in detail.text


def test_shadow_review_note_rejects_sensitive_data(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    shadow_id = seed_shadow_result(knowledge_store)
    client = make_client(knowledge_store)

    response = client.post(
        f"/admin/shadow/{shadow_id}/review",
        data={
            "actor_id": "reviewer.dev",
            "label": "other",
            "expected_version": "1",
            "note": "客戶手機 0912-345-678",
        },
        headers=ORIGIN_HEADERS,
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "error=" in response.headers["location"]
    result = DatabaseShadowReviewRepository(knowledge_store.engine).get_result(shadow_id)
    assert result.review_status is ShadowReviewStatus.PENDING
