import html
import io
import zipfile
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from knowledge_admin.api import create_app
from knowledge_admin.config import KnowledgeAdminSettings
from knowledge_admin.faq_import import (
    FaqImportError,
    FaqImportRepository,
    FaqImportStatus,
    FaqXlsxParser,
)
from knowledge_admin.governance import GovernanceActor, KnowledgeRole
from knowledge_admin.repository import (
    DatabaseKnowledgeRepository,
    QuestionVariantInput,
)
from retrieval import KnowledgeStatus, QuestionVariantUsage

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
            clock=lambda: datetime(2026, 7, 23, tzinfo=UTC),
        )
    )


def test_parser_groups_answer_with_questions_and_flags_issues() -> None:
    parsed = FaqXlsxParser().parse(_faq_workbook())

    assert parsed.sheet_name == "工作表 1"
    assert len(parsed.rows) == 3
    first, second, invalid = parsed.rows
    assert first.source_item_no == "1"
    assert first.standard_answer == "請至官網完成申請。"
    assert first.questions == ("如何申請？", "我忘記密碼怎麼辦？")
    assert any("POL-REFUSE-004" in warning for warning in first.warnings)
    assert any("其他知識項目" in warning for warning in first.warnings)
    assert any("其他知識項目" in warning for warning in second.warnings)
    assert invalid.is_valid is False
    assert invalid.errors == ("標準答案不可為空",)


def test_parser_rejects_non_xlsx_content() -> None:
    with pytest.raises(FaqImportError, match="有效的 .xlsx"):
        FaqXlsxParser().parse(b"not an Excel workbook")


def test_ui_previews_then_imports_selected_rows_as_drafts(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    client = make_client(knowledge_store)
    preview_response = client.post(
        "/admin/faq-imports/preview",
        data={
            "actor_id": "Codex-assisted draft import",
            "dataset_title": "授權 FAQ 測試集",
            "publisher": "客戶服務處",
            "source_type": "local_import",
            "source_url": "",
        },
        files={
            "workbook_file": (
                "faq.xlsx",
                _faq_workbook(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        headers=ORIGIN_HEADERS,
        follow_redirects=False,
    )

    assert preview_response.status_code == 303
    assert "/admin/faq-imports/" in preview_response.headers["location"]
    preview_page = client.get(preview_response.headers["location"])
    assert preview_page.status_code == 200
    assert "授權 FAQ 測試集" in preview_page.text
    assert "3" in preview_page.text
    assert "2" in preview_page.text
    assert "標準答案不可為空" in preview_page.text
    assert "確認所選項目並建立草稿" in preview_page.text

    batches = FaqImportRepository(knowledge_store.engine).list_batches()
    assert len(batches) == 1
    batch = batches[0]
    assert batch.status is FaqImportStatus.PREVIEW
    valid_rows = tuple(row for row in batch.rows if row.is_valid)

    commit_response = client.post(
        f"/admin/faq-imports/{batch.batch_id}/commit",
        data={
            "actor_id": "Codex-assisted draft import",
            "expected_version": str(batch.row_version),
            "selected_row_id": [row.row_id for row in valid_rows],
        },
        headers=ORIGIN_HEADERS,
        follow_redirects=False,
    )

    assert commit_response.status_code == 303
    committed = FaqImportRepository(knowledge_store.engine).get_batch(batch.batch_id)
    assert committed.status is FaqImportStatus.IMPORTED
    assert committed.imported_count == 2
    assert sum(row.imported for row in committed.rows) == 2

    first_item = knowledge_store.get_item(valid_rows[0].proposed_knowledge_id).item
    assert first_item.status is KnowledgeStatus.DRAFT
    assert first_item.public_answer_allowed is False
    assert first_item.source_type == "local_import"
    assert first_item.source_uri is None
    assert first_item.author == "Codex-assisted draft import"
    assert len(first_item.question_variants) == 2
    assert all(
        variant.usage is QuestionVariantUsage.RETRIEVAL for variant in first_item.question_variants
    )
    assert len(knowledge_store.list_items()) == 17

    result_page = client.get(commit_response.headers["location"])
    assert "已建立 2 筆 FAQ 知識草稿" in result_page.text
    assert f"/admin/knowledge/{valid_rows[0].proposed_knowledge_id}" in result_page.text
    assert f"/admin/knowledge/{committed.rows[2].proposed_knowledge_id}" not in result_page.text


def test_same_file_and_source_reuses_existing_preview(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    parsed = FaqXlsxParser().parse(_faq_workbook())
    imports = FaqImportRepository(knowledge_store.engine)
    first = imports.create_preview(
        original_filename="faq.xlsx",
        dataset_title="授權 FAQ 測試集",
        publisher="客戶服務處",
        source_url="https://intranet.example.com/faq/approved-test",
        uploaded_by="Codex-assisted draft import",
        workbook=parsed,
        now=datetime(2026, 7, 23, tzinfo=UTC),
    )
    second = imports.create_preview(
        original_filename="faq.xlsx",
        dataset_title="授權 FAQ 測試集",
        publisher="客戶服務處",
        source_url="https://intranet.example.com/faq/approved-test",
        uploaded_by="Codex-assisted draft import",
        workbook=parsed,
        now=datetime(2026, 7, 23, tzinfo=UTC),
    )

    assert second.batch_id == first.batch_id
    assert len(imports.list_batches()) == 1

    imports.import_drafts(
        batch_id=first.batch_id,
        selected_row_ids=tuple(row.row_id for row in first.rows if row.is_actionable),
        actor=GovernanceActor(
            actor_id="Codex-assisted draft import",
            roles=frozenset({KnowledgeRole.AUTHOR}),
        ),
        expected_version=first.row_version,
        now=datetime(2026, 7, 23, tzinfo=UTC),
    )
    repeated = imports.create_preview(
        original_filename="faq.xlsx",
        dataset_title="授權 FAQ 測試集",
        publisher="客戶服務處",
        source_url="https://intranet.example.com/faq/approved-test",
        uploaded_by="Codex-assisted draft import",
        workbook=parsed,
        now=datetime(2026, 7, 24, tzinfo=UTC),
    )

    assert repeated.batch_id == first.batch_id
    assert repeated.status is FaqImportStatus.IMPORTED
    with pytest.raises(FaqImportError, match="已完成匯入"):
        imports.import_drafts(
            batch_id=repeated.batch_id,
            selected_row_ids=(repeated.rows[0].row_id,),
            actor=GovernanceActor(
                actor_id="Codex-assisted draft import",
                roles=frozenset({KnowledgeRole.AUTHOR}),
            ),
            expected_version=repeated.row_version,
            now=datetime(2026, 7, 24, tzinfo=UTC),
        )


def test_approved_internal_faq_requires_https_source_url(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    imports = FaqImportRepository(knowledge_store.engine)

    with pytest.raises(FaqImportError, match="必須提供正式 HTTPS 網址"):
        imports.create_preview(
            original_filename="faq.xlsx",
            dataset_title="授權 FAQ 測試集",
            publisher="客戶服務處",
            source_type="approved_internal_faq",
            source_url=None,
            uploaded_by="Codex-assisted draft import",
            workbook=FaqXlsxParser().parse(_faq_workbook()),
            now=datetime(2026, 7, 23, tzinfo=UTC),
        )


def test_different_file_skips_standard_answers_already_imported(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    imports = FaqImportRepository(knowledge_store.engine)
    first = imports.create_preview(
        original_filename="faq-v1.xlsx",
        dataset_title="授權 FAQ 測試集",
        publisher="客戶服務處",
        source_url="https://intranet.example.com/faq/duplicate-test",
        uploaded_by="Codex-assisted draft import",
        workbook=FaqXlsxParser().parse(_faq_workbook(marker="v1")),
        now=datetime(2026, 7, 23, tzinfo=UTC),
    )
    imports.import_drafts(
        batch_id=first.batch_id,
        selected_row_ids=tuple(row.row_id for row in first.rows if row.is_actionable),
        actor=GovernanceActor(
            actor_id="Codex-assisted draft import",
            roles=frozenset({KnowledgeRole.AUTHOR}),
        ),
        expected_version=first.row_version,
        now=datetime(2026, 7, 23, tzinfo=UTC),
    )

    second = imports.create_preview(
        original_filename="faq-v2.xlsx",
        dataset_title="授權 FAQ 測試集 V2",
        publisher="客戶服務處",
        source_url="https://intranet.example.com/faq/duplicate-test",
        uploaded_by="Codex-assisted draft import",
        workbook=FaqXlsxParser().parse(_faq_workbook(marker="v2")),
        now=datetime(2026, 7, 24, tzinfo=UTC),
    )

    duplicates = tuple(row for row in second.rows if row.duplicate_knowledge_ids)
    assert len(duplicates) == 2
    assert second.valid_row_count == 0
    assert all(not row.is_actionable for row in duplicates)
    assert all("不會再建立知識草稿" in row.warnings[-1] for row in duplicates)

    with pytest.raises(FaqImportError, match="不可匯入"):
        imports.import_drafts(
            batch_id=second.batch_id,
            selected_row_ids=(duplicates[0].row_id,),
            actor=GovernanceActor(
                actor_id="Codex-assisted draft import",
                roles=frozenset({KnowledgeRole.AUTHOR}),
            ),
            expected_version=second.row_version,
            now=datetime(2026, 7, 24, tzinfo=UTC),
        )


def test_preview_flags_question_already_used_by_existing_knowledge(
    knowledge_store: DatabaseKnowledgeRepository,
) -> None:
    knowledge_store.update_question_variants(
        knowledge_id="K-CATHAY-DCA-001",
        variants=(
            QuestionVariantInput(
                variant_id=None,
                question_text="如何申請?",
                usage=QuestionVariantUsage.RETRIEVAL,
            ),
        ),
        actor=GovernanceActor(
            actor_id="Codex-assisted draft import",
            roles=frozenset({KnowledgeRole.AUTHOR}),
        ),
        expected_version=1,
        now=datetime(2026, 7, 23, tzinfo=UTC),
    )
    imports = FaqImportRepository(knowledge_store.engine)

    preview = imports.create_preview(
        original_filename="faq.xlsx",
        dataset_title="授權 FAQ 測試集",
        publisher="客戶服務處",
        source_url="https://intranet.example.com/faq/conflict-test",
        uploaded_by="Codex-assisted draft import",
        workbook=FaqXlsxParser().parse(_faq_workbook()),
        now=datetime(2026, 7, 23, tzinfo=UTC),
    )

    assert any("K-CATHAY-DCA-001" in warning for warning in preview.rows[0].warnings)


def _faq_workbook(*, marker: str = "") -> bytes:
    rows = {
        1: {"A": f"FAQ 匯入測試 {marker}"},
        3: {
            "B": "項次",
            "C": "回答內容",
            "D": "問題1",
            "E": "問題2",
            "F": "問題3",
        },
        4: {
            "B": "1",
            "C": "請至官網完成申請。",
            "D": "如何申請？",
            "E": "如何申請？",
            "F": "我忘記密碼怎麼辦？",
        },
        5: {
            "B": "2",
            "C": "可先確認申請資格與應備文件。",
            "D": "如何申請？",
            "E": "申請要帶什麼？",
        },
        6: {"B": "3", "C": "", "D": "沒有答案可以問嗎？"},
    }
    cells = []
    for row_number, row in rows.items():
        for column, value in row.items():
            escaped = html.escape(value)
            cells.append(f'<c r="{column}{row_number}" t="inlineStr"><is><t>{escaped}</t></is></c>')
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(cells)}</sheetData></worksheet>"
    )
    workbook_xml = """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="工作表 1" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""
    relationships_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    return output.getvalue()
