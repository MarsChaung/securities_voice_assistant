import hashlib
import io
import posixpath
import re
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal, cast
from uuid import uuid4
from xml.etree import ElementTree

from sqlalchemy import Engine, select
from sqlalchemy.orm import sessionmaker

from policy import DomainPolicyEngine, PolicyAction, SensitiveDataGuard
from retrieval import (
    KnowledgeItem,
    KnowledgeSource,
    KnowledgeStatus,
    QuestionVariant,
    QuestionVariantUsage,
    SourceStatus,
)

from .database import (
    FaqImportBatchRecord,
    KnowledgeItemRecord,
    KnowledgeQuestionVariantRecord,
    KnowledgeSourceRecord,
)
from .governance import GovernanceActor, KnowledgeRole

MAX_XLSX_BYTES = 10 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_ZIP_MEMBERS = 2000
MAX_IMPORT_ROWS = 2000
MAX_QUESTIONS_PER_ITEM = 200

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CELL_REF = re.compile(r"^([A-Z]+)(\d+)$")
_QUESTION_HEADER = re.compile(r"^問題(?:\d+)?$")
_NORMALIZE_QUESTION = re.compile(r"[^0-9a-z\u3400-\u9fff]")


class FaqImportStatus(StrEnum):
    PREVIEW = "preview"
    IMPORTED = "imported"


class FaqImportError(ValueError):
    pass


class FaqImportNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class ParsedFaqRow:
    row_id: str
    sheet_row: int
    source_item_no: str
    proposed_knowledge_id: str
    title: str
    standard_answer: str
    questions: tuple[str, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    duplicate_knowledge_ids: tuple[str, ...] = ()
    imported: bool = False

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def is_actionable(self) -> bool:
        return self.is_valid and not self.duplicate_knowledge_ids

    def to_json(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "sheet_row": self.sheet_row,
            "source_item_no": self.source_item_no,
            "proposed_knowledge_id": self.proposed_knowledge_id,
            "title": self.title,
            "standard_answer": self.standard_answer,
            "questions": list(self.questions),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "duplicate_knowledge_ids": list(self.duplicate_knowledge_ids),
            "imported": self.imported,
        }

    @classmethod
    def from_json(cls, value: dict[str, object]) -> "ParsedFaqRow":
        return cls(
            row_id=str(value["row_id"]),
            sheet_row=_json_int(value["sheet_row"]),
            source_item_no=str(value["source_item_no"]),
            proposed_knowledge_id=str(value["proposed_knowledge_id"]),
            title=str(value["title"]),
            standard_answer=str(value["standard_answer"]),
            questions=tuple(str(item) for item in _json_list(value["questions"])),
            warnings=tuple(str(item) for item in _json_list(value["warnings"])),
            errors=tuple(str(item) for item in _json_list(value["errors"])),
            duplicate_knowledge_ids=tuple(
                str(item) for item in _json_list(value.get("duplicate_knowledge_ids", []))
            ),
            imported=bool(value.get("imported", False)),
        )


@dataclass(frozen=True)
class ParsedFaqWorkbook:
    sheet_name: str
    rows: tuple[ParsedFaqRow, ...]
    file_sha256: str


@dataclass(frozen=True)
class FaqImportBatch:
    batch_id: str
    original_filename: str
    file_sha256: str
    dataset_title: str
    publisher: str
    source_url: str | None
    source_id: str
    source_type: Literal["local_import", "approved_internal_faq"]
    uploaded_by: str
    status: FaqImportStatus
    sheet_name: str
    rows: tuple[ParsedFaqRow, ...]
    row_count: int
    valid_row_count: int
    imported_count: int
    row_version: int
    created_at: datetime
    imported_at: datetime | None


class FaqXlsxParser:
    def parse(self, content: bytes) -> ParsedFaqWorkbook:
        if not content:
            raise FaqImportError("Excel 檔案不可為空")
        if len(content) > MAX_XLSX_BYTES:
            raise FaqImportError("Excel 檔案不可超過 10 MB")

        file_sha256 = hashlib.sha256(content).hexdigest()
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                self._validate_archive(archive)
                shared_strings = self._shared_strings(archive)
                for sheet_name, sheet_path in self._sheet_paths(archive):
                    matrix = self._read_sheet(archive, sheet_path, shared_strings)
                    header = self._find_header(matrix)
                    if header is not None:
                        rows = self._parse_rows(
                            matrix,
                            header=header,
                            file_sha256=file_sha256,
                        )
                        if not rows:
                            raise FaqImportError("找到 FAQ 欄位，但沒有可預覽的資料列")
                        return ParsedFaqWorkbook(
                            sheet_name=sheet_name,
                            rows=rows,
                            file_sha256=file_sha256,
                        )
        except zipfile.BadZipFile as exc:
            raise FaqImportError("檔案不是有效的 .xlsx Excel 活頁簿") from exc
        except (ElementTree.ParseError, KeyError, ValueError) as exc:
            raise FaqImportError("Excel 結構無法解析") from exc
        raise FaqImportError("找不到「項次／回答內容／問題1…問題N」欄位")

    @staticmethod
    def _validate_archive(archive: zipfile.ZipFile) -> None:
        members = archive.infolist()
        if len(members) > MAX_ZIP_MEMBERS:
            raise FaqImportError("Excel 壓縮內容項目過多")
        if sum(member.file_size for member in members) > MAX_UNCOMPRESSED_BYTES:
            raise FaqImportError("Excel 解壓縮後內容不可超過 50 MB")
        required = {"xl/workbook.xml", "xl/_rels/workbook.xml.rels"}
        if not required.issubset(member.filename for member in members):
            raise FaqImportError("Excel 缺少必要的活頁簿結構")

    @staticmethod
    def _shared_strings(archive: zipfile.ZipFile) -> tuple[str, ...]:
        if "xl/sharedStrings.xml" not in archive.namelist():
            return ()
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        return tuple(
            "".join(node.text or "" for node in item.iter(f"{{{_MAIN_NS}}}t"))
            for item in root.findall(f"{{{_MAIN_NS}}}si")
        )

    @staticmethod
    def _sheet_paths(archive: zipfile.ZipFile) -> tuple[tuple[str, str], ...]:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            relationship.attrib["Id"]: relationship.attrib["Target"]
            for relationship in relationships.findall(f"{{{_PACKAGE_REL_NS}}}Relationship")
        }
        resolved: list[tuple[str, str]] = []
        for sheet in workbook.findall(f".//{{{_MAIN_NS}}}sheet"):
            relationship_id = sheet.attrib[f"{{{_REL_NS}}}id"]
            target = targets[relationship_id].lstrip("/")
            path = (
                posixpath.normpath(target)
                if target.startswith("xl/")
                else posixpath.normpath(posixpath.join("xl", target))
            )
            if PurePosixPath(path).is_absolute() or path.startswith("../"):
                raise FaqImportError("Excel 工作表路徑不合法")
            resolved.append((sheet.attrib["name"], path))
        return tuple(resolved)

    @staticmethod
    def _read_sheet(
        archive: zipfile.ZipFile,
        sheet_path: str,
        shared_strings: tuple[str, ...],
    ) -> dict[int, dict[int, str]]:
        root = ElementTree.fromstring(archive.read(sheet_path))
        matrix: dict[int, dict[int, str]] = {}
        for cell in root.findall(f".//{{{_MAIN_NS}}}c"):
            match = _CELL_REF.fullmatch(cell.attrib.get("r", ""))
            if match is None:
                continue
            column_letters, row_text = match.groups()
            row = int(row_text)
            column = _column_number(column_letters)
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                value = "".join(node.text or "" for node in cell.iter(f"{{{_MAIN_NS}}}t"))
            else:
                value_node = cell.find(f"{{{_MAIN_NS}}}v")
                value = value_node.text if value_node is not None and value_node.text else ""
                if cell_type == "s" and value:
                    value = shared_strings[int(value)]
            matrix.setdefault(row, {})[column] = value
        return matrix

    @staticmethod
    def _find_header(
        matrix: dict[int, dict[int, str]],
    ) -> tuple[int, int, int, tuple[int, ...]] | None:
        for row_number in sorted(matrix):
            values = {
                column: _normalize_header(value) for column, value in matrix[row_number].items()
            }
            item_column = next(
                (column for column, value in values.items() if value == "項次"),
                None,
            )
            answer_column = next(
                (column for column, value in values.items() if value == "回答內容"),
                None,
            )
            question_columns = tuple(
                column
                for column, value in sorted(values.items())
                if _QUESTION_HEADER.fullmatch(value)
            )
            if item_column is not None and answer_column is not None and question_columns:
                return row_number, item_column, answer_column, question_columns
        return None

    @staticmethod
    def _parse_rows(
        matrix: dict[int, dict[int, str]],
        *,
        header: tuple[int, int, int, tuple[int, ...]],
        file_sha256: str,
    ) -> tuple[ParsedFaqRow, ...]:
        header_row, item_column, answer_column, question_columns = header
        parsed: list[ParsedFaqRow] = []
        for sheet_row in sorted(row for row in matrix if row > header_row):
            cells = matrix[sheet_row]
            source_item_no = _normalize_text(cells.get(item_column, ""))
            answer = _normalize_answer(cells.get(answer_column, ""))
            raw_questions = tuple(
                _normalize_text(cells.get(column, ""))
                for column in question_columns
                if _normalize_text(cells.get(column, ""))
            )
            questions = _deduplicate_questions(raw_questions)
            if not source_item_no and not answer and not questions:
                continue
            parsed.append(
                _build_row(
                    sheet_row=sheet_row,
                    source_item_no=source_item_no,
                    standard_answer=answer,
                    questions=questions,
                    file_sha256=file_sha256,
                    duplicate_question_count=len(raw_questions) - len(questions),
                )
            )
            if len(parsed) > MAX_IMPORT_ROWS:
                raise FaqImportError("單一 Excel 最多可預覽 2,000 項知識")
        return _flag_cross_row_answer_duplicates(_flag_cross_row_duplicates(tuple(parsed)))


class FaqImportRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    def list_batches(self) -> tuple[FaqImportBatch, ...]:
        with self._sessions() as session:
            records = session.scalars(
                select(FaqImportBatchRecord).order_by(
                    FaqImportBatchRecord.created_at.desc(),
                    FaqImportBatchRecord.batch_id,
                )
            )
            return tuple(_batch_to_domain(record) for record in records)

    def get_batch(self, batch_id: str) -> FaqImportBatch:
        with self._sessions() as session:
            record = session.get(FaqImportBatchRecord, batch_id)
            if record is None:
                raise FaqImportNotFoundError(batch_id)
            return _batch_to_domain(record)

    def create_preview(
        self,
        *,
        original_filename: str,
        dataset_title: str,
        publisher: str,
        uploaded_by: str,
        workbook: ParsedFaqWorkbook,
        source_type: Literal["local_import", "approved_internal_faq"] = "local_import",
        source_url: str | None = None,
        now: datetime | None = None,
    ) -> FaqImportBatch:
        occurred_at = now or datetime.now(UTC)
        if occurred_at.tzinfo is None:
            raise FaqImportError("匯入時間必須包含時區")
        normalized_filename = PurePosixPath(original_filename).name
        if not normalized_filename.lower().endswith(".xlsx"):
            raise FaqImportError("目前僅支援 .xlsx Excel 檔案")
        if len(normalized_filename) > 255:
            raise FaqImportError("Excel 檔名不可超過 255 個字元")
        normalized_source_url = source_url.strip() if source_url else None
        source_id = _source_id(
            source_type=source_type,
            source_url=normalized_source_url,
            file_sha256=workbook.file_sha256,
        )
        KnowledgeSource(
            source_id=source_id,
            supplied_url=normalized_source_url,
            canonical_url=normalized_source_url,
            title=dataset_title.strip(),
            publisher=publisher.strip(),
            source_type=source_type,
            retrieved_at=occurred_at,
            topics=["內部 FAQ"],
            status=SourceStatus.ACTIVE,
        )

        with self._sessions.begin() as session:
            existing_query = select(FaqImportBatchRecord).where(
                FaqImportBatchRecord.file_sha256 == workbook.file_sha256
            )
            if source_type == "local_import":
                existing_query = existing_query.where(
                    FaqImportBatchRecord.source_type == "local_import"
                )
            else:
                existing_query = existing_query.where(FaqImportBatchRecord.source_id == source_id)
            existing = session.scalar(existing_query)
            if existing is not None:
                return _batch_to_domain(existing)
            existing_questions = {
                normalized_text: knowledge_id
                for normalized_text, knowledge_id in session.execute(
                    select(
                        KnowledgeQuestionVariantRecord.normalized_text,
                        KnowledgeQuestionVariantRecord.knowledge_id,
                    ).where(
                        KnowledgeQuestionVariantRecord.usage == QuestionVariantUsage.RETRIEVAL.value
                    )
                )
            }
            existing_answers: dict[str, list[str]] = {}
            for knowledge_id, standard_answer in session.execute(
                select(
                    KnowledgeItemRecord.knowledge_id,
                    KnowledgeItemRecord.standard_answer,
                ).where(
                    KnowledgeItemRecord.source_type.in_(("approved_internal_faq", "local_import"))
                )
            ):
                existing_answers.setdefault(
                    _normalized_answer(standard_answer),
                    [],
                ).append(knowledge_id)
            preview_rows = _flag_existing_answer_duplicates(
                workbook.rows,
                existing_answers=existing_answers,
            )
            preview_rows = _flag_existing_conflicts(
                preview_rows,
                existing_questions=existing_questions,
            )
            record = FaqImportBatchRecord(
                batch_id=str(uuid4()),
                original_filename=normalized_filename,
                file_sha256=workbook.file_sha256,
                dataset_title=dataset_title.strip(),
                publisher=publisher.strip(),
                source_url=normalized_source_url,
                source_id=source_id,
                source_type=source_type,
                uploaded_by=uploaded_by,
                status=FaqImportStatus.PREVIEW.value,
                sheet_name=workbook.sheet_name,
                rows=[row.to_json() for row in preview_rows],
                row_count=len(preview_rows),
                valid_row_count=sum(row.is_actionable for row in preview_rows),
                imported_count=0,
                row_version=1,
                created_at=occurred_at,
                imported_at=None,
            )
            session.add(record)
            session.flush()
            return _batch_to_domain(record)

    def import_drafts(
        self,
        *,
        batch_id: str,
        selected_row_ids: tuple[str, ...],
        actor: GovernanceActor,
        expected_version: int,
        now: datetime | None = None,
    ) -> tuple[str, ...]:
        if KnowledgeRole.AUTHOR not in actor.roles:
            raise FaqImportError("FAQ 匯入必須使用作者身分")
        occurred_at = now or datetime.now(UTC)
        if occurred_at.tzinfo is None:
            raise FaqImportError("匯入時間必須包含時區")
        selected = set(selected_row_ids)
        if not selected:
            raise FaqImportError("請至少選擇一筆可匯入的 FAQ")

        with self._sessions.begin() as session:
            batch_record = session.scalar(
                select(FaqImportBatchRecord)
                .where(FaqImportBatchRecord.batch_id == batch_id)
                .with_for_update()
            )
            if batch_record is None:
                raise FaqImportNotFoundError(batch_id)
            if batch_record.status != FaqImportStatus.PREVIEW.value:
                raise FaqImportError("此批次已完成匯入")
            if batch_record.row_version != expected_version:
                raise FaqImportError("匯入預覽已被更新，請重新載入")
            if batch_record.uploaded_by != actor.actor_id:
                raise FaqImportError("只有建立預覽的作者可以確認匯入")

            rows = tuple(ParsedFaqRow.from_json(row) for row in batch_record.rows)
            available = {row.row_id: row for row in rows if row.is_actionable}
            unknown = selected - available.keys()
            if unknown:
                raise FaqImportError("選取內容包含不可匯入或不存在的 FAQ")
            chosen_rows = tuple(row for row in rows if row.row_id in selected)
            chosen_answers = {_normalized_answer(row.standard_answer) for row in chosen_rows}
            for knowledge_id, standard_answer in session.execute(
                select(
                    KnowledgeItemRecord.knowledge_id,
                    KnowledgeItemRecord.standard_answer,
                ).where(
                    KnowledgeItemRecord.source_type.in_(("approved_internal_faq", "local_import"))
                )
            ):
                if _normalized_answer(standard_answer) in chosen_answers:
                    raise FaqImportError(
                        f"預覽建立後已有相同標準答案：{knowledge_id}，請重新建立預覽"
                    )

            knowledge_ids = tuple(row.proposed_knowledge_id for row in chosen_rows)
            existing_id = session.scalar(
                select(KnowledgeItemRecord.knowledge_id).where(
                    KnowledgeItemRecord.knowledge_id.in_(knowledge_ids)
                )
            )
            if existing_id is not None:
                raise FaqImportError(f"知識編號已存在：{existing_id}")

            source_record = session.get(KnowledgeSourceRecord, batch_record.source_id)
            if source_record is None:
                source_record = KnowledgeSourceRecord(
                    source_id=batch_record.source_id,
                    supplied_url=batch_record.source_url,
                    canonical_url=batch_record.source_url,
                    title=batch_record.dataset_title,
                    publisher=batch_record.publisher,
                    source_type=batch_record.source_type,
                    retrieved_at=batch_record.created_at,
                    topics=["內部 FAQ"],
                    status=SourceStatus.ACTIVE.value,
                    notes=_source_notes(batch_record),
                )
                session.add(source_record)
                session.flush()

            for row in chosen_rows:
                item = _row_to_knowledge_item(
                    row,
                    batch=batch_record,
                    author=actor.actor_id,
                )
                session.add(_item_record(item, occurred_at))

            batch_record.status = FaqImportStatus.IMPORTED.value
            batch_record.imported_count = len(chosen_rows)
            batch_record.imported_at = occurred_at
            batch_record.row_version += 1
            batch_record.rows = [
                ParsedFaqRow(
                    row_id=row.row_id,
                    sheet_row=row.sheet_row,
                    source_item_no=row.source_item_no,
                    proposed_knowledge_id=row.proposed_knowledge_id,
                    title=row.title,
                    standard_answer=row.standard_answer,
                    questions=row.questions,
                    warnings=row.warnings,
                    errors=row.errors,
                    duplicate_knowledge_ids=row.duplicate_knowledge_ids,
                    imported=row.row_id in selected,
                ).to_json()
                for row in rows
            ]
            session.flush()
            return knowledge_ids


def _build_row(
    *,
    sheet_row: int,
    source_item_no: str,
    standard_answer: str,
    questions: tuple[str, ...],
    file_sha256: str,
    duplicate_question_count: int,
) -> ParsedFaqRow:
    warnings: list[str] = []
    errors: list[str] = []
    if not source_item_no:
        warnings.append("缺少項次，知識編號將改用 Excel 列號")
    if not standard_answer:
        errors.append("標準答案不可為空")
    if not questions:
        warnings.append("沒有問句變體，匯入後不會增加檢索泛化能力")
    if duplicate_question_count:
        warnings.append(f"已移除 {duplicate_question_count} 筆完全重複問句")
    if len(questions) > MAX_QUESTIONS_PER_ITEM:
        errors.append("單一知識項目最多可匯入 200 筆問句變體")
    if any(not _normalized_retrieval_question(question) for question in questions):
        errors.append("問句變體必須包含文字或數字")

    guard = SensitiveDataGuard()
    guarded_values = (standard_answer, *questions)
    if any(value and guard.scan(value).has_sensitive_data for value in guarded_values):
        errors.append("內容疑似包含個資、帳號、密碼或驗證碼")

    policy = DomainPolicyEngine()
    risky_rules = {
        (
            result.policy_rule_id,
            result.intent,
        )
        for question in questions
        if (result := policy.classify(question)).action
        in {PolicyAction.REFUSE, PolicyAction.HANDOFF}
        and result.policy_rule_id != "POL-DEFAULT-DENY"
    }
    for rule_id, intent in sorted(risky_rules):
        warnings.append(f"問句觸發既有風險規則：{rule_id}（{intent}）")

    item_key = _id_fragment(source_item_no) or f"R{sheet_row}"
    proposed_id = f"K-FAQ-{file_sha256[:8].upper()}-{item_key}-R{sheet_row}"
    title = _draft_title(source_item_no, questions)
    return ParsedFaqRow(
        row_id=f"row-{sheet_row}",
        sheet_row=sheet_row,
        source_item_no=source_item_no,
        proposed_knowledge_id=proposed_id,
        title=title,
        standard_answer=standard_answer,
        questions=questions,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def _flag_cross_row_duplicates(rows: tuple[ParsedFaqRow, ...]) -> tuple[ParsedFaqRow, ...]:
    owners: dict[str, list[str]] = {}
    for row in rows:
        for question in row.questions:
            owners.setdefault(_question_key(question), []).append(row.row_id)
    duplicates = {key for key, row_ids in owners.items() if len(set(row_ids)) > 1}
    if not duplicates:
        return rows

    resolved: list[ParsedFaqRow] = []
    for row in rows:
        duplicate_count = sum(_question_key(question) in duplicates for question in row.questions)
        warnings = row.warnings
        if duplicate_count:
            warnings += (f"有 {duplicate_count} 筆問句也出現在其他知識項目",)
        resolved.append(
            ParsedFaqRow(
                row_id=row.row_id,
                sheet_row=row.sheet_row,
                source_item_no=row.source_item_no,
                proposed_knowledge_id=row.proposed_knowledge_id,
                title=row.title,
                standard_answer=row.standard_answer,
                questions=row.questions,
                warnings=warnings,
                errors=row.errors,
                duplicate_knowledge_ids=row.duplicate_knowledge_ids,
                imported=row.imported,
            )
        )
    return tuple(resolved)


def _flag_cross_row_answer_duplicates(
    rows: tuple[ParsedFaqRow, ...],
) -> tuple[ParsedFaqRow, ...]:
    owners: dict[str, list[str]] = {}
    for row in rows:
        normalized = _normalized_answer(row.standard_answer)
        if normalized:
            owners.setdefault(normalized, []).append(row.row_id)
    duplicates = {normalized for normalized, row_ids in owners.items() if len(row_ids) > 1}
    if not duplicates:
        return rows

    resolved: list[ParsedFaqRow] = []
    for row in rows:
        errors = row.errors
        if _normalized_answer(row.standard_answer) in duplicates:
            errors += ("同一 Excel 內有重複的標準答案，請先確認應保留的資料列",)
        resolved.append(
            ParsedFaqRow(
                row_id=row.row_id,
                sheet_row=row.sheet_row,
                source_item_no=row.source_item_no,
                proposed_knowledge_id=row.proposed_knowledge_id,
                title=row.title,
                standard_answer=row.standard_answer,
                questions=row.questions,
                warnings=row.warnings,
                errors=errors,
                duplicate_knowledge_ids=row.duplicate_knowledge_ids,
                imported=row.imported,
            )
        )
    return tuple(resolved)


def _flag_existing_conflicts(
    rows: tuple[ParsedFaqRow, ...],
    *,
    existing_questions: dict[str, str],
) -> tuple[ParsedFaqRow, ...]:
    resolved: list[ParsedFaqRow] = []
    for row in rows:
        if row.duplicate_knowledge_ids:
            resolved.append(row)
            continue
        conflicts = {
            existing_questions[normalized]
            for question in row.questions
            if (normalized := _normalized_retrieval_question(question)) in existing_questions
        }
        warnings = row.warnings
        if conflicts:
            warnings += ("有問句已屬於既有知識項目：" + "、".join(sorted(conflicts)),)
        resolved.append(
            ParsedFaqRow(
                row_id=row.row_id,
                sheet_row=row.sheet_row,
                source_item_no=row.source_item_no,
                proposed_knowledge_id=row.proposed_knowledge_id,
                title=row.title,
                standard_answer=row.standard_answer,
                questions=row.questions,
                warnings=warnings,
                errors=row.errors,
                duplicate_knowledge_ids=row.duplicate_knowledge_ids,
                imported=row.imported,
            )
        )
    return tuple(resolved)


def _flag_existing_answer_duplicates(
    rows: tuple[ParsedFaqRow, ...],
    *,
    existing_answers: dict[str, list[str]],
) -> tuple[ParsedFaqRow, ...]:
    resolved: list[ParsedFaqRow] = []
    for row in rows:
        duplicates = tuple(
            sorted(existing_answers.get(_normalized_answer(row.standard_answer), ()))
        )
        warnings = row.warnings
        if duplicates:
            warnings += ("標準答案已存在於 " + "、".join(duplicates) + "；本列不會再建立知識草稿",)
        resolved.append(
            ParsedFaqRow(
                row_id=row.row_id,
                sheet_row=row.sheet_row,
                source_item_no=row.source_item_no,
                proposed_knowledge_id=row.proposed_knowledge_id,
                title=row.title,
                standard_answer=row.standard_answer,
                questions=row.questions,
                warnings=warnings,
                errors=row.errors,
                duplicate_knowledge_ids=duplicates,
                imported=row.imported,
            )
        )
    return tuple(resolved)


def _row_to_knowledge_item(
    row: ParsedFaqRow,
    *,
    batch: FaqImportBatchRecord,
    author: str,
) -> KnowledgeItem:
    item_locator = row.source_item_no or f"Excel row {row.sheet_row}"
    return KnowledgeItem(
        knowledge_id=row.proposed_knowledge_id,
        title=row.title,
        standard_answer=row.standard_answer,
        source_id=batch.source_id,
        source_uri=batch.source_url,
        source_locator=(
            f"FAQ 匯入批次 {batch.batch_id}／{batch.sheet_name} 第 {row.sheet_row} 列"
            f"／項次 {item_locator}"
        ),
        source_type=cast(
            Literal["local_import", "approved_internal_faq"],
            batch.source_type,
        ),
        products=[],
        platforms=[],
        app_versions=[],
        author=author,
        version="1.0-draft",
        change_summary="由核准內部 FAQ Excel 匯入為草稿",
        status=KnowledgeStatus.DRAFT,
        public_answer_allowed=False,
        allowed_intents=["faq_general_guidance"],
        prohibited_extensions=[
            "不得查詢、推測或揭露客戶個人資料",
            "不得代客執行交易、帳戶或驗證操作",
            "不得延伸為個人化投資建議",
        ],
        question_variants=[
            QuestionVariant(
                variant_id=str(uuid4()),
                question_text=question,
                usage=QuestionVariantUsage.RETRIEVAL,
            )
            for question in row.questions
        ],
    )


def _item_record(item: KnowledgeItem, now: datetime) -> KnowledgeItemRecord:
    record = KnowledgeItemRecord(
        knowledge_id=item.knowledge_id,
        title=item.title,
        standard_answer=item.standard_answer,
        source_id=item.source_id,
        source_uri=item.source_uri,
        source_locator=item.source_locator,
        source_type=item.source_type,
        products=item.products,
        platforms=item.platforms,
        app_versions=item.app_versions,
        effective_at=None,
        expires_at=None,
        review_at=None,
        owner_unit=None,
        author=item.author,
        reviewer=None,
        approver=None,
        approved_at=None,
        version=item.version,
        change_summary=item.change_summary,
        previous_version=None,
        status=item.status.value,
        public_answer_allowed=False,
        allowed_intents=item.allowed_intents,
        prohibited_extensions=item.prohibited_extensions,
        asr_terms=[],
        row_version=1,
        created_at=now,
        updated_at=now,
    )
    record.question_variants = [
        KnowledgeQuestionVariantRecord(
            variant_id=variant.variant_id,
            knowledge_id=item.knowledge_id,
            question_text=variant.question_text,
            normalized_text=_normalized_retrieval_question(variant.question_text),
            usage=variant.usage.value,
            position=position,
            created_at=now,
            updated_at=now,
        )
        for position, variant in enumerate(item.question_variants)
    ]
    return record


def _batch_to_domain(record: FaqImportBatchRecord) -> FaqImportBatch:
    return FaqImportBatch(
        batch_id=record.batch_id,
        original_filename=record.original_filename,
        file_sha256=record.file_sha256,
        dataset_title=record.dataset_title,
        publisher=record.publisher,
        source_url=record.source_url,
        source_id=record.source_id,
        source_type=cast(
            Literal["local_import", "approved_internal_faq"],
            record.source_type,
        ),
        uploaded_by=record.uploaded_by,
        status=FaqImportStatus(record.status),
        sheet_name=record.sheet_name,
        rows=tuple(ParsedFaqRow.from_json(row) for row in record.rows),
        row_count=record.row_count,
        valid_row_count=record.valid_row_count,
        imported_count=record.imported_count,
        row_version=record.row_version,
        created_at=_as_utc(record.created_at),
        imported_at=_as_utc(record.imported_at) if record.imported_at else None,
    )


def _source_id(
    *,
    source_type: Literal["local_import", "approved_internal_faq"],
    source_url: str | None,
    file_sha256: str,
) -> str:
    if source_type == "local_import":
        return f"SRC-LOCAL-{file_sha256[:12].upper()}"
    if source_url is None:
        raise FaqImportError("核准內部 FAQ 必須提供正式 HTTPS 網址")
    digest = hashlib.sha256(source_url.encode()).hexdigest()[:12].upper()
    return f"SRC-FAQ-{digest}"


def _source_notes(batch: FaqImportBatchRecord) -> str:
    if batch.source_type == "local_import":
        return (
            f"本機匯入資料；原始檔不保存。批次：{batch.batch_id}；檔案 SHA-256：{batch.file_sha256}"
        )
    return f"核准內部 FAQ 資料集；原始檔不保存。首次匯入批次：{batch.batch_id}"


def _draft_title(source_item_no: str, questions: tuple[str, ...]) -> str:
    if questions:
        return questions[0].rstrip("？?。 ").strip()[:200]
    return f"FAQ 項次 {source_item_no}" if source_item_no else "待補標題的 FAQ 草稿"


def _deduplicate_questions(questions: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    resolved: list[str] = []
    for question in questions:
        key = _normalized_retrieval_question(question) or _question_key(question)
        if key not in seen:
            seen.add(key)
            resolved.append(question)
    return tuple(resolved)


def _question_key(value: str) -> str:
    return _normalized_retrieval_question(value) or " ".join(value.split()).casefold()


def _normalized_retrieval_question(value: str) -> str:
    return _NORMALIZE_QUESTION.sub("", value.casefold())


def _normalized_answer(value: str) -> str:
    return "".join(value.split()).casefold()


def _normalize_header(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _normalize_answer(value: str) -> str:
    lines = [line.strip() for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _column_number(letters: str) -> int:
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - ord("A") + 1
    return value


def _id_fragment(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", value.upper()).strip("-")[:24]


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _json_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise FaqImportError("FAQ 匯入批次資料格式不正確")
    return value


def _json_int(value: object) -> int:
    if not isinstance(value, int):
        raise FaqImportError("FAQ 匯入批次資料格式不正確")
    return value
