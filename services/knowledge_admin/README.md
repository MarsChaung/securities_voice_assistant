# Knowledge admin

內部知識治理中心的本機開發版本。提供來源目錄、FAQ Excel 匯入預覽、知識清單、草稿內容與問句變體編輯、來源追溯、Maker–Checker 工作流及 append-only 稽核事件。

## 啟動

```bash
docker compose up -d postgres
uv run alembic upgrade head
uv run python -m knowledge_admin.seed
uv run uvicorn knowledge_admin.api:app --host 127.0.0.1 --reload --port 8081
```

開啟 <http://127.0.0.1:8081/admin/knowledge>。

## 治理邊界

- PostgreSQL 是治理資料的唯一持久化來源；`knowledge/*.json` 僅供冪等的初始匯入。
- `governance.py` 定義草稿、審核、核准、發布、退回與撤銷的角色隔離規則。
- 作者、審核人、核准人不可兼任；發布人須獨立於前三者。
- App 操作知識仍須指定適用版本，才可能成為 `approved` 或 `published`。
- 每次寫入會檢查 `row_version` 並新增治理事件，避免並行操作靜默覆蓋。
- 一組標準答案與多筆問句變體共同構成一項 FAQ；問句可標示為正式檢索、僅供評測或排除。
- 問句變體只可在草稿狀態由原作者編輯，送審、核准與發布仍以整項知識為單位。
- 建立複審新版時，完整問句變體會隨發布版保存於不可變快照。
- FAQ Excel 上傳只保存解析後預覽、SHA-256 與來源中繼資料，人工確認後才建立草稿；原始檔不保存。
- `local_import` 本機 FAQ 可不填正式網址，改以匯入批次、檔案 SHA-256 與列號追溯；
  `approved_internal_faq` 仍必須提供核准的 HTTPS 資料集登錄頁。
- 本機使用開發身分模擬器；非 development 環境不得啟用，正式環境目前會拒絕啟動。
- `published` 且通過生效、到期、複審到期時間與來源狀態檢查的知識會成為 orchestrator 檢索候選；撤銷後立即排除。
- 複審已到期的發布項目可由原作者建立下一版草稿；舊發布版會保存為不可變快照，新版必須重新走完整 Maker–Checker 流程。
