# 證券知識型語音客服

本 repository 是 `SYSTEM_PLAN.md` 的初始工程基礎。目前目標場景為內部 Web Pilot，已完成安全決策管線、知識治理資料庫、核准知識檢索、結構化 LLM 意圖路由、受控答案生成安全骨架及內部文字問答介面；ASR、TTS 及客戶資料系統尚未接入執行流程。

## 目前行為

`POST /v1/turns/evaluate` 依序執行：

1. 個資與機敏資料偵測。
2. 確定性硬性政策分類。
3. 依設定停用、影子執行或套用結構化 LLM 意圖路由。
4. 檢索符合條件的已發布知識。
5. 依回答模式產生 `answer | refuse | handoff` 結構化合約。

允許範圍的問題只會檢索 PostgreSQL 中有效的 `published` 知識，通過意圖、平台、最低分數及歧義門檻後，回傳人工核准內容與官方來源引用。檢索預設使用可解釋的詞彙模式，也可在模型通過離線評測後切換成本機 embeddings 的混合模式。沒有可靠匹配時安全拒答，不會使用模型自身知識補答。

回答模式預設為 `exact`，直接回傳人工核准標準答案。`controlled_llm` 只允許本機 LLM 依單筆已核准標準答案做有限改寫，生成失敗、逾時、格式錯誤，或輸出守門偵測到新增數字、未授權產品詞、投資建議、提示詞洩漏、機敏資料及禁止延伸時，一律回退標準答案。`fixed_message` 是緊急安全模式，允許類問題也只回固定訊息。詳見 [ADR-0003](docs/architecture_decisions/0003-controlled-answer-generation.md)。

一般技術 log 不保存原始音訊、完整逐字稿、使用者輸入或回答全文，只記錄固定白名單的決策與效能中繼資料。Pilot 回饋只接受「有幫助／沒幫助」，不接受自由文字。離線評測案例必須標示為合成資料。

意圖路由提供 `disabled`、`shadow`、`controlled` 三種模式。個資、交易、個人帳務、憑證、投資建議及既有轉人工規則永遠先執行，LLM 不得覆寫硬性結果；模型只輸出 strict JSON Schema 約束的候選意圖、信心、風險旗標及釐清狀態。低信心或模型故障時回退確定性規則，額外風險旗標只會使結果更保守。詳見 [ADR-0004](docs/architecture_decisions/0004-structured-llm-intent-router.md)。

## 初始知識資料

`knowledge/` 已登錄四個國泰綜合證券官方頁面，並整理成可追溯的初始知識草稿。檔案匯入資料預設為 `draft` 且 `public_answer_allowed=false`；資料庫中只有完成核准、尚未過期且未超過複審到期時間的 `published` 項目才可能進入 runtime。

目前不保存整頁 HTML，並排除行銷標的推薦、歷史績效、密碼與個人帳務內容。詳細範圍請見 `knowledge/README.md`。

## 本機開發

需求：`uv`、Python 3.12。

```bash
cp .env.example .env
uv sync
uv run uvicorn orchestrator.api:app --reload --port 8080
```

API 文件：<http://127.0.0.1:8080/docs>

內部 Web Pilot：<http://127.0.0.1:8080/pilot>

本機 LLM 推論平台管理介面：<http://127.0.0.1:12345/admin>。管理密碼只保存在被 Git 忽略的 `.env` 變數 `SVA_LLM_ADMIN_PASSWORD`；若 embeddings 或其他本機模型尚未安裝，可從該管理介面下載。

Pilot 會顯示目前可回答知識筆數，並呈現答案、拒答或轉人工狀態、官方來源、決策資訊及二元回饋。瀏覽器不使用 localStorage 保存問答內容，重新整理或按下「清除畫面」就會移除畫面中的對話。

知識治理中心使用 PostgreSQL 保存治理狀態。先啟動資料庫、升級 schema 並匯入初始草稿：

```bash
docker compose up -d postgres
uv run alembic upgrade head
uv run python -m knowledge_admin.seed
uv run uvicorn knowledge_admin.api:app --host 127.0.0.1 --reload --port 8081
```

管理介面：<http://127.0.0.1:8081/admin/knowledge>

介面可用開發身分模擬器操作 4 個官方來源與 15 筆知識草稿，並執行送審、完成審核、核准、發布、退回及撤銷。模擬身分只允許在 development 使用；正式 SSO/RBAC 仍留待公司環境整合。

`published` 且已生效、未過期、未超過複審到期時間的知識會立即成為 orchestrator 候選資料；撤銷後不再進入 Runtime。已超過複審到期時間的發布項目可由原作者建立下一版草稿，系統會先保存不可變的舊發布版快照，再重新執行送審、核准與發布。

混合檢索採功能開關控制，預設不啟用：

```bash
SVA_RETRIEVAL_MODE=hybrid
SVA_EMBEDDINGS_BASE_URL=http://127.0.0.1:12345/v1
SVA_EMBEDDINGS_MODEL=<通過評測的本機模型 ID>
# 端點要求驗證時，僅在未納入版控的 .env 設定：
SVA_EMBEDDINGS_API_KEY=<local-api-key>
```

Embedding 模型若要求特定輸入格式，可另設 `SVA_EMBEDDINGS_QUERY_PREFIX` 與 `SVA_EMBEDDINGS_DOCUMENT_PREFIX`。例如 multilingual-e5 檢索應分別使用 `query: ` 與 `passage: `；Qwen3 Embedding 則建議在 query 端加入描述檢索任務的一句 instruction。這些前綴必須與離線評測及 Runtime 保持一致。

目前 Pilot 實測候選為 `Qwen3-Embedding-4B-4bit-DWQ`，擴充後的知識改寫與危險近似案例結果為混合檢索 30/30、詞彙基準 11/30。設定與安全邊界詳見 [ADR-0002](docs/architecture_decisions/0002-hybrid-retrieval.md)。

本機 embeddings 服務必須提供 OpenAI 相容的 `POST /v1/embeddings`。Docker 連回宿主服務時預設使用 `http://host.docker.internal:12345/v1`，需要不同位置可設定 `SVA_EMBEDDINGS_DOCKER_BASE_URL`。啟用前先執行 `uv run python -m evals.run_hybrid` 比較詞彙與混合檢索結果。Embedding 服務失效時 Runtime 會安全退回詞彙檢索；目前預設的 `exact` 回答模式不會呼叫 LLM 改寫答案。

Phase 3 的受控答案生成先維持關閉；完成模型安全評測後才可在未納入版控的 `.env` 顯式設定：

```bash
SVA_ANSWER_MODE=controlled_llm
SVA_ANSWER_LLM_MODEL=<通過安全評測的本機模型 ID>
SVA_LLM_API_KEY=<local-api-key>
```

`controlled_llm` 只負責依核准內容改寫，與意圖路由是不同開關。目前內部 Pilot 的意圖模型為 `Qwen3.6-35B-A3B-oQ4`，44 筆擴充評測為安全意圖 29/29、風險辨識 15/15，平均延遲約 1.91 秒、P95 約 2.17 秒。可用下列設定啟用：

```bash
SVA_INTENT_ROUTER_MODE=controlled
SVA_INTENT_LLM_MODEL=Qwen3.6-35B-A3B-oQ4
SVA_INTENT_ROUTER_MINIMUM_CONFIDENCE=0.8
SVA_LLM_API_KEY=<local-api-key>
```

```bash
curl -s http://127.0.0.1:8080/healthz
curl -s -X POST http://127.0.0.1:8080/v1/turns/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"transcript":"Web 版要如何操作？","channel":"web"}'
```

## 驗證

```bash
uv run ruff check .
uv run mypy
uv run pytest --cov
uv run python -m evals.run
uv run python -m evals.run_hybrid
uv run python -m evals.run_intent_router
```

## Docker

```bash
docker compose up --build
```

Compose 會啟動 PostgreSQL、orchestrator API 與知識治理中心，所有 host port 都只綁定 `127.0.0.1`。PostgreSQL 資料保存在 `knowledge-db` volume；本機 LLM 與 MLX Audio URL 已保留在設定中，但尚未串接。

## 尚待 Phase 0 核准

- 允許、拒答與轉人工意圖矩陣的正式內容。
- 音訊、ASR 文字與稽核資料的留存期限。
- 知識撰寫、審核、發布與撤銷角色。
- 安全 KPI 與正式服務通道。
- 官方客服、申訴與緊急停用管道文案。

目前規則是保守的工程預設，不代表產品、法遵或業務核准結果。
