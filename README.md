# 證券知識型語音客服

本 repository 是 `SYSTEM_PLAN.md` 的初始工程基礎。目前目標場景為內部 Web Pilot，已完成安全決策管線、知識治理資料庫、核准知識檢索、結構化 LLM 意圖路由、背景 Shadow 答案評測與人工複核工作台，以及全地端即時 ASR／串流 TTS 的文字與語音問答介面；客戶資料系統不在 Pilot 連線範圍。

## 目前行為

`POST /v1/turns/evaluate` 依序執行：

1. 個資與機敏資料偵測。
2. 確定性硬性政策分類。
3. 依設定停用、影子執行或套用結構化 LLM 意圖路由。
4. 檢索符合條件的已發布知識。
5. 依回答模式產生 `answer | refuse | handoff` 結構化合約。

允許範圍的問題只會檢索 PostgreSQL 中有效的 `published` 知識，通過意圖、平台、最低分數及歧義門檻後，回傳人工核准內容與可追溯來源。本機匯入來源可不具正式網址，但保留批次、檔案雜湊與列號定位。檢索預設使用可解釋的詞彙模式，也可在模型通過離線評測後切換成本機 embeddings 的混合模式。沒有可靠匹配時安全拒答，不會使用模型自身知識補答。

回答模式程式預設為 `exact`，直接回傳人工核准標準答案。內部 Pilot 目前使用 `shadow_llm`：API 先回傳核准原文，再由單工背景佇列執行有限改寫與輸出守門；同一知識版本在程序內只評測一次，避免重複占用本機推論資源。`controlled_llm` 才會在守門通過時使用改寫答案；生成失敗、逾時、格式錯誤，或偵測到新增數字、未授權產品詞、投資建議、提示詞洩漏、機敏資料及禁止延伸時，一律回退標準答案。`fixed_message` 是緊急安全模式。詳見 [ADR-0003](docs/architecture_decisions/0003-controlled-answer-generation.md) 與 [ADR-0005](docs/architecture_decisions/0005-shadow-answer-generation.md)。

一般技術 log 不保存原始音訊、完整逐字稿、使用者輸入或回答全文，只記錄固定白名單的決策與效能中繼資料。Pilot 回饋只接受「有幫助／沒幫助」，不接受自由文字。離線評測案例必須標示為合成資料。

意圖路由提供 `disabled`、`shadow`、`controlled` 三種模式。個資、交易、個人帳務、憑證、投資建議及既有轉人工規則永遠先執行，LLM 不得覆寫硬性結果；模型只輸出 strict JSON Schema 約束的候選意圖、信心、風險旗標及釐清狀態。低信心或模型故障時回退確定性規則，額外風險旗標只會使結果更保守。詳見 [ADR-0004](docs/architecture_decisions/0004-structured-llm-intent-router.md)。

## 初始知識資料

`knowledge/` 已登錄四個國泰綜合證券官方頁面，並整理成可追溯的初始知識草稿。檔案匯入資料預設為 `draft` 且 `public_answer_allowed=false`；資料庫中只有完成核准、尚未過期且未超過複審到期時間的 `published` 項目才可能進入 runtime。

FAQ 採「一筆標準答案＋多筆問句變體」共同治理。作者可在草稿狀態編輯標題、標準答案，並新增、修改、刪除或分類問句變體；送審與發布以整項知識為單位。只有標記為正式檢索的問句會進入 Runtime，僅供評測或排除的問句不影響線上排序。發布版建立新版時，標準答案與問句變體會一起保存為不可變快照。詳見 [ADR-0008](docs/architecture_decisions/0008-governed-faq-question-variants.md)。

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

Pilot 會顯示目前可回答知識筆數，並呈現答案、澄清、拒答或轉人工狀態、官方來源、決策資訊及二元回饋。啟用語音服務後，麥克風按鈕會使用 MLX Audio 即時 ASR 自動判斷句尾，並優先以當下有效知識中受治理的語音辨識詞彙提供領域提示，未設定時才使用標題與產品名稱。辨識文字會進入同一安全決策管線；安全問題若一般檢索無結果，會先比對人工確認的 ASR 別名，再以受控音近候選比對核准知識，唯一高信心候選才回答，有歧義則要求重述。播放期間按麥克風可中斷並繼續說話。瀏覽器不使用 localStorage 保存問答內容，重新整理或按下「清除畫面」就會移除畫面中的對話。詳見 [ADR-0010](docs/architecture_decisions/0010-governed-asr-phonetic-recovery.md) 與 [ADR-0011](docs/architecture_decisions/0011-versioned-asr-vocabulary.md)。

知識治理中心使用 PostgreSQL 保存治理狀態。先啟動資料庫、升級 schema 並匯入初始草稿：

```bash
docker compose up -d postgres
uv run alembic upgrade head
uv run python -m knowledge_admin.seed
uv run uvicorn knowledge_admin.api:app --host 127.0.0.1 --reload --port 8081
```

管理介面：<http://127.0.0.1:8081/admin/knowledge>

Shadow 複核工作台：<http://127.0.0.1:8081/admin/shadow>

介面可用開發身分模擬器操作 4 個官方來源與 15 筆知識草稿，編輯標準答案與問句變體，並執行送審、完成審核、核准、發布、退回及撤銷。模擬身分只允許在 development 使用；正式 SSO/RBAC 仍留待公司環境整合。

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

本機 embeddings 服務必須提供 OpenAI 相容的 `POST /v1/embeddings`。Docker 連回宿主服務時預設使用 `http://host.docker.internal:12345/v1`，需要不同位置可設定 `SVA_EMBEDDINGS_DOCKER_BASE_URL`。啟用前先執行 `uv run python -m evals.run_hybrid` 比較詞彙與混合檢索結果。Embedding 服務失效時 Runtime 會安全退回詞彙檢索。

Phase 3.2 的受控答案生成目前只啟用 Shadow；使用者仍會看到人工核准原文：

```bash
SVA_ANSWER_MODE=shadow_llm
SVA_ANSWER_LLM_MODEL=Qwen3.6-35B-A3B-oQ4
SVA_SHADOW_MAX_PENDING=8
SVA_LLM_API_KEY=<local-api-key>
```

生成器只接收標準答案、禁止延伸與知識中繼資料，不接收原始使用者問題。主要 turn 稽核只記 `shadow_queued`、`shadow_cached` 或 `shadow_queue_full`；背景完成後另記不含答案全文的 `shadow_generation` 事件。專用 Shadow 複核資料表會保存標準答案快照、模型改寫與人工標記，但不保存原始問句或音訊，並與一般技術 Log 分離。管理介面可查看待複核、可接受、不採用、無法產生、輸出守門與延遲統計；人工複核不能直接啟用 `controlled_llm`。詳細設計見 [ADR-0007](docs/architecture_decisions/0007-shadow-review-workbench.md)。

15 筆知識的離線評測為輸出守門 15/15、獨立 groundedness judge 15/15；8 筆有改寫、7 筆保留原文。Judge 使用 `gemma-4-31b-it-6bit`，只在離線評測執行，不參與線上回答。

`controlled_llm` 只負責依核准內容改寫，與意圖路由是不同開關。目前內部 Pilot 的意圖模型為 `Qwen3.6-35B-A3B-oQ4`，44 筆擴充評測為安全意圖 29/29、風險辨識 15/15，平均延遲約 1.91 秒、P95 約 2.17 秒。可用下列設定啟用：

```bash
SVA_INTENT_ROUTER_MODE=controlled
SVA_INTENT_LLM_MODEL=Qwen3.6-35B-A3B-oQ4
SVA_INTENT_ROUTER_MINIMUM_CONFIDENCE=0.8
SVA_LLM_API_KEY=<local-api-key>
```

即時語音 Demo 需要另外啟動 MLX Audio `127.0.0.1:8000`，並安裝指定 ASR／TTS 模型：

```bash
SVA_VOICE_ENABLED=true
SVA_TTS_BASE_URL=http://127.0.0.1:8000/v1
SVA_AUDIO_PUBLIC_BASE_URL=http://127.0.0.1:8000/v1
SVA_ASR_MODEL=mlx-community/Qwen3-ASR-1.7B-8bit
SVA_ASR_CANDIDATE_MODEL=mlx-community/whisper-large-v3-turbo-asr-fp16
SVA_TTS_MODEL=mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit
SVA_TTS_VOICE=Vivian
SVA_TTS_REF_AUDIO=<宿主機上的授權參考音檔絕對路徑>
SVA_TTS_REF_TEXT=<參考音檔逐字稿>
```

`SVA_ASR_MODEL` 是預設模型，目前依 Web Pilot 實測採用 `Qwen3-ASR-1.7B-8bit`；`SVA_ASR_CANDIDATE_MODEL` 保留 Whisper 作為 A/B 對照與備援。設定候選模型後，Web Pilot 會顯示 A/B 選單，停止語音工作階段時可在兩個受控模型間切換。`SVA_TTS_REF_AUDIO` 與 `SVA_TTS_REF_TEXT` 必須同時設定，且只能保存在被 Git 忽略的 `.env`。MLX Audio 執行於宿主機，因此收到的是宿主機絕對路徑，不需將參考音檔掛載進 orchestrator 容器。瀏覽器只直連 MLX Audio 的即時 ASR WebSocket；LLM API key、意圖路由、知識檢索及 TTS 呼叫都留在後端。`POST /v1/voice/respond-stream` 不接受任意朗讀文字，只接受 ASR 逐字稿，並強制先執行與文字問答相同的政策及知識流程。TTS 仍透過 MLX Audio API，回答依 `，。？、` 與換行優先在 80 字附近分段，找不到合適切點時才於 96 字硬切。一般 log 不保存音訊、逐字稿、參考素材或答案全文，只記 TTS 模型、分段／音訊片段數與延遲。詳見 [ADR-0006](docs/architecture_decisions/0006-realtime-voice-pilot.md)。

Demo 前的預熱、允許／拒答案例、插話中斷與人工口音驗收步驟，請依 [即時語音 Web Pilot Demo 清單](docs/demo/voice-pilot-checklist.md) 執行。

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
uv run python -m evals.run_answer_generation
```

## Docker

```bash
docker compose up --build
```

Compose 會啟動 PostgreSQL、orchestrator API 與知識治理中心，所有 host port 都只綁定 `127.0.0.1`。PostgreSQL 資料保存在 `knowledge-db` volume；oMLX 與 MLX Audio 維持宿主機上的獨立推論服務，orchestrator 透過 `host.docker.internal` 呼叫。

## 尚待 Phase 0 核准

- 允許、拒答與轉人工意圖矩陣的正式內容。
- 音訊、ASR 文字與稽核資料的留存期限。
- 知識撰寫、審核、發布與撤銷角色。
- 安全 KPI 與正式服務通道。
- 官方客服、申訴與緊急停用管道文案。

目前規則是保守的工程預設，不代表產品、法遵或業務核准結果。
