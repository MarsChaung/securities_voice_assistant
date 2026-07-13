# 證券知識型語音客服

本 repository 是 `SYSTEM_PLAN.md` 的初始工程基礎。目前完成 Phase 1 的最小安全決策管線，尚未接入正式知識庫、LLM、ASR、TTS 或客戶資料系統。

## 目前行為

`POST /v1/turns/evaluate` 依序執行：

1. 個資與機敏資料偵測。
2. 確定性意圖與政策分類。
3. 產生 `answer | refuse | handoff` 結構化合約。

現階段沒有核准知識來源，因此即使問題通過允許範圍分類，也會安全拒答，不會使用模型自身知識補答。

## 本機開發

需求：`uv`、Python 3.12。

```bash
cp .env.example .env
uv sync
uv run uvicorn orchestrator.api:app --reload --port 8080
```

API 文件：<http://127.0.0.1:8080/docs>

```bash
curl -s http://127.0.0.1:8080/healthz
curl -s -X POST http://127.0.0.1:8080/v1/turns/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"transcript":"APP 要如何下載？","channel":"web"}'
```

## 驗證

```bash
uv run ruff check .
uv run mypy
uv run pytest --cov
```

## Docker

```bash
docker compose up --build
```

容器僅啟動 orchestrator API。本機 LLM 與 MLX Audio URL 已保留在設定中，但尚未串接，避免在安全閘道與核准知識完成前出現自由回答。

## 尚待 Phase 0 核准

- 允許、拒答與轉人工意圖矩陣的正式內容。
- 音訊、ASR 文字與稽核資料的留存期限。
- 知識撰寫、審核、發布與撤銷角色。
- 安全 KPI、Pilot 對象與正式服務通道。
- 官方客服、申訴與緊急停用管道文案。

目前規則是保守的工程預設，不代表產品、法遵或業務核准結果。
