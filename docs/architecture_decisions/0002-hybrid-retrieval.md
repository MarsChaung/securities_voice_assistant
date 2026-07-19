# ADR-0002：內部 Web Pilot 混合語意檢索

- 狀態：Pilot 候選，正式上線前仍須公司模型、法遵與資安核准
- 日期：2026-07-19

## 決策

保留確定性政策與知識治理閘門，在其後加入可切換的混合檢索：40% 詞彙分數加上 60% cosine 語意分數。混合模式使用獨立的最低分數 `0.40` 與歧義差距 `0.02`；Embedding 服務異常時，只能退回原詞彙檢索的 `0.55` 與 `0.08`，不得套用較低的混合門檻。

Runtime 只將已發布、生效、未過期且未超過複審到期時間的知識送往本機 OpenAI 相容 `POST /v1/embeddings`。文件向量以知識 ID、版本、輸入前綴與內容雜湊保存在程序內，不保存使用者問句，也不使用 LLM 自由生成或改寫標準答案。

內部 Pilot 的模型候選為 `Qwen3-Embedding-4B-4bit-DWQ`，query instruction 為：

```text
Instruct: Retrieve the approved Traditional Chinese securities knowledge passage that best answers a user question.
Query:
```

模型 ID、API URL、API 金鑰及輸入前綴全部由環境變數提供；API 金鑰不得進入版控、文件、測試資料或一般 log。

## 驗證結果

以涵蓋全部15筆知識的16筆語意改寫及11筆危險近似案例進行本機評測：

- 詞彙檢索：11/27。
- Qwen3 Embedding 4B 4-bit DWQ 混合檢索：27/27。
- Qwen3 Embedding 4B MXFP8：排序表現接近 DWQ，但資源需求較高，未選用。
- multilingual-e5-large：有8筆知識第一名排序錯誤，未選用。

## 安全邊界

- PII、交易、個人帳務、投資建議、憑證與轉人工規則仍在 embeddings 之前執行。
- 意圖未明確允許時不進入混合檢索。
- 找不到充分根據或候選分數過近時仍拒答。
- Embedding 故障不會放寬詞彙檢索門檻。
- Runtime 最終只回傳人工核准的 `standard_answer` 與來源引用。

## 後續條件

目前只有15筆文件，因此使用程序內快取最簡單。文件量、程序數或更新頻率提高後，再評估 PostgreSQL `pgvector`、索引版本、背景重建與一致性切換。正式上線前還必須擴充更多自然改寫、危險近似、ASR 誤字及跨知識歧義案例，重新校準門檻並完成公司源碼與依賴掃描。
