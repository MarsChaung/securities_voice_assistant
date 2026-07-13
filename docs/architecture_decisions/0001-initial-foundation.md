# ADR-0001：初始安全基礎

- 狀態：暫定，待 Phase 0 核准
- 日期：2026-07-14

## 決策

採 Python 3.12、FastAPI、Pydantic 與 `uv` 建立模組化單體。第一個可執行切片只包含 sensitive-data guard、確定性 domain policy、回答合約、安全稽核中繼資料與 HTTP API。

在核准知識庫可用前，允許類別仍回傳 `refuse`。LLM 與 MLX Audio 僅保留設定，不進入執行流程。

## 理由

先讓禁止意圖與個資在流程前端被阻擋，可獨立驗證 default deny，也避免把尚未核准的產品或法遵選擇固化到 RAG 實作。

## 待確認

- 正式服務範圍矩陣與規則 ID。
- 官方轉人工文案與聯絡管道。
- 稽核資料欄位、儲存位置與保留期限。

