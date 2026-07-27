# ADR-0006：內部 Web Pilot 即時語音對話

- 狀態：內部 Demo 啟用
- 日期：2026-07-21

## 決策

沿用 `TTS_Streaming` 已驗證的 MLX Audio 架構，在 Web Pilot 加入全地端即時語音通道：

1. 瀏覽器取得麥克風授權，將 16 kHz PCM 送往 MLX Audio 即時 ASR WebSocket。
2. MLX Audio 以 VAD 與語意句尾判斷回傳繁體中文 `utterance_end`。
3. 瀏覽器只把最終逐字稿送至 orchestrator `POST /v1/voice/respond-stream`。
4. orchestrator 強制執行與文字問答相同的個資、硬規則、結構化意圖、核准知識檢索與回答合約。
5. 後端只把該合約的最終回答送往 MLX Audio TTS，逐句接收完整 WAV frame，再以 NDJSON／Base64 串流給瀏覽器依序播放。

瀏覽器不持有 LLM API key、不直連 LLM，也不能要求後端朗讀任意文字。交易、個人帳務、投資建議、個資或無可靠知識來源的問題仍先拒答／轉人工，拒答文字也可由 TTS 朗讀。

## 互動方式

麥克風按鈕開啟連續語音模式；每輪播放完成後才重新聆聽，避免助理自己的聲音觸發 ASR。播放期間再次按下按鈕會中止 HTTP 串流與已排程音訊，立即回到聆聽狀態。文字輸入介面與來源引用、決策資訊及二元回饋維持不變。

## 資料與稽核

原始音訊只在瀏覽器與 MLX Audio 記憶體串流處理，orchestrator 不接收或保存音訊。ASR 逐字稿只存在該 turn 的瀏覽器與 orchestrator 記憶體，不寫入一般 log。`voice_synthesis` 稽核事件只記 Turn ID、TTS 模型、句子／音訊片段數、首段與完整延遲及固定錯誤類型，不記逐字稿、回答或音訊。

## 部署邊界

PostgreSQL、orchestrator 與知識治理中心由本專案 Compose 管理；oMLX 與 MLX Audio 維持宿主機獨立推論服務。瀏覽器使用 `SVA_AUDIO_PUBLIC_BASE_URL` 取得可連線的 ASR WebSocket URL，容器後端使用 `SVA_TTS_BASE_URL` 呼叫 TTS。

為提供台灣口音 Demo，Pilot 使用 Qwen3-TTS Base 6-bit voice clone，仍由宿主機 MLX Audio API 負責推論。授權參考音檔的宿主機絕對路徑與逐字稿只保存在被 Git 忽略的 `.env`，兩者必須同時設定；orchestrator 將它們送往宿主機 MLX Audio，並使用低溫度、`top_k=1`、`top_p=1.0` 與 repetition penalty 的穩定解碼設定。回答只優先依 `，。？、` 與換行在 80 字附近分段，若 96 字內沒有合適切點才硬切，以減少請求與片段數。公開設定 API 只回報 voice clone 是否啟用，不揭露路徑或逐字稿；一般稽核也不記錄參考素材。

## 已知限制

- 即時 ASR 無法連線時目前顯示錯誤，不自動上傳 WebM 作批次辨識。
- 首次載入 ASR／TTS 模型可能明顯較慢，Demo 前應先預熱。
- 播放期間的插話採明確按鈕中斷；尚未啟用常駐麥克風與回音消除後的自動 barge-in。
- 正式環境仍須完成語音模型版本鎖定、依賴與源碼掃描、瀏覽器相容性、噪音／ASR 誤字紅隊及公司資料留存核准。
