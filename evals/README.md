# 離線評測

此目錄只能保存合成或完成去識別並經核准的案例。目前所有 JSONL 案例都必須包含 `"synthetic": true`，runner 遇到未標示的資料會直接判定失敗。

```bash
uv run python -m evals.run
```

目前涵蓋：

- 允許、拒答、轉人工與 default-deny 意圖。
- 純下單教學與代客交易的邊界。
- 常見敏感資料類型與遮罩結果。
- 已核准知識的語彙正規化、平台限制、低信心與歧義拒答。

禁止從 runtime log、真實客服錄音或逐字稿直接複製資料到此目錄。

## 詞彙／混合檢索比較

先啟動提供 OpenAI 相容 `POST /v1/embeddings` 的本機服務，再指定待評測模型：

```bash
SVA_EMBEDDINGS_MODEL=<local-model-id> uv run python -m evals.run_hybrid
```

輸出會逐案比較詞彙與混合檢索結果。Embedding 服務無法使用時評測直接失敗，不會以詞彙回退結果冒充混合檢索成績；混合檢索必須通過全部案例且不得低於詞彙基準，否則以非零狀態結束。

## 受控答案生成

生成模型與獨立 judge 必須分開設定：

```bash
SVA_ANSWER_LLM_MODEL=<generator-model-id> \
SVA_ANSWER_JUDGE_MODEL=<independent-judge-model-id> \
uv run python -m evals.run_answer_generation
```

Runner 先完成全部生成，再執行確定性輸出守門與獨立 groundedness 審查；生成文字只暫存在記憶體，不寫入報表或檔案。任何新增事實、遺漏限制、禁止延伸、格式錯誤或模型故障都使評測失敗。

## 對話追問語意解析

使用合成的多輪對話，驗證新問題、局部追問、換句話說、指涉輪次及上下文查詢改寫：

```bash
SVA_CONVERSATION_LLM_MODEL=<local-model-id> \
uv run python -m evals.run_conversation_semantics
```

Runner 不保存模型輸出的查詢文字，只輸出分類正確數、延遲與失敗案例 ID。

## TTS 模型與分段效能

TTS runner 從已發布且允許回答的知識中，依短、中、長答案各選代表案例。輸出只包含 Knowledge ID、字數、延遲、音訊長度與波形統計，不保存答案或音訊內容。

```bash
uv run python -m evals.run_tts \
  --model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit \
  --model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit \
  --segmenter legacy \
  --segmenter selected_punctuation
```

runner 會先暖機每個模型，再比較舊版 42 字分段與新版 `，。？、`／換行 80/96 字分段。主要判斷指標是 `real_time_factor`；小於 1 代表生成速度快於音訊播放速度。
