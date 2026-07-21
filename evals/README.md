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
