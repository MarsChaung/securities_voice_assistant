import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .structured_output import (
    StructuredOutputMode,
    structured_output_content,
    structured_output_options,
)


class IntentRouterMode(StrEnum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    CONTROLLED = "controlled"


AllowedIntent = Literal[
    "app_public_help",
    "account_opening_general",
    "account_closure_general",
    "public_service_information",
    "general_securities_knowledge",
    "order_entry_tutorial",
    "web_public_help",
    "unknown",
]

RiskFlag = Literal[
    "transaction_execution",
    "personal_account_or_status",
    "investment_advice",
    "credential_or_sensitive_data",
    "complaint_or_dispute",
    "out_of_scope",
]


class IntentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_intents: list[AllowedIntent] = Field(min_length=1, max_length=2)
    confidence: float = Field(ge=0, le=1)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    needs_clarification: bool


@dataclass(frozen=True)
class IntentRouteResult:
    classification: IntentClassification
    model_id: str
    prompt_version: str
    prompt_hash: str
    latency_ms: float


class IntentRoutingError(RuntimeError):
    """意圖路由失敗；訊息不得包含使用者問題或上游回應。"""


class IntentRouter(Protocol):
    def route(self, question: str) -> IntentRouteResult: ...


class OpenAICompatibleIntentRouter:
    PROMPT_VERSION = "intent-router-v4"
    SYSTEM_PROMPT = """你是證券知識助手的意圖分類器，只能分類，不得回答問題。
根據整句語意分類，不可只靠固定關鍵字。只輸出指定 JSON 物件。

可用意圖：
- account_opening_general：證券帳戶、美股交易帳戶、複委託帳戶的開立、申請或加開流程與一般資格。
- account_closure_general：證券帳戶、帳券帳戶的註銷、銷戶、停用流程，以及銷戶後重新開戶的公開規則。
- public_service_information：每日交易時段、營業或服務時間、公開費率、手續費、客服與聯絡方式。
  商品規則何時生效、扣款日遇休市如何處理，不屬於此類。
- general_securities_knowledge：證券名詞、差異、交割、商品規則與一般概念；不包含上述更具體意圖。
- order_entry_tutorial：詢問如何操作下單介面或下單步驟，但未要求實際執行交易。
- app_public_help：問題明確提到國泰證券 App、樹精靈 App 或 App 畫面的公開操作教學。
  沒有提到 App 的一般定期投資問題，不屬於此類。
- web_public_help：Web 或網頁版的公開操作教學。
- unknown：無法歸類、資訊不足或不在上述範圍。

風險旗標：
- transaction_execution：要求助手現在替使用者實際買進、賣出、下單、改單或取消委託。
- personal_account_or_status：查詢個人庫存、餘額、成交、交割、帳務或申請狀態。
- investment_advice：要求推薦、預測、目標價、買賣時點，或詢問某商品「可以買嗎」「值得買嗎」。
- credential_or_sensitive_data：涉及密碼、驗證碼、憑證、帳號或個人資料。
- complaint_or_dispute：申訴、爭議、權益受損或要求人工。
- out_of_scope：與公開證券知識及操作教學無關。

candidate_intents 依可能性排序，最多兩個。若有風險、語意不完整或無法可靠分類，
使用 unknown 或 needs_clarification=true，不得為了提高回答率而猜測。
詢問帳戶註銷、銷戶或停用的公開辦理流程屬 account_closure_general，
不等於查詢個人帳戶狀態，不得標記 personal_account_or_status。
詢問 App 如何設定定期投資或如何操作下單介面屬公開教學，不等於要求助手執行交易，
不得標記 transaction_execution。只有要求「幫我／替我」實際執行交易才標記該風險。

範例：
-「國泰證券 App 要怎麼設定固定每月買台股？」=> app_public_help，無風險。
-「美股定期投資有哪些扣款作法？」=> general_securities_knowledge，無風險。
-「扣款日遇到股市休市如何處理？」=> general_securities_knowledge，無風險。
-「註銷證券帳戶要怎麼辦理？」=> account_closure_general，無風險。
-「請幫我下單買進範例公司」=> unknown，transaction_execution。
-「美股定期投資可以買嗎？」=> unknown，investment_advice。

輸出格式：
{"candidate_intents":["..."],"confidence":0.0,"risk_flags":[],"needs_clarification":false}
"""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 8.0,
        max_tokens: int = 768,
        structured_output_mode: StructuredOutputMode = "auto",
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._structured_output_mode = structured_output_mode
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._prompt_hash = hashlib.sha256(self.SYSTEM_PROMPT.encode()).hexdigest()

    def route(self, question: str) -> IntentRouteResult:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        started_at = perf_counter()
        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self._model,
                    "temperature": 0,
                    "max_tokens": self._max_tokens,
                    **structured_output_options(
                        name="intent_classification",
                        schema=IntentClassification.model_json_schema(),
                        mode=self._structured_output_mode,
                        model=self._model,
                    ),
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"question": question},
                                ensure_ascii=False,
                            ),
                        },
                    ],
                },
            )
            response.raise_for_status()
            classification = IntentClassification.model_validate_json(
                structured_output_content(
                    response.json(),
                    name="intent_classification",
                    mode=self._structured_output_mode,
                    model=self._model,
                )
            )
        except (httpx.HTTPError, json.JSONDecodeError, ValidationError, ValueError) as error:
            raise IntentRoutingError("structured intent routing failed") from error

        return IntentRouteResult(
            classification=classification,
            model_id=self._model,
            prompt_version=self.PROMPT_VERSION,
            prompt_hash=self._prompt_hash,
            latency_ms=(perf_counter() - started_at) * 1_000,
        )
