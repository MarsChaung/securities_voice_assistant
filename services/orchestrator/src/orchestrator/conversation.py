import hashlib
import json
import logging
import re
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from threading import Lock
from time import monotonic, perf_counter
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from policy import SensitiveDataGuard


class ReplyMode(StrEnum):
    EXACT = "exact"
    NATURAL = "natural"


class FollowUpKind(StrEnum):
    NEW_QUESTION = "new_question"
    ELABORATE = "elaborate"
    REPHRASE = "rephrase"


class ConversationSemanticMode(StrEnum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    CONTROLLED = "controlled"


@dataclass(frozen=True)
class ConversationExchange:
    user_utterance: str
    resolved_query: str
    assistant_answer: str
    decision: str
    knowledge_id: str | None
    knowledge_version: str | None


@dataclass(frozen=True)
class ConversationResolution:
    kind: FollowUpKind
    retrieval_query: str
    history: tuple[ConversationExchange, ...]
    reference_knowledge_id: str | None = None
    focus: str | None = None
    semantic_confidence: float | None = None
    semantic_applied: bool = False
    resolution_latency_ms: float | None = None
    semantic_latency_ms: float | None = None


class ConversationSemanticAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: FollowUpKind
    reference_turn_id: str | None = Field(default=None, pattern=r"^T[1-4]$")
    rewritten_query: str = Field(min_length=1, max_length=1_000)
    focus: str | None = Field(default=None, max_length=200)
    confidence: float = Field(ge=0, le=1)


@dataclass(frozen=True)
class ConversationSemanticResult:
    assessment: ConversationSemanticAssessment
    model_id: str
    prompt_version: str
    prompt_hash: str
    latency_ms: float


class ConversationSemanticRoutingError(RuntimeError):
    """對話語意解析失敗；訊息不得包含使用者內容或上游回應。"""


class ConversationSemanticAnalyzer(Protocol):
    def analyze(
        self,
        *,
        utterance: str,
        history: tuple[ConversationExchange, ...],
    ) -> ConversationSemanticResult: ...


class _ChatMessage(BaseModel):
    content: str


class _ChatChoice(BaseModel):
    message: _ChatMessage


class _ChatCompletion(BaseModel):
    choices: list[_ChatChoice]


class OpenAICompatibleConversationSemanticAnalyzer:
    PROMPT_VERSION = "conversation-semantic-v3"
    SYSTEM_PROMPT = """你是證券語音客服的對話語意解析器，只能解析，不得回答問題。
根據目前問句與最近對話，判斷目前問句是新問題、針對既有回答的局部追問，或要求換句話說。
不可只靠固定關鍵字；要理解省略主詞、代名詞、期間、條件及口語改述。
reference_turn_id 規則：
- kind 是 elaborate 或 rephrase 時，reference_turn_id 絕對不可為 null，必須從 recent_conversation
  實際存在的 turn_id 選一個；通常是最後一輪，除非問句明確指向更早內容。
- kind 是 new_question 時，reference_turn_id 必須為 null。
rewritten_query 必須把省略的主題補成可獨立檢索的完整問句，但不得新增使用者與歷史都沒有的事實。
focus 簡短描述本輪希望了解的局部資訊；若是新問題或沒有明確焦點可為 null。
相同 knowledge_id 的連續輪次代表同一個治理知識主題；目前問句若延伸詢問該主題的條件、
限制或後續處理，應判為 elaborate，不可只因最近一輪聚焦時間或操作步驟就判成新問題。
「那、那我、所以、如果、另外」等承接語本身不代表新問題。若目前問句詢問最近回答剛提到的
動作、管道或名詞，例如回答提到現股當沖需線上簽署，下一句問「那我想線上簽署，要怎麼操作」，
應判為 elaborate，並將省略的現股當沖主題補入 rewritten_query。
若無法可靠判斷，使用 new_question 並降低 confidence。只輸出符合 strict JSON schema 的物件。

範例：最近 T1 討論銷戶，使用者問「那三個月後能再線上辦嗎？」時，kind=elaborate、
reference_turn_id=T1，rewritten_query 要明確包含銷戶後重新線上開戶的問題。
"""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 8.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._prompt_hash = hashlib.sha256(self.SYSTEM_PROMPT.encode()).hexdigest()

    def analyze(
        self,
        *,
        utterance: str,
        history: tuple[ConversationExchange, ...],
    ) -> ConversationSemanticResult:
        recent_conversation = [
            {
                "turn_id": f"T{index + 1}",
                "user": exchange.user_utterance[:300],
                "resolved_query": exchange.resolved_query[:500],
                "assistant": exchange.assistant_answer[:800],
                "knowledge_id": exchange.knowledge_id,
            }
            for index, exchange in enumerate(history[-4:])
            if exchange.decision == "answer" and exchange.knowledge_id is not None
        ]
        payload = {
            "current_utterance": utterance,
            "recent_conversation": recent_conversation,
        }
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
                    "max_tokens": 300,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "conversation_semantic_resolution",
                            "strict": True,
                            "schema": ConversationSemanticAssessment.model_json_schema(),
                        },
                    },
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(payload, ensure_ascii=False),
                        },
                    ],
                },
            )
            response.raise_for_status()
            completion = _ChatCompletion.model_validate(response.json())
            if not completion.choices:
                raise ValueError("missing choice")
            assessment = ConversationSemanticAssessment.model_validate_json(
                completion.choices[0].message.content
            )
        except (httpx.HTTPError, json.JSONDecodeError, ValidationError, ValueError) as error:
            raise ConversationSemanticRoutingError(
                "conversation semantic routing failed"
            ) from error

        return ConversationSemanticResult(
            assessment=assessment,
            model_id=self._model,
            prompt_version=self.PROMPT_VERSION,
            prompt_hash=self._prompt_hash,
            latency_ms=(perf_counter() - started_at) * 1_000,
        )


@dataclass
class _StoredConversation:
    exchanges: deque[ConversationExchange]
    updated_at: float


class ConversationContextStore:
    def __init__(
        self,
        *,
        max_turns: int = 8,
        ttl_seconds: float = 30 * 60,
        max_conversations: int = 100,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_conversations < 1:
            raise ValueError("max_conversations must be positive")
        self._max_turns = max_turns
        self._ttl_seconds = ttl_seconds
        self._max_conversations = max_conversations
        self._clock = clock
        self._conversations: OrderedDict[str, _StoredConversation] = OrderedDict()
        self._lock = Lock()

    def history(self, conversation_id: str) -> tuple[ConversationExchange, ...]:
        with self._lock:
            now = self._clock()
            self._purge_expired(now)
            conversation = self._conversations.get(conversation_id)
            if conversation is None:
                return ()
            conversation.updated_at = now
            self._conversations.move_to_end(conversation_id)
            return tuple(conversation.exchanges)

    def append(self, conversation_id: str, exchange: ConversationExchange) -> None:
        with self._lock:
            now = self._clock()
            self._purge_expired(now)
            conversation = self._conversations.get(conversation_id)
            if conversation is None:
                self._evict_at_capacity()
                conversation = _StoredConversation(
                    exchanges=deque(maxlen=self._max_turns),
                    updated_at=now,
                )
                self._conversations[conversation_id] = conversation
            conversation.exchanges.append(exchange)
            conversation.updated_at = now
            self._conversations.move_to_end(conversation_id)

    def clear(self, conversation_id: str) -> None:
        with self._lock:
            self._conversations.pop(conversation_id, None)

    def _purge_expired(self, now: float) -> None:
        expired = [
            conversation_id
            for conversation_id, conversation in self._conversations.items()
            if now - conversation.updated_at > self._ttl_seconds
        ]
        for conversation_id in expired:
            del self._conversations[conversation_id]

    def _evict_at_capacity(self) -> None:
        while len(self._conversations) >= self._max_conversations:
            self._conversations.popitem(last=False)


class FollowUpResolver:
    _rephrase_pattern = re.compile(
        r"沒(?:有)?聽清楚|聽不清楚|沒聽懂|聽不懂|不太懂|"
        r"再說一次|重(?:新)?說|換個方式|換句話"
    )
    _elaboration_pattern = re.compile(
        r"剛才|剛剛|上一(?:段|個|題|輪)|那一(?:段|個|部分)|這一(?:段|個|部分)|"
        r"其中|這部分|那部分|詳細|多說|進一步|再說明|補充|"
        r"(?:費用|時間|步驟|資格|限制|原因|方式).{0,8}(?:呢|嗎|[？?])?$"
    )
    _elliptical_follow_up_pattern = re.compile(
        r"父母|法定代理人|證件|文件|材料|要帶|準備|"
        r"費用|時間|幾點|步驟|資格|限制|地點|哪裡|多久|幾歲|"
        r"到場|親臨|本人|親自|(?:小孩|未成年人).{0,6}(?:要去|要來|需要去|需要到)"
    )

    def __init__(
        self,
        *,
        max_history_turns: int = 8,
        semantic_mode: ConversationSemanticMode | str = ConversationSemanticMode.DISABLED,
        semantic_analyzer: ConversationSemanticAnalyzer | None = None,
        semantic_minimum_confidence: float = 0.85,
        sensitive_data_guard: SensitiveDataGuard | None = None,
    ) -> None:
        if max_history_turns < 1:
            raise ValueError("max_history_turns must be positive")
        if not 0 <= semantic_minimum_confidence <= 1:
            raise ValueError("semantic minimum confidence must be between 0 and 1")
        self._max_history_turns = max_history_turns
        self._semantic_mode = ConversationSemanticMode(semantic_mode)
        self._semantic_analyzer = semantic_analyzer
        self._semantic_minimum_confidence = semantic_minimum_confidence
        self._sensitive_data_guard = sensitive_data_guard or SensitiveDataGuard()
        self._logger = logging.getLogger("sva.conversation")
        if (
            self._semantic_mode is not ConversationSemanticMode.DISABLED
            and self._semantic_analyzer is None
        ):
            raise ValueError("enabled semantic mode requires a semantic analyzer")

    def resolve(
        self,
        *,
        utterance: str,
        history: Sequence[ConversationExchange],
    ) -> ConversationResolution:
        started_at = perf_counter()

        def traced(
            resolution: ConversationResolution,
            *,
            semantic_latency_ms: float | None = None,
        ) -> ConversationResolution:
            return replace(
                resolution,
                resolution_latency_ms=(perf_counter() - started_at) * 1_000,
                semantic_latency_ms=semantic_latency_ms,
            )

        recent_history = tuple(history[-self._max_history_turns :])
        reference = next(
            (
                exchange
                for exchange in reversed(recent_history)
                if exchange.decision == "answer" and exchange.knowledge_id is not None
            ),
            None,
        )
        if reference is None:
            return traced(
                ConversationResolution(
                    kind=FollowUpKind.NEW_QUESTION,
                    retrieval_query=utterance,
                    history=recent_history,
                    reference_knowledge_id=None,
                )
            )

        is_rephrase = self._rephrase_pattern.search(utterance) is not None
        has_specific_focus = (
            self._elliptical_follow_up_pattern.search(utterance) is not None
        )
        if is_rephrase:
            kind = (
                FollowUpKind.ELABORATE
                if has_specific_focus
                else FollowUpKind.REPHRASE
            )
        elif self._elaboration_pattern.search(utterance) or (
            len(utterance) <= 24 and has_specific_focus
        ):
            kind = FollowUpKind.ELABORATE
        else:
            kind = FollowUpKind.NEW_QUESTION

        deterministic = ConversationResolution(
            kind=kind,
            retrieval_query=(
                utterance
                if kind is FollowUpKind.NEW_QUESTION
                else f"{reference.resolved_query}；使用者追問：{utterance}"
            )[:1_000],
            history=recent_history,
            reference_knowledge_id=(
                reference.knowledge_id if kind is not FollowUpKind.NEW_QUESTION else None
            ),
        )
        if (
            kind is not FollowUpKind.NEW_QUESTION
            or self._semantic_mode is ConversationSemanticMode.DISABLED
            or self._sensitive_data_guard.scan(utterance).has_sensitive_data
        ):
            return traced(deterministic)

        assert self._semantic_analyzer is not None
        semantic_history = self._select_semantic_history(
            recent_history,
            reference=reference,
        )
        semantic_started_at = perf_counter()
        try:
            semantic = self._semantic_analyzer.analyze(
                utterance=utterance,
                history=semantic_history,
            )
        except ConversationSemanticRoutingError:
            self._logger.info(
                "conversation_semantic_resolution mode=%s applied=false reason=routing_error",
                self._semantic_mode.value,
            )
            return traced(
                deterministic,
                semantic_latency_ms=(perf_counter() - semantic_started_at) * 1_000,
            )

        assessment = semantic.assessment
        semantic_applied = (
            self._semantic_mode is ConversationSemanticMode.CONTROLLED
            and assessment.confidence >= self._semantic_minimum_confidence
            and assessment.kind is not FollowUpKind.NEW_QUESTION
            and assessment.reference_turn_id is not None
        )
        self._logger.info(
            "conversation_semantic_resolution mode=%s kind=%s confidence=%.3f applied=%s",
            self._semantic_mode.value,
            assessment.kind.value,
            assessment.confidence,
            semantic_applied,
        )
        if not semantic_applied:
            return traced(deterministic, semantic_latency_ms=semantic.latency_ms)

        assert assessment.reference_turn_id is not None
        reference_map = {
            f"T{index + 1}": exchange
            for index, exchange in enumerate(semantic_history)
            if exchange.decision == "answer" and exchange.knowledge_id is not None
        }
        semantic_reference = reference_map.get(assessment.reference_turn_id)
        if semantic_reference is None:
            return traced(deterministic, semantic_latency_ms=semantic.latency_ms)

        return traced(
            ConversationResolution(
                kind=assessment.kind,
                retrieval_query=assessment.rewritten_query,
                history=recent_history,
                reference_knowledge_id=semantic_reference.knowledge_id,
                focus=assessment.focus,
                semantic_confidence=assessment.confidence,
                semantic_applied=True,
            ),
            semantic_latency_ms=semantic.latency_ms,
        )

    @staticmethod
    def _select_semantic_history(
        history: tuple[ConversationExchange, ...],
        *,
        reference: ConversationExchange,
    ) -> tuple[ConversationExchange, ...]:
        selected = list(history[-4:])
        topic_anchor = next(
            (
                exchange
                for exchange in history
                if exchange.decision == "answer"
                and exchange.knowledge_id == reference.knowledge_id
            ),
            reference,
        )
        if all(exchange is not topic_anchor for exchange in selected):
            selected = [topic_anchor, *history[-3:]]
        return tuple(selected)
