import json
from dataclasses import replace

import httpx

from orchestrator.conversation import (
    ConversationContextStore,
    ConversationExchange,
    ConversationSemanticAssessment,
    ConversationSemanticResult,
    ConversationSemanticRoutingError,
    FollowUpKind,
    FollowUpResolver,
    OpenAICompatibleConversationSemanticAnalyzer,
    ReplyMode,
)


def exchange(
    user_utterance: str = "如何修改個人基本資料？",
    *,
    resolved_query: str | None = None,
    assistant_answer: str = "您可以依核准流程辦理。",
) -> ConversationExchange:
    return ConversationExchange(
        user_utterance=user_utterance,
        resolved_query=resolved_query or user_utterance,
        assistant_answer=assistant_answer,
        decision="answer",
        knowledge_id="K-FAQ-001",
        knowledge_version="1.0",
    )


def test_reply_mode_defaults_are_explicit() -> None:
    assert ReplyMode.EXACT.value == "exact"
    assert ReplyMode.NATURAL.value == "natural"


def test_context_store_isolates_conversations_and_bounds_history() -> None:
    store = ConversationContextStore(max_turns=2)
    store.append("call-a", exchange("第一題"))
    store.append("call-a", exchange("第二題"))
    store.append("call-a", exchange("第三題"))
    store.append("call-b", exchange("另一通電話"))

    assert [turn.user_utterance for turn in store.history("call-a")] == [
        "第二題",
        "第三題",
    ]
    assert [turn.user_utterance for turn in store.history("call-b")] == ["另一通電話"]


def test_context_store_default_retains_enough_turns_for_a_topic_anchor() -> None:
    store = ConversationContextStore()

    for index in range(6):
        store.append("call-a", exchange(f"第 {index + 1} 輪"))

    assert [turn.user_utterance for turn in store.history("call-a")] == [
        "第 1 輪",
        "第 2 輪",
        "第 3 輪",
        "第 4 輪",
        "第 5 輪",
        "第 6 輪",
    ]


def test_context_store_expires_and_clears_ephemeral_history() -> None:
    now = [10.0]
    store = ConversationContextStore(ttl_seconds=5, clock=lambda: now[0])
    store.append("call-a", exchange())

    now[0] = 16.0

    assert store.history("call-a") == ()
    store.append("call-a", exchange("重新開始"))
    store.clear("call-a")
    assert store.history("call-a") == ()


def test_context_store_evicts_oldest_session_at_capacity() -> None:
    now = [1.0]
    store = ConversationContextStore(
        max_conversations=2,
        clock=lambda: now[0],
    )
    store.append("call-a", exchange("A"))
    now[0] = 2.0
    store.append("call-b", exchange("B"))
    now[0] = 3.0
    store.append("call-c", exchange("C"))

    assert store.history("call-a") == ()
    assert store.history("call-b")
    assert store.history("call-c")


def test_follow_up_resolver_uses_recent_successful_turn_for_elaboration() -> None:
    history = (
        exchange("開戶需要什麼？"),
        exchange("如何修改個人基本資料？"),
    )

    resolution = FollowUpResolver().resolve(
        utterance="剛才臨櫃那一段可以再說詳細一點嗎？",
        history=history,
    )

    assert resolution.kind is FollowUpKind.ELABORATE
    assert "如何修改個人基本資料" in resolution.retrieval_query
    assert "臨櫃" in resolution.retrieval_query
    assert resolution.history == history


def test_follow_up_resolver_distinguishes_rephrase_and_new_question() -> None:
    history = (exchange(),)

    rephrase = FollowUpResolver().resolve(
        utterance="我沒聽清楚，可以換個方式說嗎？",
        history=history,
    )
    new_question = FollowUpResolver().resolve(
        utterance="什麼是台股定期定額？",
        history=history,
    )

    assert rephrase.kind is FollowUpKind.REPHRASE
    assert "如何修改個人基本資料" in rephrase.retrieval_query
    assert new_question.kind is FollowUpKind.NEW_QUESTION
    assert new_question.retrieval_query == "什麼是台股定期定額？"


def test_follow_up_resolver_connects_short_elliptical_document_question() -> None:
    resolution = FollowUpResolver().resolve(
        utterance="父母要帶什麼證件",
        history=(exchange("未成年怎麼開戶"),),
    )

    assert resolution.kind is FollowUpKind.ELABORATE
    assert "未成年怎麼開戶" in resolution.retrieval_query
    assert "父母要帶什麼證件" in resolution.retrieval_query
    assert resolution.reference_knowledge_id == "K-FAQ-001"


def test_follow_up_resolver_connects_child_attendance_question() -> None:
    resolution = FollowUpResolver().resolve(
        utterance="開戶時，小孩也要到場嗎?",
        history=(exchange("未成年怎麼開戶"),),
    )

    assert resolution.kind is FollowUpKind.ELABORATE
    assert "未成年怎麼開戶" in resolution.retrieval_query
    assert resolution.reference_knowledge_id == "K-FAQ-001"


def test_follow_up_resolver_prefers_specific_focus_over_rephrase_wording() -> None:
    resolution = FollowUpResolver().resolve(
        utterance="剛剛沒聽清楚，辨理的時間是幾點到幾點?",
        history=(exchange("註銷證券帳戶要怎麼辦理"),),
    )

    assert resolution.kind is FollowUpKind.ELABORATE
    assert "註銷證券帳戶要怎麼辦理" in resolution.retrieval_query


def test_follow_up_resolver_ignores_failed_turn_when_finding_reference() -> None:
    failed = replace(
        exchange("無法回答的問題"),
        decision="refuse",
        knowledge_id=None,
        knowledge_version=None,
    )

    resolution = FollowUpResolver().resolve(
        utterance="那費用呢？",
        history=(exchange(), failed),
    )

    assert resolution.kind is FollowUpKind.ELABORATE
    assert "如何修改個人基本資料" in resolution.retrieval_query


class StaticConversationSemanticAnalyzer:
    def __init__(
        self,
        assessment: ConversationSemanticAssessment | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self.assessment = assessment
        self.fail = fail
        self.calls: list[tuple[str, tuple[ConversationExchange, ...]]] = []

    def analyze(
        self,
        *,
        utterance: str,
        history: tuple[ConversationExchange, ...],
    ) -> ConversationSemanticResult:
        self.calls.append((utterance, history))
        if self.fail:
            raise ConversationSemanticRoutingError("synthetic failure")
        assert self.assessment is not None
        return ConversationSemanticResult(
            assessment=self.assessment,
            model_id="synthetic-semantic-model",
            prompt_version="conversation-semantic-v1",
            prompt_hash="d" * 64,
            latency_ms=8.0,
        )


def test_conversation_semantic_analyzer_uses_structured_bounded_context() -> None:
    history = (
        exchange("註銷證券帳戶要怎麼辦理"),
        exchange(
            "辦理時間是幾點到幾點",
            resolved_query="註銷證券帳戶辦理時間",
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        assert body["temperature"] == 0
        assert body["response_format"]["json_schema"]["strict"] is True
        payload = json.loads(body["messages"][1]["content"])
        assert payload["current_utterance"] == "若銷戶後3個月，可以再線上開戶嗎?"
        assert [turn["turn_id"] for turn in payload["recent_conversation"]] == [
            "T1",
            "T2",
        ]
        assert payload["recent_conversation"][1]["resolved_query"] == (
            "註銷證券帳戶辦理時間"
        )
        assert len(payload["recent_conversation"][0]["user"]) <= 300
        assert len(payload["recent_conversation"][0]["resolved_query"]) <= 500
        assert len(payload["recent_conversation"][0]["assistant"]) <= 800
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "kind": "elaborate",
                                    "reference_turn_id": "T2",
                                    "rewritten_query": (
                                        "證券帳戶銷戶後3個月是否可以再次線上開戶"
                                    ),
                                    "focus": "銷戶後重新開戶的限制期間",
                                    "confidence": 0.97,
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    analyzer = OpenAICompatibleConversationSemanticAnalyzer(
        base_url="http://llm.test/v1",
        model="synthetic-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = analyzer.analyze(
        utterance="若銷戶後3個月，可以再線上開戶嗎?",
        history=history,
    )

    assert result.assessment.kind is FollowUpKind.ELABORATE
    assert result.assessment.reference_turn_id == "T2"
    assert result.prompt_version == "conversation-semantic-v2"


def test_hybrid_follow_up_resolver_recovers_semantic_restriction_question() -> None:
    history = (
        exchange("註銷證券帳戶要怎麼辦理"),
        exchange(
            "辦理時間是幾點到幾點",
            resolved_query="註銷證券帳戶辦理時間",
        ),
    )
    analyzer = StaticConversationSemanticAnalyzer(
        ConversationSemanticAssessment(
            kind=FollowUpKind.ELABORATE,
            reference_turn_id="T2",
            rewritten_query="證券帳戶銷戶後3個月是否可以再次線上開戶",
            focus="銷戶後重新開戶的限制期間",
            confidence=0.96,
        )
    )

    resolution = FollowUpResolver(
        semantic_mode="controlled",
        semantic_analyzer=analyzer,
    ).resolve(
        utterance="若銷戶後3個月，可以再線上開戶嗎?",
        history=history,
    )

    assert resolution.kind is FollowUpKind.ELABORATE
    assert resolution.retrieval_query == "證券帳戶銷戶後3個月是否可以再次線上開戶"
    assert resolution.reference_knowledge_id == "K-FAQ-001"
    assert resolution.focus == "銷戶後重新開戶的限制期間"
    assert resolution.semantic_applied is True
    assert resolution.resolution_latency_ms is not None
    assert resolution.resolution_latency_ms >= 0
    assert resolution.semantic_latency_ms == 8.0


def test_semantic_resolver_preserves_same_knowledge_topic_anchor() -> None:
    anchor = exchange(
        "註銷證券帳戶要怎麼辦理？",
        assistant_answer="銷戶完成後6個月內無法線上開戶。",
    )
    history = (
        anchor,
        exchange("請問臨櫃辦理的時間"),
        exchange("線上申請銷戶要怎麼操作"),
        exchange("剛剛沒聽清楚，申辦時間是幾點到幾點"),
        exchange("線上申辦的時間是幾點到幾點"),
    )
    analyzer = StaticConversationSemanticAnalyzer(
        ConversationSemanticAssessment(
            kind=FollowUpKind.ELABORATE,
            reference_turn_id="T1",
            rewritten_query="銷戶後3個月是否可以再次線上開戶",
            focus="銷戶後重新開戶的限制期間",
            confidence=0.95,
        )
    )

    resolution = FollowUpResolver(
        semantic_mode="controlled",
        semantic_analyzer=analyzer,
    ).resolve(
        utterance="若銷戶後3個月，可以再線上開戶嗎?",
        history=history,
    )

    semantic_history = analyzer.calls[0][1]
    assert len(semantic_history) == 4
    assert semantic_history[0] is anchor
    assert resolution.kind is FollowUpKind.ELABORATE
    assert resolution.reference_knowledge_id == anchor.knowledge_id


def test_hybrid_follow_up_resolver_keeps_rules_on_shadow_or_semantic_failure() -> None:
    assessment = ConversationSemanticAssessment(
        kind=FollowUpKind.ELABORATE,
        reference_turn_id="T1",
        rewritten_query="上下文改寫問句",
        focus="限制期間",
        confidence=0.99,
    )
    history = (exchange("註銷證券帳戶要怎麼辦理"),)

    shadow = FollowUpResolver(
        semantic_mode="shadow",
        semantic_analyzer=StaticConversationSemanticAnalyzer(assessment),
    ).resolve(
        utterance="若過一季能在線上重辦嗎?",
        history=history,
    )
    failed = FollowUpResolver(
        semantic_mode="controlled",
        semantic_analyzer=StaticConversationSemanticAnalyzer(fail=True),
    ).resolve(
        utterance="若過一季能在線上重辦嗎?",
        history=history,
    )

    assert shadow.kind is FollowUpKind.NEW_QUESTION
    assert shadow.semantic_applied is False
    assert failed.kind is FollowUpKind.NEW_QUESTION
    assert failed.retrieval_query == "若過一季能在線上重辦嗎?"


def test_hybrid_follow_up_resolver_never_sends_sensitive_text_to_semantic_model() -> None:
    analyzer = StaticConversationSemanticAnalyzer(
        ConversationSemanticAssessment(
            kind=FollowUpKind.ELABORATE,
            reference_turn_id="T1",
            rewritten_query="不應使用",
            focus=None,
            confidence=1.0,
        )
    )

    resolution = FollowUpResolver(
        semantic_mode="controlled",
        semantic_analyzer=analyzer,
    ).resolve(
        utterance="我的驗證碼是123456，那可以再申請嗎?",
        history=(exchange("如何申請"),),
    )

    assert resolution.kind is FollowUpKind.NEW_QUESTION
    assert analyzer.calls == []
