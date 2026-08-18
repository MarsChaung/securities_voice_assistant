import re
from collections.abc import Sequence
from dataclasses import dataclass

from .models import KnowledgeDocument, QuestionVariantUsage, RetrievalMatch

_NON_SEARCH_CHARACTERS = re.compile(r"[^0-9a-z\u3400-\u9fff]")
_QUERY_REPLACEMENTS = (
    ("如果我想", ""),
    ("定期投資", "定期定額"),
    ("股利自動再投入", "股息再投資"),
    ("股利再次投入", "股息再投資"),
    ("股利再投入", "股息再投資"),
    ("有什麼不一樣", "差別"),
    ("有哪裡不一樣", "差別"),
    ("有什麼不同", "差別"),
    ("有何不同", "差別"),
    ("差異在哪裡", "差別"),
    ("差異在哪", "差別"),
    ("差在哪裡", "差別"),
    ("差在哪", "差別"),
    ("有什麼差別", "差別"),
    ("不一樣", "差別"),
    ("差異", "差別"),
    ("跟", "與"),
    ("和", "與"),
    ("要如何操作", "步驟"),
    ("如何操作", "步驟"),
    ("怎麼操作", "步驟"),
    ("操作方式", "步驟"),
    ("是什麼", ""),
    ("什麼是", ""),
    ("請說明", ""),
    ("一般", ""),
)
_CANONICAL_REPLACEMENTS = (
    ("甚麼", "什麼"),
    ("賬戶", "帳戶"),
    ("帳號", "帳戶"),
    ("臺", "台"),
)
_DOMAIN_TERMS = ("定期定額", "股息再投資", "交割帳戶", "美股", "台股")
_PLATFORM_TERMS = ("國泰證券app", "樹精靈app", "樹精靈", "web", "網頁版")
_CONTACT_PHONE_DOCUMENT_TERMS = ("電話", "專線", "市話", "請撥")


@dataclass(frozen=True)
class LexicalKnowledgeRetriever:
    minimum_score: float = 0.55
    ambiguity_margin: float = 0.08

    def search(
        self,
        *,
        query: str,
        intent: str,
        documents: Sequence[KnowledgeDocument],
    ) -> RetrievalMatch | None:
        ranked = self.rank(query=query, intent=intent, documents=documents)
        return self.select(ranked)

    def rank(
        self,
        *,
        query: str,
        intent: str,
        documents: Sequence[KnowledgeDocument],
    ) -> tuple[RetrievalMatch, ...]:
        target_platform = _target_platform(query)
        candidates = [
            document
            for document in documents
            if _matches_intent(document, intent)
            and _matches_platform(document, target_platform)
            and _matches_query_constraints(query, document)
        ]
        return tuple(
            sorted(
                (
                    RetrievalMatch(document=document, score=_score(query, document))
                    for document in candidates
                ),
                key=lambda match: (-match.score, match.document.item.knowledge_id),
            )
        )

    def select(self, ranked: Sequence[RetrievalMatch]) -> RetrievalMatch | None:
        if not ranked or ranked[0].score < self.minimum_score:
            return None
        if len(ranked) > 1 and ranked[0].score - ranked[1].score < self.ambiguity_margin:
            return None
        return ranked[0]


def _matches_intent(document: KnowledgeDocument, intent: str) -> bool:
    allowed_intents = document.item.allowed_intents
    return intent in allowed_intents or "faq_general_guidance" in allowed_intents


def _score(query: str, document: KnowledgeDocument) -> float:
    normalized_query = _normalize_query(query)
    normalized_title = _normalize(document.item.title)
    normalized_products = [_normalize(product) for product in document.item.products]
    normalized_locator = _normalize(document.item.source_locator)
    base_candidate = normalized_title + "".join(normalized_products) + normalized_locator
    normalized_variants = [
        _normalize(variant.question_text)
        for variant in document.item.question_variants
        if variant.usage is QuestionVariantUsage.RETRIEVAL
    ]
    candidates = [base_candidate, *normalized_variants]
    combined_candidate = "".join(candidates)

    query_bigrams = _bigrams(normalized_query)
    overlap = 0.0
    if query_bigrams:
        overlap = max(
            (
                2
                * len(query_bigrams & candidate_bigrams)
                / (len(query_bigrams) + len(candidate_bigrams))
                for candidate in candidates
                if (candidate_bigrams := _bigrams(candidate))
            ),
            default=0.0,
        )

    product_bonus = (
        0.2
        if any(product and product in normalized_query for product in normalized_products)
        else 0
    )
    title_bonus = (
        0.15
        if normalized_query
        and any(
            normalized_query in candidate or candidate in normalized_query
            for candidate in (normalized_title, *normalized_variants)
        )
        else 0
    )
    domain_bonus = _shared_term_bonus(
        normalized_query,
        combined_candidate,
        terms=_DOMAIN_TERMS,
        bonus=0.1,
    )
    platform_bonus = _shared_term_bonus(
        normalized_query,
        combined_candidate,
        terms=_PLATFORM_TERMS,
        bonus=0.1,
    )
    intent_cue_bonus = _intent_cue_bonus(query, normalized_title)
    return round(
        min(
            1.0,
            overlap * 0.55
            + product_bonus
            + title_bonus
            + domain_bonus
            + platform_bonus
            + intent_cue_bonus,
        ),
        4,
    )


def _target_platform(query: str) -> str | None:
    normalized = _normalize(query)
    mentions_web = any(token in normalized for token in ("web", "網頁版", "網站"))
    mentions_app = any(token in normalized for token in ("app", "樹精靈"))
    if mentions_web and mentions_app:
        return "ambiguous"
    if not mentions_web and not mentions_app:
        return None
    return "web" if mentions_web else "app"


def _matches_platform(document: KnowledgeDocument, target_platform: str | None) -> bool:
    if target_platform is None:
        return True
    platforms = {platform.casefold() for platform in document.item.platforms}
    if target_platform == "web":
        return "web" in platforms
    if target_platform == "app":
        return bool({"ios", "android"} & platforms)
    return False


def _matches_query_constraints(query: str, document: KnowledgeDocument) -> bool:
    normalized_query = _normalize(query)
    asks_for_contact_phone = (
        "專線" in normalized_query
        or "電話號碼" in normalized_query
        or any(
            phrase in normalized_query
            for phrase in (
                "接單中心電話",
                "接單電話",
                "客服電話",
                "聯絡電話",
            )
        )
        or (
            "電話" in normalized_query
            and any(
                phrase in normalized_query
                for phrase in ("電話是多少", "電話是什麼", "電話幾號", "電話哪支")
            )
        )
    )
    if not asks_for_contact_phone:
        return True

    searchable_text = _normalize(
        "\n".join(
            (
                document.item.title,
                document.item.standard_answer,
                *(
                    variant.question_text
                    for variant in document.item.question_variants
                    if variant.usage is QuestionVariantUsage.RETRIEVAL
                ),
            )
        )
    )
    return any(term in searchable_text for term in _CONTACT_PHONE_DOCUMENT_TERMS)


def _normalize_query(value: str) -> str:
    normalized = value.casefold()
    for source, replacement in _QUERY_REPLACEMENTS:
        normalized = normalized.replace(source, replacement)
    return _normalize(normalized)


def _normalize(value: str) -> str:
    normalized = value.casefold()
    for source, replacement in _CANONICAL_REPLACEMENTS:
        normalized = normalized.replace(source, replacement)
    return _NON_SEARCH_CHARACTERS.sub("", normalized).replace("的", "").replace("與", "")


def _bigrams(value: str) -> set[str]:
    if len(value) < 2:
        return {value} if value else set()
    return {value[index : index + 2] for index in range(len(value) - 1)}


def _shared_term_bonus(
    query: str,
    candidate: str,
    *,
    terms: tuple[str, ...],
    bonus: float,
) -> float:
    return bonus if any(term in query and term in candidate for term in terms) else 0


def _intent_cue_bonus(query: str, title: str) -> float:
    normalized_query = query.casefold()
    asks_for_concept = any(
        phrase in normalized_query
        for phrase in (
            "什麼是",
            "是什麼",
            "不一樣",
            "不同",
            "差異",
            "差別",
            "差在哪",
        )
    )
    is_concept = any(phrase in title for phrase in ("基本概念", "差別"))
    asks_for_steps = any(phrase in normalized_query for phrase in ("操作", "步驟", "怎麼", "如何"))
    is_tutorial = any(phrase in title for phrase in ("步驟", "設定", "找到", "開啟"))
    return 0.15 if (asks_for_concept and is_concept) or (asks_for_steps and is_tutorial) else 0
