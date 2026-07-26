import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from pypinyin import Style, lazy_pinyin

from retrieval import (
    KnowledgeDocument,
    QuestionVariantUsage,
    RetrievalMatch,
)

_NON_CJK = re.compile(r"[^\u3400-\u9fff]")
_QUESTION_NOISE = (
    "阿發請問",
    "我想知道",
    "想請問",
    "麻煩說明",
    "請說明",
    "說明一下",
    "請問",
    "什麼是",
    "是什麼",
    "有什麼意思",
    "意思是什麼",
)
_TITLE_SUFFIXES = (
    "的一般說明",
    "的一般流程",
    "的一般步驟",
    "的基本概念",
    "維持率計算方式",
    "一般說明",
    "一般流程",
    "一般步驟",
    "基本概念",
    "方式說明",
    "說明",
)


@dataclass(frozen=True)
class PhoneticResolution:
    match: RetrievalMatch | None = None
    candidates: tuple[RetrievalMatch, ...] = ()

    @property
    def ambiguous(self) -> bool:
        return self.match is None and len(self.candidates) > 1


class MandarinPhoneticResolver:
    """以核准知識詞彙召回中文 ASR 音近字；不進行自由文字改寫。"""

    def __init__(self, *, minimum_score: float = 0.9, ambiguity_margin: float = 0.08) -> None:
        if not 0 <= minimum_score <= 1 or not 0 <= ambiguity_margin <= 1:
            raise ValueError("音近檢索門檻必須介於 0 與 1")
        self._minimum_score = minimum_score
        self._ambiguity_margin = ambiguity_margin

    def resolve(
        self,
        *,
        query: str,
        intent: str,
        documents: Sequence[KnowledgeDocument],
    ) -> PhoneticResolution:
        query_core = _core_text(query)
        if not 3 <= len(query_core) <= 24:
            return PhoneticResolution()
        query_key = _phonetic_key(query_core)
        if not query_key:
            return PhoneticResolution()

        ranked = tuple(
            sorted(
                (
                    RetrievalMatch(
                        document=document,
                        score=round(
                            max(
                                (
                                    SequenceMatcher(
                                        None,
                                        query_key,
                                        candidate_key,
                                    ).ratio()
                                    for text in _document_terms(document)
                                    if (candidate_key := _phonetic_key(_core_text(text)))
                                ),
                                default=0.0,
                            ),
                            4,
                        ),
                    )
                    for document in documents
                    if _matches_intent(document, intent)
                ),
                key=lambda candidate: (
                    -candidate.score,
                    candidate.document.item.knowledge_id,
                ),
            )
        )
        if not ranked or ranked[0].score < self._minimum_score:
            return PhoneticResolution()
        if (
            len(ranked) > 1
            and ranked[1].score >= self._minimum_score
            and ranked[0].score - ranked[1].score < self._ambiguity_margin
        ):
            return PhoneticResolution(candidates=ranked[:2])
        return PhoneticResolution(match=ranked[0], candidates=ranked[:1])


def build_asr_context(
    documents: Sequence[KnowledgeDocument],
    *,
    max_chars: int = 512,
) -> str:
    """從已通過 Runtime 資格檢查的知識產生精簡 ASR 領域詞彙。"""
    if max_chars <= 0:
        return ""
    terms: list[str] = []
    seen: set[str] = set()
    for document in documents:
        title = _strip_title_suffix(document.item.title)
        for term in (title, *document.item.products):
            normalized = " ".join(term.split()).strip("「」")
            if len(normalized) < 2 or normalized in seen:
                continue
            proposed = "、".join((*terms, normalized))
            if len(proposed) > max_chars:
                return "、".join(terms)
            terms.append(normalized)
            seen.add(normalized)
    return "、".join(terms)


def _document_terms(document: KnowledgeDocument) -> tuple[str, ...]:
    return (
        _strip_title_suffix(document.item.title),
        *(
            variant.question_text
            for variant in document.item.question_variants
            if variant.usage is QuestionVariantUsage.RETRIEVAL
        ),
    )


def _core_text(value: str) -> str:
    normalized = _NON_CJK.sub("", value)
    for phrase in _QUESTION_NOISE:
        normalized = normalized.replace(phrase, "")
    return _strip_title_suffix(normalized)


def _strip_title_suffix(value: str) -> str:
    normalized = value.strip().strip("「」")
    for suffix in _TITLE_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
            return normalized[: -len(suffix)]
    return normalized


def _phonetic_key(value: str) -> str:
    return "".join(lazy_pinyin(value, style=Style.NORMAL, errors="ignore"))


def _matches_intent(document: KnowledgeDocument, intent: str) -> bool:
    allowed_intents = document.item.allowed_intents
    return intent in allowed_intents or "faq_general_guidance" in allowed_intents
