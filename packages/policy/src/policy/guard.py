import re
from dataclasses import dataclass
from re import Pattern

from .models import GuardResult


@dataclass(frozen=True)
class SensitivePattern:
    data_type: str
    pattern: Pattern[str]


class SensitiveDataGuard:
    """在文字送入 LLM 或知識服務前，以確定性規則攔截常見機敏資料。"""

    _patterns = (
        SensitivePattern(
            "taiwan_national_id",
            re.compile(r"(?<![A-Za-z0-9])[A-Z][12]\d{8}(?![A-Za-z0-9])", re.IGNORECASE),
        ),
        SensitivePattern(
            "otp",
            re.compile(r"(?:OTP|驗證碼|動態密碼)\s*(?:是|為|:|：)?\s*\d{4,8}", re.IGNORECASE),
        ),
        SensitivePattern(
            "password",
            re.compile(r"(?:密碼|password)\s*(?:是|為|:|：)\s*\S{4,}", re.IGNORECASE),
        ),
        SensitivePattern(
            "account_number",
            re.compile(
                r"(?:帳號|證券帳戶|銀行帳戶)\s*(?:是|為|:|：)?\s*[A-Za-z0-9-]{5,}",
                re.IGNORECASE,
            ),
        ),
        SensitivePattern(
            "email",
            re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])", re.IGNORECASE),
        ),
        SensitivePattern(
            "phone_number",
            re.compile(r"(?<!\d)(?:\+?886[- ]?)?0?9\d{2}[- ]?\d{3}[- ]?\d{3}(?!\d)"),
        ),
    )

    def scan(self, text: str) -> GuardResult:
        redacted_text = text
        detected_types: list[str] = []

        for item in self._patterns:
            if item.pattern.search(redacted_text):
                detected_types.append(item.data_type)
                redacted_text = item.pattern.sub("[REDACTED]", redacted_text)

        return GuardResult(
            has_sensitive_data=bool(detected_types),
            detected_types=detected_types,
            redacted_text=redacted_text,
        )
