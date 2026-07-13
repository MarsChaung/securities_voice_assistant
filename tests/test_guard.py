import pytest

from policy import SensitiveDataGuard


@pytest.mark.parametrize(
    ("text", "expected_type", "secret"),
    [
        ("我的身分證是 A123456789", "taiwan_national_id", "A123456789"),
        ("驗證碼 123456", "otp", "123456"),
        ("密碼是 abcD1234", "password", "abcD1234"),
        ("我的 Email 是 user@example.com", "email", "user@example.com"),
        ("手機 0912-345-678", "phone_number", "0912-345-678"),
    ],
)
def test_sensitive_data_is_detected_and_redacted(
    text: str,
    expected_type: str,
    secret: str,
) -> None:
    result = SensitiveDataGuard().scan(text)

    assert result.has_sensitive_data
    assert expected_type in result.detected_types
    assert secret not in result.redacted_text


def test_public_question_passes_guard() -> None:
    result = SensitiveDataGuard().scan("APP 要如何下載？")

    assert not result.has_sensitive_data
    assert result.redacted_text == "APP 要如何下載？"
