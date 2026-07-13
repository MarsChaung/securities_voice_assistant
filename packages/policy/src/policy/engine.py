from dataclasses import dataclass

from .models import PolicyAction, PolicyResult


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    intent: str
    action: PolicyAction
    keywords: tuple[str, ...]

    def matches(self, normalized_text: str) -> bool:
        return any(keyword.casefold() in normalized_text for keyword in self.keywords)


class DomainPolicyEngine:
    """Phase 0 核准前的保守規則集；未知或混合意圖一律不放行。"""

    _rules = (
        PolicyRule(
            "POL-HANDOFF-001",
            "complaint_or_dispute",
            PolicyAction.HANDOFF,
            ("申訴", "消費爭議", "權益受損", "轉人工", "客服人員"),
        ),
        PolicyRule(
            "POL-HANDOFF-002",
            "personal_data_change",
            PolicyAction.HANDOFF,
            ("修改個人資料", "更改個人資料", "變更手機", "變更地址"),
        ),
        PolicyRule(
            "POL-REFUSE-001",
            "transaction_request",
            PolicyAction.REFUSE,
            ("下單", "買進", "賣出", "改單", "刪單", "取消委託", "幫我買", "幫我賣"),
        ),
        PolicyRule(
            "POL-REFUSE-002",
            "personal_account_query",
            PolicyAction.REFUSE,
            ("我的庫存", "我的餘額", "我的成交", "我的交割", "我的帳務", "申請狀態"),
        ),
        PolicyRule(
            "POL-REFUSE-003",
            "investment_advice",
            PolicyAction.REFUSE,
            ("推薦股票", "推薦個股", "可以買嗎", "能不能買", "會漲嗎", "目標價", "買賣時點"),
        ),
        PolicyRule(
            "POL-REFUSE-004",
            "credential_or_identity_support",
            PolicyAction.REFUSE,
            ("忘記密碼", "重設密碼", "OTP", "驗證碼", "憑證", "裝置綁定"),
        ),
        PolicyRule(
            "POL-REFUSE-005",
            "rumor_or_non_public_information",
            PolicyAction.REFUSE,
            ("內線消息", "未公開資訊", "市場傳聞", "小道消息"),
        ),
        PolicyRule(
            "POL-ALLOW-001",
            "app_public_help",
            PolicyAction.ALLOW,
            (
                "下載 APP",
                "下載APP",
                "APP 要如何下載",
                "APP如何下載",
                "APP 怎麼下載",
                "APP怎麼下載",
                "更新 APP",
                "更新APP",
                "APP 操作",
                "APP功能",
                "APP 畫面",
                "閃退",
                "錯誤訊息",
            ),
        ),
        PolicyRule(
            "POL-ALLOW-002",
            "account_opening_general",
            PolicyAction.ALLOW,
            ("開戶流程", "如何開戶", "申請帳戶流程"),
        ),
        PolicyRule(
            "POL-ALLOW-003",
            "public_service_information",
            PolicyAction.ALLOW,
            ("服務時間", "公開費率", "手續費規則", "官方客服", "聯絡方式"),
        ),
        PolicyRule(
            "POL-ALLOW-004",
            "general_securities_knowledge",
            PolicyAction.ALLOW,
            ("什麼是", "是什麼", "名詞解釋"),
        ),
    )

    def classify(self, text: str) -> PolicyResult:
        normalized_text = text.casefold()
        matches = [rule for rule in self._rules if rule.matches(normalized_text)]

        handoff_matches = [rule for rule in matches if rule.action is PolicyAction.HANDOFF]
        if handoff_matches:
            return self._to_result(handoff_matches[0])

        refuse_matches = [rule for rule in matches if rule.action is PolicyAction.REFUSE]
        if refuse_matches:
            return self._to_result(refuse_matches[0])

        allow_matches = [rule for rule in matches if rule.action is PolicyAction.ALLOW]
        if len(allow_matches) == 1:
            return self._to_result(allow_matches[0])

        return PolicyResult(
            action=PolicyAction.REFUSE,
            intent="unknown_or_ambiguous",
            policy_rule_id="POL-DEFAULT-DENY",
            confidence=1.0,
        )

    @staticmethod
    def _to_result(rule: PolicyRule) -> PolicyResult:
        return PolicyResult(
            action=rule.action,
            intent=rule.intent,
            policy_rule_id=rule.rule_id,
            confidence=1.0,
        )
