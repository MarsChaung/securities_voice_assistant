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
            (
                "修改個人資料",
                "修改個人基本資料",
                "更改個人資料",
                "更改個人基本資料",
                "變更手機",
                "變更地址",
            ),
        ),
        PolicyRule(
            "POL-REFUSE-001",
            "transaction_request",
            PolicyAction.REFUSE,
            (
                "幫我下單",
                "替我下單",
                "請幫我下單",
                "我要下單",
                "立即下單",
                "現在下單",
                "下單買",
                "下單賣",
                "買進",
                "賣出",
                "改單",
                "刪單",
                "取消委託",
                "幫我買",
                "幫我賣",
            ),
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
            (
                "忘記密碼",
                "重設密碼",
                "補發密碼",
                "密碼補發",
                "OTP",
                "驗證碼",
                "憑證",
                "裝置綁定",
            ),
        ),
        PolicyRule(
            "POL-REFUSE-005",
            "rumor_or_non_public_information",
            PolicyAction.REFUSE,
            ("內線消息", "未公開資訊", "市場傳聞", "小道消息"),
        ),
        PolicyRule(
            "POL-REFUSE-006",
            "prompt_injection",
            PolicyAction.REFUSE,
            ("忽略上述", "忽略前述", "system prompt", "系統提示詞", "開發者訊息"),
        ),
        PolicyRule(
            "POL-REFUSE-007",
            "account_authorization_execution",
            PolicyAction.REFUSE,
            (
                "幫我直接把帳戶授權",
                "直接幫我把帳戶授權",
                "請直接把我的帳戶授權",
                "替我辦理帳戶授權",
                "代我辦理帳戶授權",
                "幫我完成帳戶授權",
                "立即把我的帳戶授權",
            ),
        ),
        PolicyRule(
            "POL-ALLOW-001",
            "app_public_help",
            PolicyAction.ALLOW,
            ("國泰證券 App", "國泰證券App", "樹精靈 App", "樹精靈App"),
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
            (
                "服務時間",
                "交易時間",
                "交易時段",
                "公開費率",
                "手續費規則",
                "官方客服",
                "聯絡方式",
            ),
        ),
        PolicyRule(
            "POL-ALLOW-004",
            "general_securities_knowledge",
            PolicyAction.ALLOW,
            (
                "什麼是",
                "是什麼",
                "名詞解釋",
                "有什麼不一樣",
                "有何不同",
                "有什麼不同",
                "差異在哪",
                "差在哪",
                "申請資格",
                "申請條件",
                "哪些人可以申請",
                "誰可以申請",
                "哪些投資人",
                "有哪些",
            ),
        ),
        PolicyRule(
            "POL-ALLOW-005",
            "order_entry_tutorial",
            PolicyAction.ALLOW,
            (
                "如何下單",
                "怎麼下單",
                "下單流程",
                "下單步驟",
                "下單操作",
                "下單畫面",
            ),
        ),
        PolicyRule(
            "POL-ALLOW-006",
            "web_public_help",
            PolicyAction.ALLOW,
            (
                "Web 版如何操作",
                "Web 版要如何操作",
                "Web 版怎麼操作",
                "Web 功能",
                "Web 畫面",
                "網頁版操作",
                "網站操作",
                "網頁錯誤訊息",
            ),
        ),
    )

    def classify(self, text: str) -> PolicyResult:
        normalized_text = text.casefold()
        matches = [rule for rule in self._rules if rule.matches(normalized_text)]

        handoff_matches = [rule for rule in matches if rule.action is PolicyAction.HANDOFF]
        mandatory_handoffs = [
            rule for rule in handoff_matches if rule.intent != "personal_data_change"
        ]
        if mandatory_handoffs:
            return self._to_result(mandatory_handoffs[0])

        refuse_matches = [rule for rule in matches if rule.action is PolicyAction.REFUSE]
        non_credential_refusals = [
            rule
            for rule in refuse_matches
            if rule.intent != "credential_or_identity_support"
        ]
        if non_credential_refusals:
            return self._to_result(non_credential_refusals[0])

        if self._is_public_personal_data_change_guidance(normalized_text):
            return PolicyResult(
                action=PolicyAction.ALLOW,
                intent="personal_data_change_guidance",
                policy_rule_id="POL-ALLOW-009",
                confidence=1.0,
            )

        if handoff_matches:
            return self._to_result(handoff_matches[0])

        if self._is_public_credential_recovery_guidance(normalized_text):
            return PolicyResult(
                action=PolicyAction.ALLOW,
                intent="credential_recovery_guidance",
                policy_rule_id="POL-ALLOW-007",
                confidence=1.0,
            )

        if refuse_matches:
            return self._to_result(refuse_matches[0])

        if self._is_public_account_authorization_guidance(normalized_text):
            return PolicyResult(
                action=PolicyAction.ALLOW,
                intent="account_authorization_guidance",
                policy_rule_id="POL-ALLOW-008",
                confidence=1.0,
            )

        allow_matches = [rule for rule in matches if rule.action is PolicyAction.ALLOW]
        if len(allow_matches) == 1:
            return self._to_result(allow_matches[0])

        specific_allow_matches = [
            rule for rule in allow_matches if rule.intent != "general_securities_knowledge"
        ]
        if len(specific_allow_matches) == 1:
            return self._to_result(specific_allow_matches[0])

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

    @staticmethod
    def _is_public_credential_recovery_guidance(normalized_text: str) -> bool:
        if any(
            phrase in normalized_text
            for phrase in (
                "幫我補發",
                "替我補發",
                "直接補發",
                "幫我重設",
                "替我重設",
                "直接重設",
                "幫我變更",
                "替我變更",
                "憑證",
                "裝置綁定",
            )
        ):
            return False

        asks_for_password_reissue = any(
            phrase in normalized_text for phrase in ("補發密碼", "密碼補發")
        )
        phone_changed = any(
            phrase in normalized_text
            for phrase in (
                "手機變",
                "換手機",
                "手機號碼更改",
                "電話號碼更改",
                "行動電話已改",
                "行動電話已變更",
            )
        )
        cannot_receive_code = any(
            phrase in normalized_text
            for phrase in (
                "無法拿到驗證碼",
                "無法取得驗證碼",
                "無法收到驗證碼",
            )
        )
        forgot_password = "忘記密碼" in normalized_text
        return asks_for_password_reissue or (
            phone_changed and (cannot_receive_code or forgot_password)
        )

    @staticmethod
    def _is_public_account_authorization_guidance(normalized_text: str) -> bool:
        return any(
            phrase in normalized_text
            for phrase in (
                "帳戶授權",
                "帳戶委任",
                "委任授權",
                "委任帳戶",
                "授權他人買賣",
                "授權給別人",
                "多帳號授權",
                "多賬戶授權",
            )
        )

    @staticmethod
    def _is_public_personal_data_change_guidance(normalized_text: str) -> bool:
        if any(
            phrase in normalized_text
            for phrase in (
                "幫我",
                "替我",
                "代我",
                "直接修改",
                "直接更改",
                "直接變更",
            )
        ):
            return False

        asks_for_instructions = any(
            phrase in normalized_text
            for phrase in ("如何", "怎麼", "怎樣", "哪裡", "方式", "流程", "步驟")
        )
        concerns_personal_data_change = any(
            phrase in normalized_text
            for phrase in (
                "修改個人基本資料",
                "更改個人基本資料",
                "變更個人基本資料",
                "修改個人資料",
                "更改個人資料",
                "變更個人資料",
                "個資變更",
            )
        )
        return asks_for_instructions and concerns_personal_data_change
