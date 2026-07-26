from dataclasses import dataclass
from enum import StrEnum

from retrieval import KnowledgeItem, KnowledgeStatus


class KnowledgeRole(StrEnum):
    AUTHOR = "author"
    REVIEWER = "reviewer"
    APPROVER = "approver"
    PUBLISHER = "publisher"
    REVOKER = "revoker"


class GovernanceAction(StrEnum):
    UPDATE_CONTENT = "update_content"
    UPDATE_QUESTION_VARIANTS = "update_question_variants"
    START_REVISION = "start_revision"
    SUBMIT_REVIEW = "submit_review"
    COMPLETE_REVIEW = "complete_review"
    APPROVE = "approve"
    PUBLISH = "publish"
    RETURN_DRAFT = "return_draft"
    REVOKE = "revoke"


@dataclass(frozen=True)
class GovernanceActor:
    actor_id: str
    roles: frozenset[KnowledgeRole]


class GovernanceError(ValueError):
    """Raised when an actor attempts a disallowed knowledge transition."""


class GovernancePolicy:
    _TRANSITIONS = {
        (KnowledgeStatus.DRAFT, GovernanceAction.UPDATE_CONTENT): KnowledgeStatus.DRAFT,
        (
            KnowledgeStatus.DRAFT,
            GovernanceAction.UPDATE_QUESTION_VARIANTS,
        ): KnowledgeStatus.DRAFT,
        (KnowledgeStatus.PUBLISHED, GovernanceAction.START_REVISION): KnowledgeStatus.DRAFT,
        (KnowledgeStatus.DRAFT, GovernanceAction.SUBMIT_REVIEW): KnowledgeStatus.REVIEW,
        (KnowledgeStatus.REVIEW, GovernanceAction.COMPLETE_REVIEW): KnowledgeStatus.REVIEW,
        (KnowledgeStatus.REVIEW, GovernanceAction.APPROVE): KnowledgeStatus.APPROVED,
        (KnowledgeStatus.APPROVED, GovernanceAction.PUBLISH): KnowledgeStatus.PUBLISHED,
        (KnowledgeStatus.REVIEW, GovernanceAction.RETURN_DRAFT): KnowledgeStatus.DRAFT,
        (KnowledgeStatus.APPROVED, GovernanceAction.RETURN_DRAFT): KnowledgeStatus.DRAFT,
        (KnowledgeStatus.APPROVED, GovernanceAction.REVOKE): KnowledgeStatus.REVOKED,
        (KnowledgeStatus.PUBLISHED, GovernanceAction.REVOKE): KnowledgeStatus.REVOKED,
    }

    _ACTION_ROLES = {
        GovernanceAction.UPDATE_CONTENT: frozenset({KnowledgeRole.AUTHOR}),
        GovernanceAction.UPDATE_QUESTION_VARIANTS: frozenset({KnowledgeRole.AUTHOR}),
        GovernanceAction.START_REVISION: frozenset({KnowledgeRole.AUTHOR}),
        GovernanceAction.SUBMIT_REVIEW: frozenset({KnowledgeRole.AUTHOR}),
        GovernanceAction.COMPLETE_REVIEW: frozenset({KnowledgeRole.REVIEWER}),
        GovernanceAction.APPROVE: frozenset({KnowledgeRole.APPROVER}),
        GovernanceAction.PUBLISH: frozenset({KnowledgeRole.PUBLISHER}),
        GovernanceAction.RETURN_DRAFT: frozenset({KnowledgeRole.REVIEWER, KnowledgeRole.APPROVER}),
        GovernanceAction.REVOKE: frozenset({KnowledgeRole.REVOKER}),
    }

    @classmethod
    def next_status(
        cls,
        item: KnowledgeItem,
        action: GovernanceAction,
        actor: GovernanceActor,
    ) -> KnowledgeStatus:
        next_status = cls._TRANSITIONS.get((item.status, action))
        if next_status is None:
            raise GovernanceError(f"{item.status.value} 狀態不允許執行 {action.value}")

        required_roles = cls._ACTION_ROLES[action]
        if actor.roles.isdisjoint(required_roles):
            raise GovernanceError(f"執行 {action.value} 缺少必要角色")

        cls._enforce_separation_of_duties(item, action, actor)
        return next_status

    @staticmethod
    def _enforce_separation_of_duties(
        item: KnowledgeItem,
        action: GovernanceAction,
        actor: GovernanceActor,
    ) -> None:
        if action is GovernanceAction.SUBMIT_REVIEW and actor.actor_id != item.author:
            raise GovernanceError("只有原作者可以送交審核")

        if action in {
            GovernanceAction.UPDATE_CONTENT,
            GovernanceAction.UPDATE_QUESTION_VARIANTS,
        } and actor.actor_id != item.author:
            raise GovernanceError("只有原作者可以編輯知識草稿")

        if action is GovernanceAction.START_REVISION and actor.actor_id != item.author:
            raise GovernanceError("只有原作者可以建立複審新版")

        if action is GovernanceAction.COMPLETE_REVIEW:
            if actor.actor_id == item.author:
                raise GovernanceError("作者不得審核自己的知識")
            if item.reviewer is not None:
                raise GovernanceError("此知識已完成審核")

        if action is GovernanceAction.APPROVE:
            if item.reviewer is None:
                raise GovernanceError("核准前必須指定審核人")
            if actor.actor_id in {item.author, item.reviewer}:
                raise GovernanceError("作者、審核人與核准人不得兼任")

        if action is GovernanceAction.PUBLISH:
            prior_actors = {item.author, item.reviewer, item.approver}
            if actor.actor_id in prior_actors:
                raise GovernanceError("發布人必須獨立於作者、審核人與核准人")

        if action is GovernanceAction.RETURN_DRAFT:
            assigned_actor = (
                item.reviewer if item.status is KnowledgeStatus.REVIEW else item.approver
            )
            if assigned_actor is not None and actor.actor_id != assigned_actor:
                raise GovernanceError("只有目前階段的負責人可以退回草稿")
