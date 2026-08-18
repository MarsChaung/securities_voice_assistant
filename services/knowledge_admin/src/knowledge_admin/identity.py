from dataclasses import dataclass

from .governance import GovernanceActor, KnowledgeRole


@dataclass(frozen=True)
class DevelopmentIdentityProvider:
    """Local-only identities used to exercise governance before SSO integration."""

    actors: tuple[GovernanceActor, ...] = (
        GovernanceActor(
            actor_id="Codex-assisted draft import",
            roles=frozenset({KnowledgeRole.AUTHOR}),
        ),
        GovernanceActor(
            actor_id="reviewer.dev",
            roles=frozenset({KnowledgeRole.REVIEWER}),
        ),
        GovernanceActor(
            actor_id="approver.dev",
            roles=frozenset({KnowledgeRole.APPROVER}),
        ),
        GovernanceActor(
            actor_id="publisher.dev",
            roles=frozenset({KnowledgeRole.PUBLISHER}),
        ),
        GovernanceActor(
            actor_id="revoker.dev",
            roles=frozenset({KnowledgeRole.REVOKER}),
        ),
    )

    def get_actor(self, actor_id: str) -> GovernanceActor:
        actor = next(
            (candidate for candidate in self.actors if candidate.actor_id == actor_id),
            None,
        )
        if actor is None:
            raise ValueError("未知的開發身分")
        return actor

    def actors_for(self, role: KnowledgeRole) -> tuple[GovernanceActor, ...]:
        return tuple(actor for actor in self.actors if role in actor.roles)
