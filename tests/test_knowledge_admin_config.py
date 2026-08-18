import pytest

from knowledge_admin.config import KnowledgeAdminSettings


def test_development_identity_cannot_run_in_non_development_environment() -> None:
    settings = KnowledgeAdminSettings(
        app_env="staging",
        knowledge_admin_dev_identity_enabled=True,
    )

    with pytest.raises(RuntimeError, match="development"):
        settings.validate_identity_mode()


def test_production_admin_refuses_to_start_without_company_identity_provider() -> None:
    settings = KnowledgeAdminSettings(
        app_env="production",
        knowledge_admin_dev_identity_enabled=False,
    )

    with pytest.raises(RuntimeError, match="公司身分提供者"):
        settings.validate_identity_mode()
