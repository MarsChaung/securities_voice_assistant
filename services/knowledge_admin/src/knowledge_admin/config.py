from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class KnowledgeAdminSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SVA_",
        extra="ignore",
    )

    app_env: str = "development"
    database_url: str = (
        "postgresql+psycopg://sva:sva-dev-only@127.0.0.1:5433/securities_voice_assistant"
    )
    knowledge_admin_dev_identity_enabled: bool = True
    voice_test_url: HttpUrl = HttpUrl("http://127.0.0.1:8080/voice-test")

    def validate_identity_mode(self) -> None:
        if self.app_env != "development" and self.knowledge_admin_dev_identity_enabled:
            raise RuntimeError("開發身分模擬器只能在 development 環境啟用")
        if self.app_env == "production":
            raise RuntimeError("正式知識治理介面必須先整合公司身分提供者")
