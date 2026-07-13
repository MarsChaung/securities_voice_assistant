from functools import lru_cache

from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SVA_",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"
    llm_base_url: HttpUrl = HttpUrl("http://127.0.0.1:12345/v1")
    tts_base_url: HttpUrl = HttpUrl("http://127.0.0.1:8000/v1")


@lru_cache
def get_settings() -> Settings:
    return Settings()
