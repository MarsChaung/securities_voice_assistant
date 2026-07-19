from functools import lru_cache
from typing import Literal

from pydantic import Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SVA_",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"
    database_url: str = (
        "postgresql+psycopg://sva:sva-dev-only@127.0.0.1:5433/securities_voice_assistant"
    )
    retrieval_minimum_score: float = 0.55
    retrieval_ambiguity_margin: float = 0.08
    retrieval_mode: Literal["lexical", "hybrid"] = "lexical"
    embeddings_base_url: HttpUrl = HttpUrl("http://127.0.0.1:12345/v1")
    embeddings_model: str | None = None
    embeddings_api_key: SecretStr | None = None
    embeddings_timeout_seconds: float = Field(default=5.0, gt=0)
    embeddings_query_prefix: str = ""
    embeddings_document_prefix: str = ""
    hybrid_retrieval_minimum_score: float = Field(default=0.4, ge=0, le=1)
    hybrid_retrieval_ambiguity_margin: float = Field(default=0.02, ge=0, le=1)
    knowledge_admin_url: HttpUrl = HttpUrl("http://127.0.0.1:8081/admin/knowledge")
    llm_base_url: HttpUrl = HttpUrl("http://127.0.0.1:12345/v1")
    llm_api_key: SecretStr | None = None
    answer_mode: Literal["exact", "controlled_llm", "fixed_message"] = "exact"
    answer_llm_model: str | None = None
    answer_llm_timeout_seconds: float = Field(default=8.0, gt=0)
    intent_router_mode: Literal["disabled", "shadow", "controlled"] = "disabled"
    intent_llm_model: str | None = None
    intent_llm_timeout_seconds: float = Field(default=8.0, gt=0)
    intent_router_minimum_confidence: float = Field(default=0.8, ge=0, le=1)
    tts_base_url: HttpUrl = HttpUrl("http://127.0.0.1:8000/v1")

    @model_validator(mode="after")
    def require_embedding_model_for_hybrid_mode(self) -> "Settings":
        if self.retrieval_mode == "hybrid" and not self.embeddings_model:
            raise ValueError("hybrid retrieval 必須設定 SVA_EMBEDDINGS_MODEL")
        if self.answer_mode == "controlled_llm" and not self.answer_llm_model:
            raise ValueError("controlled_llm answer mode 必須設定 SVA_ANSWER_LLM_MODEL")
        if self.intent_router_mode != "disabled" and not self.intent_llm_model:
            raise ValueError("啟用 intent router 必須設定 SVA_INTENT_LLM_MODEL")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
