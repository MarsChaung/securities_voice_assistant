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
    system_diagnostics_enabled: bool = False
    system_diagnostics_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
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
    knowledge_admin_internal_url: HttpUrl | None = None
    llm_base_url: HttpUrl = HttpUrl("http://127.0.0.1:12345/v1")
    llm_api_key: SecretStr | None = None
    llm_structured_output_mode: Literal["auto", "json_schema", "tool_call"] = "auto"
    answer_mode: Literal["exact", "shadow_llm", "controlled_llm", "fixed_message"] = "exact"
    answer_llm_model: str | None = None
    answer_llm_timeout_seconds: float = Field(default=8.0, gt=0)
    answer_llm_max_tokens: int = Field(default=768, ge=128, le=4096)
    natural_answer_enabled: bool = False
    voice_test_content_logging_enabled: bool = False
    shadow_max_pending: int = Field(default=8, ge=1, le=100)
    answer_judge_model: str | None = None
    answer_judge_timeout_seconds: float = Field(default=15.0, gt=0)
    intent_router_mode: Literal["disabled", "shadow", "controlled"] = "disabled"
    intent_llm_model: str | None = None
    intent_llm_timeout_seconds: float = Field(default=8.0, gt=0)
    intent_llm_max_tokens: int = Field(default=768, ge=128, le=4096)
    intent_router_minimum_confidence: float = Field(default=0.8, ge=0, le=1)
    conversation_semantic_mode: Literal["disabled", "shadow", "controlled"] = "disabled"
    conversation_llm_model: str | None = None
    conversation_llm_timeout_seconds: float = Field(default=8.0, gt=0)
    conversation_llm_max_tokens: int = Field(default=768, ge=128, le=4096)
    conversation_semantic_minimum_confidence: float = Field(default=0.85, ge=0, le=1)
    voice_enabled: bool = False
    tts_base_url: HttpUrl = HttpUrl("http://127.0.0.1:8000/v1")
    audio_public_base_url: HttpUrl = HttpUrl("http://127.0.0.1:8000/v1")
    asr_model: str | None = None
    asr_candidate_model: str | None = None
    tts_model: str | None = None
    tts_voice: str = "Vivian"
    tts_ref_audio: str | None = None
    tts_ref_text: SecretStr | None = None
    voice_timeout_seconds: float = Field(default=180.0, gt=0)
    voice_acknowledgement_delay_ms: int = Field(default=450, ge=100, le=5000)
    asr_endpoint_grace_ms: int = Field(default=1200, ge=0, le=5000)
    barge_in_enabled: bool = True
    barge_in_default_mode: Literal["sensitive", "standard", "resistant"] = "standard"

    @model_validator(mode="after")
    def require_embedding_model_for_hybrid_mode(self) -> "Settings":
        if self.retrieval_mode == "hybrid" and not self.embeddings_model:
            raise ValueError("hybrid retrieval 必須設定 SVA_EMBEDDINGS_MODEL")
        if self.answer_mode in {"shadow_llm", "controlled_llm"} and not self.answer_llm_model:
            raise ValueError("LLM answer mode 必須設定 SVA_ANSWER_LLM_MODEL")
        if self.natural_answer_enabled and not self.answer_llm_model:
            raise ValueError("自然對話模式必須設定 SVA_ANSWER_LLM_MODEL")
        if self.intent_router_mode != "disabled" and not self.intent_llm_model:
            raise ValueError("啟用 intent router 必須設定 SVA_INTENT_LLM_MODEL")
        if self.conversation_semantic_mode != "disabled" and not self.conversation_llm_model:
            raise ValueError(
                "啟用 conversation semantic resolver 必須設定 SVA_CONVERSATION_LLM_MODEL"
            )
        if self.voice_enabled and (not self.asr_model or not self.tts_model):
            raise ValueError("啟用語音功能必須設定 SVA_ASR_MODEL 與 SVA_TTS_MODEL")
        if bool(self.tts_ref_audio) != bool(self.tts_ref_text):
            raise ValueError("voice clone 必須同時設定 SVA_TTS_REF_AUDIO 與 SVA_TTS_REF_TEXT")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
