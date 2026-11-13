from __future__ import annotations

import json
from functools import lru_cache
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    google_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    retrieval_confidence_threshold: float = 0.7
    retrieval_min_confident_docs: int = 3
    max_retrieval_attempts: int = 2
    postgres_url: str | None = None
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    bm25_store_path: str = "./data/bm25_index.pkl"

    # Salesforce JWT
    sf_client_id: str | None = None
    sf_login_url: str = "https://login.salesforce.com"
    sf_private_key_path: str | None = None
    sf_domain: str = "https://your-domain.my.salesforce.com"

    # API Authentication & Security
    api_secret_key: SecretStr | None = None
    sf_default_username: str | None = None
    api_key_mapping: dict[str, dict[str, str]] = {}

    # Demo & Mock CRM Mode (run without Salesforce credentials)
    mock_salesforce: bool = False

    # Write safety caps
    # Set WRITES_ENABLED=false in .env to strip ALL write tools at startup (read-only mode).
    writes_enabled: bool = True
    # Max Salesforce write operations allowed per conversation session.
    max_writes_per_session: int = 5

    # Multi-agent architecture toggle: routes through orchestrator + 3 specialist subgraphs
    use_multi_agent_architecture: bool = False

    # CORS Configuration (Debt H9)
    cors_origins: list[str] = ["*"]

    # Observability & Tracing (P1: LangSmith, Langfuse, OpenTelemetry)
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "advance-rag-multidoc-chatbot"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "advance-rag-chatbot"

    @field_validator("langsmith_tracing", "langfuse_enabled", "otel_enabled", "mock_salesforce", mode="before")
    @classmethod
    def parse_bool_env(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "on")
        return bool(v)

    @field_validator("api_key_mapping", mode="before")
    @classmethod
    def parse_api_key_mapping(cls, v):
        if isinstance(v, str):
            v_str = v.strip()
            if v_str:
                try:
                    return json.loads(v_str)
                except Exception:
                    pass
        return v or {}

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            v_str = v.strip()
            if v_str.startswith("["):
                try:
                    return json.loads(v_str)
                except Exception:
                    pass
            return [origin.strip() for origin in v_str.split(",") if origin.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
