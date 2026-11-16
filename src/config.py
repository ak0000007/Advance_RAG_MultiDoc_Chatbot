from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal, Any
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    google_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    use_reranker: bool = False
    retrieval_confidence_threshold: float = 0.7
    retrieval_min_score: float | None = None
    retrieval_min_confident_docs: int = 3
    max_retrieval_attempts: int = 2
    crag_retry_irrelevance_threshold: float = 0.8
    citation_verification_mode: str = "flag"
    postgres_url: str | None = None
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    bm25_store_path: str = "./data/bm25_index.pkl"

    # Salesforce JWT
    sf_client_id: str | None = None
    sf_login_url: str = "https://login.salesforce.com"
    sf_private_key_path: str | None = None
    sf_domain: str = "https://your-domain.my.salesforce.com"

    # Salesforce Record-Level Access Control & Short-Lived User JWT Auth
    sf_auth_enabled: bool = False
    sf_jwt_public_key: SecretStr | None = None
    sf_jwt_public_key_path: str | None = None
    sf_jwt_audience: str = "advance-rag-travel-copilot"
    sf_jwt_issuer: str = "https://login.salesforce.com"
    sf_jwt_max_lifetime_seconds: int = 120
    sf_access_cache_ttl_seconds: int = 60
    sf_access_overfetch_factor: int = 4

    # API Authentication & Security
    api_secret_key: SecretStr | None = None
    sf_default_username: str | None = None
    api_key_mapping: dict[str, Any] = {}

    # LLM Provider, Deployment Topology & BYOK Credential Isolation
    llm_provider: str = "google"
    llm_model: str | None = None
    deployment_mode: Literal["solo", "hosted_byok", "hosted_shared"] = "solo"
    allow_operator_fallback: bool = True
    tenant_quota_daily_tokens: int | None = None
    tenant_quota_monthly_tokens: int | None = None

    # Environment & Developer Mode
    environment: Literal["development", "production"] = "development"
    dev_mode: bool = False

    # Thread Ownership & Access Control (P0-1)
    allow_unowned_threads: bool = False
    approver_roles: list[str] = []

    # CRM Module toggle (defaults to True if SF credentials / mock configured, else False)
    crm_enabled: bool | None = None

    # Demo & Mock CRM Mode (run without Salesforce credentials)
    mock_salesforce: bool = False

    # Write safety caps
    # Set WRITES_ENABLED=false in .env to strip ALL write tools at startup (read-only mode).
    writes_enabled: bool = True
    # Max Salesforce write operations allowed per conversation session.
    max_writes_per_session: int = 5

    # Memory Management (bounded working context for LLM invocation)
    max_messages: int = 50
    max_message_tokens: int | None = None

    # Rate Limiting & Cost Governance
    rate_limit_enabled: bool = False
    rate_limit_requests_per_minute: int = 60
    max_streaming_tokens: int = 4096

    # Multi-agent architecture toggle: routes through orchestrator + 3 specialist subgraphs
    use_multi_agent_architecture: bool = False

    # CORS Configuration (Debt H9: Restricted default instead of wildcard)
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000", "http://localhost:5173"]

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

    @property
    def is_crm_active(self) -> bool:
        if self.crm_enabled is not None:
            return self.crm_enabled
        return bool(self.sf_client_id and self.sf_private_key_path) or self.mock_salesforce

    @property
    def operator_fallback_allowed(self) -> bool:
        if self.deployment_mode == "hosted_byok":
            return False
        return self.allow_operator_fallback

    @property
    def global_llm_keys_present(self) -> bool:
        return any([
            bool(self.google_api_key.get_secret_value() if self.google_api_key else False),
            bool(self.openai_api_key.get_secret_value() if self.openai_api_key else False),
            bool(self.deepseek_api_key.get_secret_value() if self.deepseek_api_key else False),
            bool(self.openrouter_api_key.get_secret_value() if self.openrouter_api_key else False),
        ])

    @field_validator(
        "langsmith_tracing",
        "langfuse_enabled",
        "otel_enabled",
        "mock_salesforce",
        "sf_auth_enabled",
        "use_reranker",
        "dev_mode",
        "allow_unowned_threads",
        "rate_limit_enabled",
        "allow_operator_fallback",
        mode="before",
    )
    @classmethod
    def parse_bool_env(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "on")
        return bool(v)

    @field_validator("crm_enabled", mode="before")
    @classmethod
    def parse_optional_bool_env(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            v_clean = v.strip().lower()
            if v_clean in ("true", "1", "yes", "on"):
                return True
            if v_clean in ("false", "0", "no", "off"):
                return False
            return None
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

    @field_validator("cors_origins", "approver_roles", mode="before")
    @classmethod
    def parse_list_origins(cls, v):
        if isinstance(v, str):
            v_str = v.strip()
            if v_str.startswith("["):
                try:
                    return json.loads(v_str)
                except Exception:
                    pass
            return [item.strip() for item in v_str.split(",") if item.strip()]
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
