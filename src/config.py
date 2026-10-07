from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    google_api_key: str | None = None
    deepseek_api_key: str | None = None
    openai_api_key: str | None = None
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    retrieval_confidence_threshold: float = 0.7
    retrieval_min_confident_docs: int = 3
    max_retrieval_attempts: int = 2
    postgres_url: str | None = None
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    bm25_store_path: str = "./data/bm25_index.pkl"

    # Salesforce JWT
    sf_client_id: str | None = None
    sf_login_url: str = "https://login.salesforce.com"
    sf_private_key_path: str | None = None
    sf_domain: str = "https://your-domain.my.salesforce.com"

    # API Authentication & Security
    api_secret_key: str | None = None
    sf_default_username: str | None = None

    # Write safety caps
    # Set WRITES_ENABLED=false in .env to strip ALL write tools at startup (read-only mode).
    writes_enabled: bool = True
    # Max Salesforce write operations allowed per conversation session.
    max_writes_per_session: int = 5

    # Multi-agent architecture toggle: routes through orchestrator + 3 specialist subgraphs
    use_multi_agent_architecture: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
