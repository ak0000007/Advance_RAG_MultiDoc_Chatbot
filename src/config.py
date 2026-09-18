from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    langsmith_tracing: bool = True
    langsmith_api_key: str | None = None
    langsmith_project: str | None = None
    google_api_key: str | None = None
    deepseek_api_key: str | None = None
    openai_api_key: str | None = None
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
