from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "agentic-ai"
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
