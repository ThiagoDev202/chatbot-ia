"""Configuracao centralizada via variaveis de ambiente."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuracoes da aplicacao carregadas do ambiente ou arquivo .env."""

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.3

    langchain_tracing_v2: bool = True
    langchain_api_key: str | None = None
    langchain_project: str = "chatbot-ia"

    app_env: str = "dev"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """Retorna instancia unica de Settings (cache via lru_cache)."""
    return Settings()
