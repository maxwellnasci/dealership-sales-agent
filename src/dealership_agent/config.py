"""Configuração central (pydantic-settings).

Lê de variáveis de ambiente / `.env`. Nada de segredo hard-coded. Os valores
default são seguros para desenvolvimento local e para a demo auto-contida.

Credenciais são `SecretStr`: não vazam em `repr`, logs, tracebacks nem em
`model_dump()`. O valor só é lido explicitamente via `secret()`, no ponto exato
em que vai para o cliente HTTP.
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM --------------------------------------------------------------
    # Compatível com qualquer endpoint OpenAI-like (OpenAI, DeepSeek, Ollama,
    # etc.). A ausência de `openai_api_key` NÃO quebra os testes (eles usam o
    # cliente fake) nem a demo (ela cai num modo de fallback gracioso).
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    # Sem timeout explícito, um endpoint travado segura o webhook do WhatsApp
    # indefinidamente. Retries cobrem só falhas transitórias (SDK OpenAI).
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2

    # --- Roteador Jev (TypeSafe AI via OpenRouter, opcional) --------------
    # Sem `openrouter_api_key`, o roteador Jev delega tudo ao roteador
    # clássico (fallback padrão). Com a chave, classifica via endpoint de
    # decisões do OpenRouter. Ver graph/router_jev.py.
    openrouter_api_key: SecretStr | None = None
    jev_model: str = "typesafe/jev-1.14"
    jev_timeout_s: float = 5.0

    # --- Checkpointing / memória -----------------------------------------
    # memory  -> InMemorySaver (testes)
    # sqlite  -> AsyncSqliteSaver em arquivo (demo auto-contida, sem Postgres)
    # postgres-> AsyncPostgresSaver (produção, via docker-compose)
    checkpoint_backend: str = "sqlite"
    sqlite_path: str = "./data/checkpoints.sqlite"
    postgres_dsn: SecretStr = SecretStr("postgresql://dealer:dealer@localhost:5432/dealership")

    # --- Observabilidade (Langfuse, opcional) ----------------------------
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Negócio ----------------------------------------------------------
    # Nome do próprio assistente (usado na persona das respostas).
    agent_name: str = "Léo"
    # Operador humano que recebe o transbordo (human-in-the-loop).
    sales_rep_name: str = "o vendedor de plantão"
    # Número máximo de veículos sugeridos por resposta.
    max_vehicle_suggestions: int = 3

    @property
    def llm_configured(self) -> bool:
        """True quando há credencial para falar com um LLM real."""
        return bool(secret(self.openai_api_key))

    @property
    def jev_configured(self) -> bool:
        """True quando o roteador Jev (OpenRouter) pode ser usado."""
        return bool(secret(self.openrouter_api_key))


def secret(value: SecretStr | None) -> str | None:
    """Valor em claro de um `SecretStr`, ou None se ausente/vazio (`KEY=` no .env)."""
    if value is None:
        return None
    return value.get_secret_value() or None


settings = Settings()
