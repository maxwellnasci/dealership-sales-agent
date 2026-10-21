"""Factory do checkpointer (memória da conversa entre mensagens de WhatsApp).

Três backends, mesma interface (`BaseCheckpointSaver`), escolhidos por config:

  - memory  : InMemorySaver — testes (rápido, efêmero).
  - sqlite  : AsyncSqliteSaver — demo auto-contida, sem Postgres.
  - postgres: AsyncPostgresSaver — produção (via docker-compose).

Trocar de backend NÃO muda uma linha do grafo: `build_graph` só espera um
`BaseCheckpointSaver`. O `thread_id` (configurável no invoke) é o telefone do
lead — é isso que amarra cada conversa à sua própria memória persistida.

Serialização: o estado carrega modelos Pydantic (`LeadProfile`, `Vehicle`...).
O `JsonPlusSerializer` padrão aceita QUALQUER tipo do checkpoint (com aviso) —
quem escreve no banco poderia induzir a importação de classes arbitrárias. Aqui
a allowlist é explícita: só os tipos seguros do LangGraph e os nossos.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from dealership_agent.config import secret, settings
from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.schemas.tools import FinancingQuote, TestDriveSlot, Vehicle

# Tipos de domínio que trafegam no `GraphState` e vão para o checkpoint.
CHECKPOINT_TYPES: tuple[type, ...] = (LeadProfile, Vehicle, FinancingQuote, TestDriveSlot)


def checkpoint_serde() -> JsonPlusSerializer:
    """Serializer com allowlist explícita (sem aviso e sem tipo arbitrário)."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[(t.__module__, t.__name__) for t in CHECKPOINT_TYPES]
    )


@asynccontextmanager
async def checkpointer_context():
    """Abre o checkpointer conforme `settings.checkpoint_backend`.

    É um context manager porque as conexões (SQLite/Postgres) vivem durante
    toda a aplicação e devem ser abertas uma vez (lifespan do FastAPI), não
    por request.
    """
    backend = settings.checkpoint_backend.lower()

    if backend == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver(serde=checkpoint_serde())
        return

    if backend == "sqlite":
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        # `from_conn_string` não aceita serde; abrimos a conexão nós mesmos.
        async with aiosqlite.connect(settings.sqlite_path) as conn:
            yield AsyncSqliteSaver(conn, serde=checkpoint_serde())
        return

    if backend == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        dsn = secret(settings.postgres_dsn) or ""
        async with AsyncPostgresSaver.from_conn_string(dsn, serde=checkpoint_serde()) as saver:
            await saver.setup()  # cria as tabelas na primeira subida
            yield saver
        return

    raise ValueError(f"checkpoint_backend desconhecido: {settings.checkpoint_backend!r}")
