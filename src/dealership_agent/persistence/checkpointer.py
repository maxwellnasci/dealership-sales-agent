"""Factory do checkpointer (memória da conversa entre mensagens de WhatsApp).

Três backends, mesma interface (`BaseCheckpointSaver`), escolhidos por config:

  - memory  : InMemorySaver — testes (rápido, efêmero).
  - sqlite  : AsyncSqliteSaver — demo auto-contida, sem Postgres.
  - postgres: AsyncPostgresSaver — produção (via docker-compose).

Trocar de backend NÃO muda uma linha do grafo: `build_graph` só espera um
`BaseCheckpointSaver`. O `thread_id` (configurável no invoke) é o telefone do
lead — é isso que amarra cada conversa à sua própria memória persistida.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from dealership_agent.config import secret, settings


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

        yield InMemorySaver()
        return

    if backend == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(settings.sqlite_path) as saver:
            yield saver
        return

    if backend == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        dsn = secret(settings.postgres_dsn) or ""
        async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
            await saver.setup()  # cria as tabelas na primeira subida
            yield saver
        return

    raise ValueError(f"checkpoint_backend desconhecido: {settings.checkpoint_backend!r}")
