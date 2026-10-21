"""Factory do checkpointer: os três backends e o erro de configuração."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from dealership_agent.config import settings
from dealership_agent.graph.builder import build_graph
from dealership_agent.graph.state import initial_state
from dealership_agent.llm import TemplateLLMClient
from dealership_agent.persistence.checkpointer import checkpointer_context


async def test_backend_memory(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_backend", "MEMORY")  # case-insensitive
    async with checkpointer_context() as saver:
        assert isinstance(saver, InMemorySaver)


async def test_backend_sqlite_persiste_entre_conexoes(monkeypatch, tmp_path):
    # Persistência real: reabre o arquivo e o perfil do lead continua lá.
    monkeypatch.setattr(settings, "checkpoint_backend", "sqlite")
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "ckpt.sqlite"))
    config = {"configurable": {"thread_id": "+5541911110000"}}

    async with checkpointer_context() as saver:
        assert isinstance(saver, BaseCheckpointSaver)
        graph = build_graph(saver, TemplateLLMClient())
        await graph.ainvoke(initial_state("Quero um SUV até 140 mil", "+5541911110000"), config)

    async with checkpointer_context() as saver:
        graph = build_graph(saver, TemplateLLMClient())
        snapshot = await graph.aget_state(config)
        assert snapshot.values["lead_profile"].body_type == "suv"


async def test_backend_postgres_desembrulha_dsn_e_roda_setup(monkeypatch):
    from langgraph.checkpoint.postgres import aio

    calls: dict = {}

    class _FakeSaver:
        async def setup(self):
            calls["setup"] = True

    @asynccontextmanager
    async def _from_conn_string(dsn):
        calls["dsn"] = dsn
        yield _FakeSaver()

    monkeypatch.setattr(aio.AsyncPostgresSaver, "from_conn_string", _from_conn_string)
    monkeypatch.setattr(settings, "checkpoint_backend", "postgres")
    monkeypatch.setattr(settings, "postgres_dsn", SecretStr("postgresql://u:s3cr3t@db/x"))

    async with checkpointer_context() as saver:
        assert isinstance(saver, _FakeSaver)
    # O driver recebe a string em claro; o SecretStr só protege repr/log.
    assert calls == {"dsn": "postgresql://u:s3cr3t@db/x", "setup": True}


async def test_backend_desconhecido_falha_explicitamente(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_backend", "redis")
    with pytest.raises(ValueError, match="redis"):
        async with checkpointer_context():
            pass
