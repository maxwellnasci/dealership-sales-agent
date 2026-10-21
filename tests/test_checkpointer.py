"""Factory do checkpointer: os três backends e o erro de configuração."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
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
    async def _from_conn_string(dsn, *, serde=None):
        calls["dsn"] = dsn
        calls["serde"] = serde
        yield _FakeSaver()

    monkeypatch.setattr(aio.AsyncPostgresSaver, "from_conn_string", _from_conn_string)
    monkeypatch.setattr(settings, "checkpoint_backend", "postgres")
    monkeypatch.setattr(settings, "postgres_dsn", SecretStr("postgresql://u:s3cr3t@db/x"))

    async with checkpointer_context() as saver:
        assert isinstance(saver, _FakeSaver)
    # O driver recebe a string em claro; o SecretStr só protege repr/log.
    assert calls["dsn"] == "postgresql://u:s3cr3t@db/x"
    assert calls["setup"] is True
    assert isinstance(calls["serde"], JsonPlusSerializer)


async def test_backend_desconhecido_falha_explicitamente(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_backend", "redis")
    with pytest.raises(ValueError, match="redis"):
        async with checkpointer_context():
            pass


async def test_serde_reidrata_tipos_de_dominio_sem_aviso(caplog):
    # Round-trip de cada tipo do estado: volta como o modelo Pydantic, sem o
    # aviso "Deserializing unregistered type".
    from datetime import datetime

    from dealership_agent.persistence.checkpointer import checkpoint_serde
    from dealership_agent.schemas.lead import LeadProfile
    from dealership_agent.schemas.tools import FinancingQuote, TestDriveSlot, Vehicle

    vehicle = Vehicle(
        vehicle_id="V001", make="Toyota", model="Corolla", year=2022,
        price=129900.0, mileage_km=38000, body_type="sedan", color="prata",
    )
    values = [
        LeadProfile(name="Ana", body_type="suv"),
        vehicle,
        FinancingQuote(
            vehicle_id="V001", vehicle_label="Toyota Corolla 2022", price=129900.0,
            down_payment=25980.0, term_months=48, monthly_rate=0.019,
            monthly_payment=3200.0, total_paid=179580.0,
        ),
        TestDriveSlot(
            slot_id="s", vehicle_id="V001", vehicle_label="Toyota Corolla 2022",
            starts_at=datetime(2026, 10, 19, 9, 0),
        ),
    ]
    serde = checkpoint_serde()
    for value in values:
        assert serde.loads_typed(serde.dumps_typed(value)) == value
    assert "unregistered type" not in caplog.text


async def test_serde_bloqueia_tipo_fora_da_allowlist(caplog):
    from pydantic import BaseModel

    from dealership_agent.persistence.checkpointer import checkpoint_serde

    class Intruso(BaseModel):
        x: int = 1

    serde = checkpoint_serde()
    restored = serde.loads_typed(serde.dumps_typed(Intruso()))
    assert not isinstance(restored, Intruso)
