"""API FastAPI (webhook de WhatsApp): lifespan, turno, handoff e resume.

Roda o app de verdade (lifespan + grafo compilado) com checkpointer em memória
e o LLM determinístico — sem rede, sem chave.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dealership_agent import main
from dealership_agent.config import settings
from dealership_agent.llm import TemplateLLMClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_backend", "memory")
    monkeypatch.setattr(main, "get_llm_client", TemplateLLMClient)
    monkeypatch.setattr(main, "get_langfuse_handler", lambda: None)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_webhook_devolve_resposta_do_agente(client):
    resp = client.post("/webhook", json={"phone": "+5541900000001", "text": "Me mostra SUVs"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["interrupted"] is False
    assert body["interrupt_payload"] is None
    assert "R$" in body["reply"]


def test_webhook_mantem_memoria_por_telefone(client):
    phone = "+5541900000002"
    client.post("/webhook", json={"phone": phone, "text": "Quero um SUV até 140 mil"})
    body = client.post("/webhook", json={"phone": phone, "text": "e o financiamento?"}).json()
    # Turno 2 não repetiu o perfil, mas a simulação saiu (memória do turno 1).
    assert "parcela" in body["reply"]


def test_webhook_handoff_interrompe_e_resume_retoma(client):
    phone = "+5541900000003"
    body = client.post(
        "/webhook", json={"phone": phone, "text": "Quero falar com um vendedor"}
    ).json()
    assert body["interrupted"] is True
    assert body["reply"] is None
    assert body["interrupt_payload"]["reason"] == "handoff_requested"

    rep = "Oi! Aqui é o Carlos, posso te ligar?"
    resumed = client.post("/resume", json={"phone": phone, "rep_text": rep}).json()
    assert resumed == {"reply": rep, "interrupted": False, "interrupt_payload": None}


def test_webhook_valida_payload(client):
    assert client.post("/webhook", json={"phone": "+55"}).status_code == 422


def test_run_config_amarra_thread_ao_telefone_e_anexa_tracing(monkeypatch):
    monkeypatch.setattr(main, "get_langfuse_handler", lambda: None)
    assert main._run_config("+55") == {"configurable": {"thread_id": "+55"}}

    handler = object()
    monkeypatch.setattr(main, "get_langfuse_handler", lambda: handler)
    assert main._run_config("+55")["callbacks"] == [handler]
